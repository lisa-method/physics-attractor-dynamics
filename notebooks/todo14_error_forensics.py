"""TODO 14 companion: forensic audit of TODO 11 selector errors.

This block does not fit a model or tune a threshold.  It reconstructs the
outer-LOSO decisions that were already made in TODO 11, links them to the
complete-state material failures, and asks whether the remaining mistakes are
mostly low-confidence threshold cases or feature-space collisions consistent
with missing state.
"""


TODO14_MARGIN_BANDS = (0.05, 0.10)
TODO14_NEIGHBOR_CONTROL_PER_CLASS = 64


def _todo14_p1_trace():
    """Recover active-before, candidate and threshold for the TODO 11 scan."""
    _active_before = np.zeros(len(todo9_event_dataset), dtype=bool)
    _candidate = np.zeros(len(todo9_event_dataset), dtype=bool)
    _threshold = np.full(len(todo9_event_dataset), np.nan, dtype=float)
    _prediction = np.zeros(len(todo9_event_dataset), dtype=bool)

    for _indices in _todo11_sequence_indices:
        _active = False
        for _event_index in _indices:
            _active_before[_event_index] = _active
            if _todo9_loop_prediction[_event_index]:
                _active = True
                continue
            _candidate[_event_index] = bool(
                _todo11_base_candidate[_event_index] or _active
            )
            if not _candidate[_event_index]:
                _active = False
                continue
            _segment_id = int(
                todo9_event_dataset.at[_event_index, "segment_id"]
            )
            _enter, _stay = todo11_threshold_by_segment[_segment_id]
            _threshold[_event_index] = _stay if _active else _enter
            _active = bool(
                _todo11_outer_probability[_event_index]
                >= _threshold[_event_index]
            )
            _prediction[_event_index] = _active

    assert np.array_equal(_prediction, _todo11_event_prediction)
    return _active_before, _candidate, _threshold


(
    _todo14_p1_active_before_all,
    _todo14_p1_candidate_all,
    _todo14_p1_threshold_all,
) = _todo14_p1_trace()

_todo14_transition_count = len(_todo9_transition_indices)
_todo14_p2_active_before = np.zeros(_todo14_transition_count, dtype=bool)
_todo14_p2_candidate = np.zeros(_todo14_transition_count, dtype=bool)
_todo14_p2_threshold = np.full(
    _todo14_transition_count, np.nan, dtype=float
)
_todo14_p2_trace_prediction = np.zeros(
    _todo14_transition_count, dtype=bool
)
_todo14_p2_records = [None] * _todo14_transition_count
_todo14_p2_vectors = np.full(
    (_todo14_transition_count, len(TODO9_FEATURE_COLUMNS)),
    np.nan,
    dtype=np.float32,
)
_todo14_p2_active = {
    int(_segment_id): False for _segment_id in _todo11_segments
}

# Object 2 must be reconstructed with the distance predicted after object 1.
# Using the original same-row event features here would silently audit a
# different model from the sequential one that produced the 49 failures.
for _transition_row, (_data_index, _event_index, _driver_distance) in enumerate(
    zip(
        _todo9_transition_indices,
        _todo9_p2_event_indices,
        _todo11_distance_prediction,
    )
):
    _event = todo9_event_dataset.loc[_event_index]
    _segment_id = int(_event["segment_id"])
    _active = _todo14_p2_active[_segment_id]
    _todo14_p2_active_before[_transition_row] = _active

    if bool(_event["predict_period2_from_previous"]):
        _todo14_p2_active[_segment_id] = True
        continue

    _baseline = _todo8_update_object(
        todo7_p2[_data_index - 1],
        todo7_angle2[_data_index - 1],
        float(_driver_distance),
        -1.0,
    )
    _record = _todo9_feature_record(
        _event_index, _baseline, float(_driver_distance)
    )
    _todo14_p2_records[_transition_row] = _record
    _todo14_p2_vectors[_transition_row] = _todo10_feature_vector(_record)
    _candidate = bool(
        np.any(_baseline["crossing"])
        or _event["previous_wall_contact"]
        or _event["previous_period2_exact"]
        or _active
    )
    _todo14_p2_candidate[_transition_row] = _candidate
    if not _candidate:
        _todo14_p2_active[_segment_id] = False
        continue

    _enter, _stay = todo11_threshold_by_segment[_segment_id]
    _threshold = _stay if _active else _enter
    _todo14_p2_threshold[_transition_row] = _threshold
    _todo14_p2_active[_segment_id] = bool(
        _todo11_p2_probability[_transition_row] >= _threshold
    )
    _todo14_p2_trace_prediction[_transition_row] = _todo14_p2_active[
        _segment_id
    ]

assert np.array_equal(_todo11_p2_special, _todo14_p2_trace_prediction)

# Batch-verify that the reconstructed dynamic object-2 vectors are exactly the
# inputs behind the stored probabilities.  This is much faster than traversing
# every forest tree one Python row at a time.
for _segment_id in _todo11_segments:
    _rows = np.flatnonzero(
        (_todo9_transition_segments == _segment_id)
        & _todo14_p2_candidate
        & ~_todo9_loop_prediction[_todo9_p2_event_indices]
    )
    if not len(_rows):
        continue
    _frame = pd.DataFrame(
        _todo14_p2_vectors[_rows], columns=TODO9_FEATURE_COLUMNS
    )
    _reconstructed_probability = _todo11_true_probability(
        todo9_residual_models[int(_segment_id)], _frame
    )
    assert np.allclose(
        _reconstructed_probability,
        _todo11_p2_probability[_rows],
        atol=1e-12,
        rtol=0.0,
    )


def _todo14_phase(event_index, target):
    if bool(_todo12_hidden_initialization[event_index]):
        return "hidden segment start"
    if target:
        return (
            "continuation"
            if _todo12_previous_active_target[event_index]
            else "entry"
        )
    return (
        "exit"
        if _todo12_previous_active_target[event_index]
        else "normal"
    )


def _todo14_physical_values(record):
    _step = max(abs(float(record["step_length"])), 1e-12)
    _x_penetration = max(float(record["x_penetration"]), 0.0)
    _y_penetration = max(float(record["y_penetration"]), 0.0)
    _penetration_norm = float(np.hypot(_x_penetration, _y_penetration))
    _hit = np.asarray(
        [record["x_hit_fraction"], record["y_hit_fraction"]], dtype=float
    )
    _finite_hit = _hit[np.isfinite(_hit)]
    return {
        "axis_angle_distance": float(
            abs(((float(record["free_angle"]) + 45.0) % 90.0) - 45.0)
        ),
        "penetration_norm": _penetration_norm,
        "normalized_penetration": _penetration_norm / _step,
        "minimum_hit_fraction": (
            float(_finite_hit.min()) if len(_finite_hit) else np.nan
        ),
        "hit_fraction_gap": (
            float(abs(_hit[0] - _hit[1]))
            if np.isfinite(_hit).all()
            else np.nan
        ),
        "minimum_previous_wall_gap": float(
            min(
                abs(float(record["previous_x_gap"])),
                abs(float(record["previous_y_gap"])),
            )
        ),
        "distance_step_ratio": float(record["driver_distance"]) / _step,
    }


_todo14_material_cause = np.full(
    _todo14_transition_count, "", dtype=object
)
for _failure_row, _transition_row in zip(
    todo11_material_failures.itertuples(index=False),
    np.flatnonzero(~_todo11_sequential_material),
):
    assert int(_failure_row.source_row) == int(
        _todo9_transition_source_rows[_transition_row]
    )
    _todo14_material_cause[_transition_row] = str(_failure_row.cause)


_todo14_rows = []
_todo14_model_vectors = []
for _transition_row in range(_todo14_transition_count):
    for _object_name, _event_index in (
        ("object1", _todo9_p1_event_indices[_transition_row]),
        ("object2", _todo9_p2_event_indices[_transition_row]),
    ):
        _event = todo9_event_dataset.loc[_event_index]
        _loop = bool(_todo9_loop_prediction[_event_index])
        if _object_name == "object1":
            _record = {
                _name: _event[_name]
                for _name in TODO9_NUMERIC_FEATURES
                + TODO9_CATEGORICAL_FEATURES
            }
            _vector = _todo11_all_X.iloc[_event_index].to_numpy(
                dtype=np.float32
            )
            _active_before = bool(
                _todo14_p1_active_before_all[_event_index]
            )
            _candidate = bool(_todo14_p1_candidate_all[_event_index])
            _threshold = float(_todo14_p1_threshold_all[_event_index])
            _probability = float(_todo11_outer_probability[_event_index])
            _prediction = bool(_todo11_p1_special[_transition_row])
        else:
            _record = _todo14_p2_records[_transition_row]
            _vector = _todo14_p2_vectors[_transition_row]
            _active_before = bool(_todo14_p2_active_before[_transition_row])
            _candidate = bool(_todo14_p2_candidate[_transition_row])
            _threshold = float(_todo14_p2_threshold[_transition_row])
            _probability = float(_todo11_p2_probability[_transition_row])
            _prediction = bool(_todo11_p2_special[_transition_row])

        _target = bool(_todo9_residual_target_all[_event_index])
        _queried = bool(_candidate and not _loop)
        _probability = _probability if _queried else np.nan
        _threshold = _threshold if _queried else np.nan
        _confusion = (
            "PERIOD2"
            if _loop
            else "TP"
            if _target and _prediction
            else "FN"
            if _target
            else "FP"
            if _prediction
            else "TN"
        )
        _error_mechanism = (
            "deterministic period-2"
            if _loop
            else "hidden/reset initialization"
            if bool(_todo12_hidden_initialization[_event_index])
            else "candidate-gate miss"
            if _target and not _prediction and not _queried
            else "queried false negative"
            if _confusion == "FN"
            else "queried false positive"
            if _confusion == "FP"
            else "correct decision"
        )
        _margin = (
            _probability - _threshold
            if np.isfinite(_probability) and np.isfinite(_threshold)
            else np.nan
        )
        _row = {
            "transition_row": int(_transition_row),
            "event_index": int(_event_index),
            "source_row": int(_event["source_row"]),
            "segment_id": int(_event["segment_id"]),
            "object": _object_name,
            "event_mode": str(_event["event_mode"]),
            "actual_branch": str(_event["actual_branch"]),
            "phase": _todo14_phase(_event_index, _target),
            "active_before": _active_before,
            "candidate": _candidate,
            "model_queried": _queried,
            "target_residual_special": _target,
            "predicted_residual_special": _prediction,
            "confusion": _confusion,
            "error_mechanism": _error_mechanism,
            "probability": _probability,
            "threshold": _threshold,
            "probability_margin": _margin,
            "absolute_margin": abs(_margin) if np.isfinite(_margin) else np.nan,
            "material_failure": bool(
                not _todo11_sequential_material[_transition_row]
            ),
            "material_failure_cause": str(
                _todo14_material_cause[_transition_row]
            ),
            "maximum_position_error": float(
                _todo11_maximum_position_error[_transition_row]
            ),
            "maximum_angle_error": float(
                _todo11_maximum_angle_error[_transition_row]
            ),
            "distance_error": float(_todo11_distance_error[_transition_row]),
        }
        if _record is not None:
            _row.update({_name: _record[_name] for _name in TODO9_NUMERIC_FEATURES})
            _row.update(
                {_name: str(_record[_name]) for _name in TODO9_CATEGORICAL_FEATURES}
            )
            _row.update(_todo14_physical_values(_record))
        _todo14_rows.append(_row)
        _todo14_model_vectors.append(_vector)

todo14_event_audit = pd.DataFrame(_todo14_rows)
_todo14_model_matrix = np.asarray(_todo14_model_vectors, dtype=np.float32)
assert len(todo14_event_audit) == 2 * _todo14_transition_count
assert (
    todo14_event_audit["source_row"].to_numpy()
    == np.repeat(_todo9_transition_source_rows, 2)
).all()
assert np.isfinite(
    _todo14_model_matrix[
        todo14_event_audit["model_queried"].to_numpy(dtype=bool)
    ]
).all()


# ---------------------------------------------------------------------------
# 14.A — where the errors occur and whether the model was actually queried
# ---------------------------------------------------------------------------
_todo14_non_loop = todo14_event_audit["confusion"].ne("PERIOD2")
_todo14_selector_error = todo14_event_audit["confusion"].isin(["FP", "FN"])
_todo14_material_selector_error = (
    _todo14_selector_error & todo14_event_audit["material_failure"]
)

todo14_confusion_summary = (
    todo14_event_audit.loc[_todo14_non_loop]
    .groupby("confusion", sort=False)
    .agg(
        events=("event_index", "size"),
        segments=("segment_id", "nunique"),
        material_failure_events=("material_failure", "sum"),
        model_query_share=("model_queried", "mean"),
        median_probability=("probability", "median"),
        median_margin=("probability_margin", "median"),
        median_absolute_margin=("absolute_margin", "median"),
    )
    .reindex(["TP", "FN", "FP", "TN"])
)

_todo14_material_transition_rows = pd.DataFrame(
    {
        "segment_id": _todo9_transition_segments,
        "material_failure": ~_todo11_sequential_material,
    }
)
_todo14_material_error_events = todo14_event_audit.loc[
    _todo14_material_selector_error
]
_todo14_segment_event_counts = (
    _todo14_material_error_events.pivot_table(
        index="segment_id",
        columns="confusion",
        values="event_index",
        aggfunc="size",
        fill_value=0,
    )
    .reindex(columns=["FN", "FP"], fill_value=0)
)
todo14_material_failures_by_segment = (
    _todo14_material_transition_rows.groupby("segment_id")
    .agg(
        transitions=("material_failure", "size"),
        material_failed_transitions=("material_failure", "sum"),
    )
    .join(_todo14_segment_event_counts, how="left")
    .fillna(0)
)
todo14_material_failures_by_segment[["FN", "FP"]] = (
    todo14_material_failures_by_segment[["FN", "FP"]].astype(int)
)

_todo14_detail_columns = [
    "segment_id",
    "source_row",
    "object",
    "confusion",
    "error_mechanism",
    "phase",
    "model_queried",
    "active_before",
    "probability",
    "threshold",
    "probability_margin",
    "previous_wall_mask",
    "free_quadrant",
    "axis_angle_distance",
    "normalized_penetration",
    "minimum_hit_fraction",
    "previous_boundary_run_length",
    "maximum_position_error",
    "maximum_angle_error",
    "material_failure_cause",
]
todo14_material_error_details = (
    todo14_event_audit.loc[
        _todo14_material_selector_error, _todo14_detail_columns
    ]
    .sort_values(["segment_id", "source_row", "object"])
    .reset_index(drop=True)
)

todo14_material_error_context = (
    todo14_material_error_details.groupby(
        [
            "confusion",
            "object",
            "phase",
            "previous_wall_mask",
            "free_quadrant",
        ],
        dropna=False,
    )
    .size()
    .rename("events")
    .sort_values(ascending=False)
    .to_frame()
)
todo14_material_error_mechanisms = (
    todo14_event_audit.loc[_todo14_material_selector_error]
    .groupby(["error_mechanism", "confusion"], dropna=False)
    .agg(
        events=("event_index", "size"),
        segments=("segment_id", "nunique"),
        objects=("object", "nunique"),
    )
    .sort_values("events", ascending=False)
)
todo14_material_error_phase_object = (
    todo14_event_audit.loc[_todo14_material_selector_error]
    .pivot_table(
        index=["phase", "object"],
        columns="confusion",
        values="event_index",
        aggfunc="size",
        fill_value=0,
    )
    .reindex(columns=["FN", "FP"], fill_value=0)
)

_todo14_queried_material_errors = todo14_event_audit.loc[
    _todo14_material_selector_error
    & todo14_event_audit["model_queried"]
].copy()
todo14_material_margin_summary = (
    _todo14_queried_material_errors.groupby("confusion", sort=False)
    .agg(
        events=("event_index", "size"),
        median_probability=("probability", "median"),
        median_signed_margin=("probability_margin", "median"),
        median_absolute_margin=("absolute_margin", "median"),
        share_within_005=(
            "absolute_margin",
            lambda _values: float((_values <= TODO14_MARGIN_BANDS[0]).mean()),
        ),
        share_within_010=(
            "absolute_margin",
            lambda _values: float((_values <= TODO14_MARGIN_BANDS[1]).mean()),
        ),
    )
    .reindex(["FN", "FP"])
)


# ---------------------------------------------------------------------------
# 14.B — robust physical-feature contrasts, descriptive rather than causal
# ---------------------------------------------------------------------------
_todo14_contrast_features = (
    "driver_distance",
    "free_angle",
    "axis_angle_distance",
    "step_length",
    "x_penetration",
    "y_penetration",
    "penetration_norm",
    "normalized_penetration",
    "x_hit_fraction",
    "y_hit_fraction",
    "minimum_hit_fraction",
    "hit_fraction_gap",
    "previous_x_gap",
    "previous_y_gap",
    "minimum_previous_wall_gap",
    "previous_boundary_run_length",
    "previous_period2_position_error",
    "previous_period2_angle_error",
    "distance_step_ratio",
)


def _todo14_robust_contrast(error_label, reference_label, contrast_name):
    _rows = []
    # Compare only rows on which the same RF decision mechanism was actually
    # invoked.  Otherwise FP-versus-TN is dominated by easy interior rows that
    # never passed the causal candidate gate.
    _queried = todo14_event_audit["model_queried"]
    _error = todo14_event_audit["confusion"].eq(error_label) & _queried
    _reference = todo14_event_audit["confusion"].eq(reference_label) & _queried
    for _feature in _todo14_contrast_features:
        _error_values = pd.to_numeric(
            todo14_event_audit.loc[_error, _feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        _reference_values = pd.to_numeric(
            todo14_event_audit.loc[_reference, _feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if not len(_error_values) or not len(_reference_values):
            continue
        _combined = pd.concat([_error_values, _reference_values])
        _scale = float(_combined.quantile(0.75) - _combined.quantile(0.25))
        if _scale <= 1e-12:
            _scale = float(_combined.std(ddof=0))
        if _scale <= 1e-12:
            continue
        _error_median = float(_error_values.median())
        _reference_median = float(_reference_values.median())
        _effect = (_error_median - _reference_median) / _scale

        _segment_signs = []
        for _segment_id in _todo11_segments:
            _segment = todo14_event_audit["segment_id"].eq(_segment_id)
            _a = pd.to_numeric(
                todo14_event_audit.loc[_error & _segment, _feature],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()
            _b = pd.to_numeric(
                todo14_event_audit.loc[_reference & _segment, _feature],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(_a) and len(_b):
                _segment_signs.append(np.sign(float(_a.median() - _b.median())))
        _direction = np.sign(_effect)
        _agreement = (
            float(np.mean(np.asarray(_segment_signs) == _direction))
            if _segment_signs and _direction != 0
            else np.nan
        )
        _rows.append(
            {
                "contrast": contrast_name,
                "feature": _feature,
                "error_n": int(len(_error_values)),
                "reference_n": int(len(_reference_values)),
                "error_median": _error_median,
                "reference_median": _reference_median,
                "robust_shift_IQR": float(_effect),
                "segments_compared": int(len(_segment_signs)),
                "direction_agreement": _agreement,
            }
        )
    return pd.DataFrame(_rows)


todo14_feature_contrasts = pd.concat(
    [
        _todo14_robust_contrast(
            "FN", "TP", "queried FN versus queried TP"
        ),
        _todo14_robust_contrast(
            "FP", "TN", "queried FP versus queried TN"
        ),
    ],
    ignore_index=True,
)
todo14_top_feature_contrasts = (
    todo14_feature_contrasts.assign(
        absolute_robust_shift=lambda _frame: _frame[
            "robust_shift_IQR"
        ].abs()
    )
    .sort_values(
        ["contrast", "absolute_robust_shift"], ascending=[True, False]
    )
    .groupby("contrast", sort=False)
    .head(8)
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------------
# 14.C — cross-segment nearest opposite-label states in the actual RF space
# ---------------------------------------------------------------------------
_todo14_numeric_indices = np.asarray(
    [TODO9_FEATURE_COLUMNS.index(_name) for _name in TODO9_NUMERIC_FEATURES],
    dtype=int,
)
_todo14_query_mask = todo14_event_audit["model_queried"].to_numpy(dtype=bool)
_todo14_numeric_matrix = _todo14_model_matrix[:, _todo14_numeric_indices].astype(
    float
)
_todo14_categories = todo14_event_audit[
    list(TODO9_CATEGORICAL_FEATURES)
].astype(str).to_numpy()
_todo14_objects = todo14_event_audit["object"].astype(str).to_numpy()
_todo14_targets = todo14_event_audit[
    "target_residual_special"
].to_numpy(dtype=bool)
_todo14_segments_all = todo14_event_audit["segment_id"].to_numpy(dtype=int)
_todo14_numeric_scale_by_segment = {}
for _segment_id in _todo11_segments:
    # The focal held-out segment cannot define its own diagnostic geometry.
    # Scaling therefore uses queried rows from the corresponding outer-train
    # segments only; labels are not involved.
    _train = _todo14_query_mask & (_todo14_segments_all != _segment_id)
    _reference = _todo14_numeric_matrix[_train]
    _iqr = np.quantile(_reference, 0.75, axis=0) - np.quantile(
        _reference, 0.25, axis=0
    )
    _std = np.std(_reference, axis=0)
    _todo14_numeric_scale_by_segment[int(_segment_id)] = np.where(
        _iqr > 1e-12, _iqr, np.where(_std > 1e-12, _std, 1.0)
    )


def _todo14_top_feature_differences(row_index, neighbor_index):
    _scale = _todo14_numeric_scale_by_segment[
        int(_todo14_segments_all[row_index])
    ]
    _scores = {
        _name: float(abs(
            _todo14_numeric_matrix[row_index, _feature_index]
            - _todo14_numeric_matrix[neighbor_index, _feature_index]
        ) / _scale[_feature_index]
        )
        for _feature_index, _name in enumerate(TODO9_NUMERIC_FEATURES)
    }
    for _feature_index, _name in enumerate(TODO9_CATEGORICAL_FEATURES):
        if (
            _todo14_categories[row_index, _feature_index]
            != _todo14_categories[neighbor_index, _feature_index]
        ):
            _scores[_name] = 1.0
    return ", ".join(
        _name
        for _name, _score in sorted(
            _scores.items(), key=lambda _item: (-_item[1], _item[0])
        )[:3]
    )


def _todo14_nearest(row_index, target, correct_confusion):
    _candidate_indices = np.flatnonzero(
        _todo14_query_mask
        & (_todo14_targets == target)
        & (_todo14_segments_all != _todo14_segments_all[row_index])
        & (_todo14_objects == _todo14_objects[row_index])
        & todo14_event_audit["confusion"].eq(correct_confusion).to_numpy()
    )
    if not len(_candidate_indices):
        return None
    _scale = _todo14_numeric_scale_by_segment[
        int(_todo14_segments_all[row_index])
    ]
    _numeric_difference = np.clip(
        (
            _todo14_numeric_matrix[_candidate_indices]
            - _todo14_numeric_matrix[row_index]
        )
        / _scale,
        -20.0,
        20.0,
    )
    _categorical_difference = (
        _todo14_categories[_candidate_indices]
        != _todo14_categories[row_index]
    ).astype(float)
    _squared_distance = np.concatenate(
        [_numeric_difference, _categorical_difference], axis=1
    ) ** 2
    _distances = np.sqrt(np.mean(_squared_distance, axis=1))
    _location = int(np.argmin(_distances))
    return int(_candidate_indices[_location]), float(_distances[_location])


def _todo14_nearest_opposite(row_index):
    _opposite_target = not bool(_todo14_targets[row_index])
    _correct_confusion = "TP" if _opposite_target else "TN"
    return _todo14_nearest(
        row_index, _opposite_target, _correct_confusion
    )


def _todo14_nearest_same_label(row_index):
    _target = bool(_todo14_targets[row_index])
    _correct_confusion = "TP" if _target else "TN"
    return _todo14_nearest(row_index, _target, _correct_confusion)


_todo14_neighbor_rows = []
_todo14_error_indices = np.flatnonzero(
    _todo14_material_selector_error.to_numpy()
    & _todo14_query_mask
)
for _row_index in _todo14_error_indices:
    _nearest = _todo14_nearest_opposite(_row_index)
    if _nearest is None:
        continue
    _neighbor_index, _distance = _nearest
    _same = _todo14_nearest_same_label(_row_index)
    _same_distance = _same[1] if _same is not None else np.nan
    _row = todo14_event_audit.iloc[_row_index]
    _neighbor = todo14_event_audit.iloc[_neighbor_index]
    _todo14_neighbor_rows.append(
        {
            "error_confusion": _row["confusion"],
            "error_segment": int(_row["segment_id"]),
            "error_source_row": int(_row["source_row"]),
            "error_object": str(_row["object"]),
            "error_phase": str(_row["phase"]),
            "error_probability": float(_row["probability"]),
            "error_margin": float(_row["probability_margin"]),
            "neighbor_segment": int(_neighbor["segment_id"]),
            "neighbor_source_row": int(_neighbor["source_row"]),
            "neighbor_object": str(_neighbor["object"]),
            "neighbor_target": bool(_neighbor["target_residual_special"]),
            "neighbor_confusion": str(_neighbor["confusion"]),
            "neighbor_distance": _distance,
            "same_label_correct_distance": _same_distance,
            "opposite_to_same_distance_ratio": (
                _distance / _same_distance
                if np.isfinite(_same_distance) and _same_distance > 0
                else np.nan
            ),
            "largest_feature_differences": _todo14_top_feature_differences(
                _row_index, _neighbor_index
            ),
        }
    )

todo14_nearest_opposite_pairs = pd.DataFrame(_todo14_neighbor_rows).sort_values(
    "neighbor_distance"
).reset_index(drop=True)


def _todo14_evenly_spaced(indices, maximum):
    _indices = np.asarray(indices, dtype=int)
    if len(_indices) <= maximum:
        return _indices
    return _indices[
        np.unique(np.linspace(0, len(_indices) - 1, maximum).round().astype(int))
    ]


_todo14_control_indices = []
for _target, _confusion in ((True, "TP"), (False, "TN")):
    _indices = np.flatnonzero(
        _todo14_query_mask
        & (_todo14_targets == _target)
        & todo14_event_audit["confusion"].eq(_confusion).to_numpy()
    )
    _todo14_control_indices.extend(
        _todo14_evenly_spaced(
            _indices, TODO14_NEIGHBOR_CONTROL_PER_CLASS
        ).tolist()
    )

_todo14_control_distances = []
for _row_index in _todo14_control_indices:
    _nearest = _todo14_nearest_opposite(_row_index)
    if _nearest is not None:
        _todo14_control_distances.append(_nearest[1])

_todo14_error_distances = (
    todo14_nearest_opposite_pairs["neighbor_distance"].to_numpy(dtype=float)
    if len(todo14_nearest_opposite_pairs)
    else np.array([], dtype=float)
)
_todo14_control_distances = np.asarray(_todo14_control_distances, dtype=float)
todo14_neighbor_summary = pd.DataFrame(
    [
        {
            "population": "material FP/FN queried",
            "events": len(_todo14_error_distances),
            "q10_distance": (
                float(np.quantile(_todo14_error_distances, 0.10))
                if len(_todo14_error_distances)
                else np.nan
            ),
            "median_distance": (
                float(np.median(_todo14_error_distances))
                if len(_todo14_error_distances)
                else np.nan
            ),
            "q90_distance": (
                float(np.quantile(_todo14_error_distances, 0.90))
                if len(_todo14_error_distances)
                else np.nan
            ),
        },
        {
            "population": "correct queried control",
            "events": len(_todo14_control_distances),
            "q10_distance": (
                float(np.quantile(_todo14_control_distances, 0.10))
                if len(_todo14_control_distances)
                else np.nan
            ),
            "median_distance": (
                float(np.median(_todo14_control_distances))
                if len(_todo14_control_distances)
                else np.nan
            ),
            "q90_distance": (
                float(np.quantile(_todo14_control_distances, 0.90))
                if len(_todo14_control_distances)
                else np.nan
            ),
        },
    ]
).set_index("population")
todo14_neighbor_breakdown = (
    todo14_nearest_opposite_pairs.groupby("error_confusion", sort=False)
    .agg(
        events=("error_source_row", "size"),
        median_opposite_distance=("neighbor_distance", "median"),
        median_same_label_distance=(
            "same_label_correct_distance", "median"
        ),
        median_opposite_to_same_ratio=(
            "opposite_to_same_distance_ratio", "median"
        ),
        share_opposite_no_farther=(
            "opposite_to_same_distance_ratio",
            lambda _values: float((_values <= 1.0).mean()),
        ),
    )
)


# ---------------------------------------------------------------------------
# 14.D — frozen diagnostic verdict and compact figures
# ---------------------------------------------------------------------------
_todo14_material_error_query_mask = (
    _todo14_material_selector_error
    & todo14_event_audit["model_queried"]
)
_todo14_material_error_margins = todo14_event_audit.loc[
    _todo14_material_error_query_mask, "absolute_margin"
].dropna()
_todo14_failed_transition_set = set(
    np.flatnonzero(~_todo11_sequential_material).tolist()
)
_todo14_selector_failed_transition_set = set(
    todo14_event_audit.loc[
        _todo14_material_selector_error, "transition_row"
    ].astype(int).tolist()
)
_todo14_nearest_ratio = (
    float(np.median(_todo14_error_distances))
    / float(np.median(_todo14_control_distances))
    if len(_todo14_error_distances)
    and len(_todo14_control_distances)
    and float(np.median(_todo14_control_distances)) > 0
    else np.nan
)
_todo14_pair_ratios = todo14_nearest_opposite_pairs[
    "opposite_to_same_distance_ratio"
].dropna()
_todo14_top_three_segment_share = float(
    todo14_material_failures_by_segment[
        "material_failed_transitions"
    ].nlargest(3).sum()
    / max(1, (~_todo11_sequential_material).sum())
)
todo14_manifest = pd.Series(
    {
        "diagnostic only; no refit or threshold tuning": True,
        "outer-LOSO predictions inspected after scoring": True,
        "sequential object2 inputs reconstructed": True,
        "TODO11 material failed transitions": int(
            (~_todo11_sequential_material).sum()
        ),
        "material selector-error events": int(
            _todo14_material_selector_error.sum()
        ),
        "material failures without selector FP/FN": int(
            len(
                _todo14_failed_transition_set
                - _todo14_selector_failed_transition_set
            )
        ),
        "material selector errors not queried": int(
            (
                _todo14_material_selector_error
                & ~todo14_event_audit["model_queried"]
            ).sum()
        ),
        "material errors within 0.05 probability margin": int(
            (_todo14_material_error_margins <= TODO14_MARGIN_BANDS[0]).sum()
        ),
        "material errors within 0.10 probability margin": int(
            (_todo14_material_error_margins <= TODO14_MARGIN_BANDS[1]).sum()
        ),
        "material errors above 0.10 probability margin": int(
            (_todo14_material_error_margins > TODO14_MARGIN_BANDS[1]).sum()
        ),
        "top-3 segment share of material failures": (
            _todo14_top_three_segment_share
        ),
        "median nearest-opposite distance / correct control": (
            _todo14_nearest_ratio
        ),
        "share nearest opposite no farther than same-label": (
            float((_todo14_pair_ratios <= 1.0).mean())
            if len(_todo14_pair_ratios)
            else np.nan
        ),
        "fully blind test preserved": False,
    },
    name="value",
)

_todo14_figure, _todo14_axes = plt.subplots(1, 2, figsize=(15, 5))
_todo14_plot = todo14_material_failures_by_segment.reindex(
    _todo11_segments, fill_value=0
)
_todo14_axes[0].bar(
    _todo14_plot.index - 0.18,
    _todo14_plot["FN"],
    width=0.36,
    label="material FN events",
    color="#d95f02",
)
_todo14_axes[0].bar(
    _todo14_plot.index + 0.18,
    _todo14_plot["FP"],
    width=0.36,
    label="material FP events",
    color="#1b9e77",
)
_todo14_axes[0].plot(
    _todo14_plot.index,
    _todo14_plot["material_failed_transitions"],
    color="#404040",
    marker="o",
    label="all material failed transitions",
)
_todo14_axes[0].set(
    xlabel="held-out segment",
    ylabel="count",
    title="Where TODO 11 still fails",
)
_todo14_axes[0].legend(fontsize=8)

for _label, _color in (("FN", "#d95f02"), ("FP", "#1b9e77")):
    _mask = _todo14_material_error_query_mask & todo14_event_audit[
        "confusion"
    ].eq(_label)
    _todo14_axes[1].scatter(
        todo14_event_audit.loc[_mask, "axis_angle_distance"],
        todo14_event_audit.loc[_mask, "probability_margin"],
        label=_label,
        color=_color,
        alpha=0.8,
    )
_todo14_axes[1].axhline(0.0, color="#404040", linewidth=1)
_todo14_axes[1].axhspan(-0.05, 0.05, color="#bdbdbd", alpha=0.2)
_todo14_axes[1].set(
    xlabel="distance of free angle to nearest axis, degrees",
    ylabel="P(special) - active threshold",
    title="Threshold uncertainty versus collision geometry",
)
_todo14_axes[1].legend()
_todo14_figure.tight_layout()

display(
    Markdown("#### TODO 14 — frozen error-analysis manifest"),
    todo14_manifest.to_frame(),
    Markdown("#### Sequential selector confusion and confidence"),
    todo14_confusion_summary.style.format(precision=6),
    Markdown("#### Material failures by held-out segment"),
    todo14_material_failures_by_segment.style.format(precision=6),
    Markdown("#### Material-error mechanism, phase and confidence"),
    todo14_material_error_mechanisms,
    todo14_material_error_phase_object,
    todo14_material_margin_summary.style.format(precision=6),
    _todo14_figure,
    Markdown("#### Largest robust physical-feature shifts"),
    todo14_top_feature_contrasts.style.format(precision=6),
    Markdown("#### Material-error contexts"),
    todo14_material_error_context.head(20),
    Markdown("#### Nearest opposite-label states from other segments"),
    todo14_neighbor_summary.style.format(precision=6),
    todo14_neighbor_breakdown.style.format(precision=6),
    todo14_nearest_opposite_pairs.head(15).style.format(precision=6),
    Markdown("#### Every material FP/FN event"),
    todo14_material_error_details.style.format(precision=6),
)
