"""TODO 16 companion: compact causal dynamics for SPECIAL entry.

This experiment isolates representation from classifier specialization.  A
current-only entry head and an otherwise identical compact-dynamics entry head
use the same training population, RF architecture, nested LOSO selection and
sequential simulator.  Geometry, PERIOD2 and TODO 11 continuation stay frozen.
"""

import gc
from itertools import combinations


TODO16_REPRESENTATIONS = ("current_only", "compact_dynamics")
TODO16_ENTRY_THRESHOLDS = TODO11_ENTER_THRESHOLDS
TODO16_MAXIMUM_MATERIAL_FAILURES = 40
TODO16_MINIMUM_IMPROVED_SEGMENTS = 8
TODO16_RANDOM_STATE = TODO9_RANDOM_STATE

assert all(
    0.0 <= _stay <= min(TODO16_ENTRY_THRESHOLDS) <= 1.0
    for _, _stay in todo11_threshold_by_segment.values()
)


# ---------------------------------------------------------------------------
# 16.A — causal lag indices and compact motion/phase summaries
# ---------------------------------------------------------------------------
_todo16_event_count = len(todo9_event_dataset)
_todo16_lag1 = np.full(_todo16_event_count, -1, dtype=np.int32)
_todo16_lag2 = np.full(_todo16_event_count, -1, dtype=np.int32)
for _indices in _todo11_sequence_indices:
    _indices = np.asarray(_indices, dtype=np.int32)
    if len(_indices) > 1:
        _todo16_lag1[_indices[1:]] = _indices[:-1]
    if len(_indices) > 2:
        _todo16_lag2[_indices[2:]] = _indices[:-2]

_todo16_valid1 = _todo16_lag1 >= 0
_todo16_valid2 = _todo16_lag2 >= 0
for _indices in _todo11_sequence_indices:
    _sequence = todo9_event_dataset.loc[
        _indices, ["object", "segment_id", "segment_step"]
    ]
    _steps = _sequence["segment_step"].to_numpy(dtype=int)
    assert _sequence["object"].nunique() == 1
    assert _sequence["segment_id"].nunique() == 1
    assert np.all(np.diff(_steps) == 1)
    assert _todo16_lag1[int(_indices[0])] == -1
    assert _todo16_lag2[int(_indices[0])] == -1
    if len(_indices) > 1:
        assert _todo16_lag2[int(_indices[1])] == -1
        assert np.array_equal(
            _todo16_lag1[_indices[1:]], np.asarray(_indices[:-1])
        )
    if len(_indices) > 2:
        assert np.array_equal(
            _todo16_lag2[_indices[2:]], np.asarray(_indices[:-2])
        )

_todo16_base_matrix = _todo15_current_matrix
_todo16_previous_special_column = TODO9_FEATURE_COLUMNS.index(
    "previous_special"
)
_todo16_column = {
    _name: TODO9_FEATURE_COLUMNS.index(_name)
    for _name in (
        "previous_x",
        "previous_y",
        "previous_angle",
        "driver_distance",
        "free_angle",
        "step_length",
        "free_x",
        "free_y",
        "x_penetration",
        "y_penetration",
        "x_hit_fraction",
        "y_hit_fraction",
        "previous_x_gap",
        "previous_y_gap",
        "predicted_x_crossing",
        "predicted_y_crossing",
        "both_axis_proposal",
        "previous_wall_contact",
        "previous_boundary_run_length",
    )
}

_todo16_delta_names = (
    "previous_x",
    "previous_y",
    "driver_distance",
    "step_length",
    "free_x",
    "free_y",
    "x_penetration",
    "y_penetration",
    "previous_x_gap",
    "previous_y_gap",
)
_todo16_dynamic_feature_names = (
    tuple(f"delta1__{_name}" for _name in _todo16_delta_names)
    + (
        "delta2__driver_distance",
        "delta1_previous_angle_sin",
        "delta1_previous_angle_one_minus_cos",
        "delta2_previous_angle_sin",
        "delta2_previous_angle_one_minus_cos",
        "delta1_free_angle_sin",
        "delta1_free_angle_one_minus_cos",
        "delta2_free_angle_sin",
        "delta2_free_angle_one_minus_cos",
        "delta1_crossing_any",
        "delta1_previous_wall_contact",
        "approach_wall_max",
        "delta_penetration_norm",
        "age_since_crossing_capped8",
        "age_since_crossing_valid",
        "age_since_contact_capped8",
        "age_since_contact_valid",
        "boundary_run_parity",
        "boundary_run_capped8",
        "lag1_valid",
        "lag2_valid",
    )
)

_todo16_age_cap = 8
_todo16_age_before_crossing = np.full(
    _todo16_event_count, _todo16_age_cap, dtype=np.float32
)
_todo16_crossing_seen_before = np.zeros(
    _todo16_event_count, dtype=bool
)
_todo16_age_before_contact = np.full(
    _todo16_event_count, _todo16_age_cap, dtype=np.float32
)
_todo16_contact_seen_before = np.zeros(_todo16_event_count, dtype=bool)
for _indices in _todo11_sequence_indices:
    _crossing_age = _todo16_age_cap
    _contact_age = _todo16_age_cap
    _crossing_seen = False
    _contact_seen = False
    for _row in _indices:
        _row = int(_row)
        _todo16_age_before_crossing[_row] = _crossing_age
        _todo16_crossing_seen_before[_row] = _crossing_seen
        _todo16_age_before_contact[_row] = _contact_age
        _todo16_contact_seen_before[_row] = _contact_seen
        _crossing_now = bool(
            _todo16_base_matrix[
                _row, _todo16_column["predicted_x_crossing"]
            ]
            or _todo16_base_matrix[
                _row, _todo16_column["predicted_y_crossing"]
            ]
        )
        _contact_now = bool(
            _todo16_base_matrix[
                _row, _todo16_column["previous_wall_contact"]
            ]
        )
        if _crossing_now:
            _crossing_age = 1
            _crossing_seen = True
        elif _crossing_seen:
            _crossing_age = min(
                _todo16_age_cap, _crossing_age + 1
            )
        if _contact_now:
            _contact_age = 1
            _contact_seen = True
        elif _contact_seen:
            _contact_age = min(_todo16_age_cap, _contact_age + 1)


def _todo16_take_lag(base_matrix, event_indices, lag_indices):
    """Return lag values and zero-fill only unavailable segment starts."""
    _event_indices = np.asarray(event_indices, dtype=int)
    _source = lag_indices[_event_indices]
    _valid = _source >= 0
    _values = np.zeros(
        (len(_event_indices), base_matrix.shape[1]), dtype=np.float32
    )
    if _valid.any():
        _values[_valid] = base_matrix[_source[_valid]]
    return _values, _valid


def _todo16_compact_features(current_vectors, event_indices):
    """Compress two observed past rows into causal differences and phase."""
    _current = np.asarray(current_vectors, dtype=np.float32)
    if _current.ndim == 1:
        _current = _current.reshape(1, -1)
    _event_indices = np.asarray(event_indices, dtype=int)
    _lag1, _valid1 = _todo16_take_lag(
        _todo16_base_matrix, _event_indices, _todo16_lag1
    )
    _lag2, _valid2 = _todo16_take_lag(
        _todo16_base_matrix, _event_indices, _todo16_lag2
    )
    _features = []

    for _name in _todo16_delta_names:
        _column = _todo16_column[_name]
        _delta1 = _current[:, _column] - _lag1[:, _column]
        _delta1[~_valid1] = 0.0
        _features.append(_delta1)

    _distance_column = _todo16_column["driver_distance"]
    _distance_delta2 = (
        _lag1[:, _distance_column] - _lag2[:, _distance_column]
    )
    _distance_delta2[~_valid2] = 0.0
    _features.append(_distance_delta2)

    for _name in ("previous_angle", "free_angle"):
        _column = _todo16_column[_name]
        _difference1 = np.deg2rad(
            _todo7_wrap_degrees(
                _current[:, _column] - _lag1[:, _column]
            )
        )
        _difference2 = np.deg2rad(
            _todo7_wrap_degrees(
                _current[:, _column] - _lag2[:, _column]
            )
        )
        _sine1 = np.sin(_difference1)
        _cosine1 = 1.0 - np.cos(_difference1)
        _sine2 = np.sin(_difference2)
        _cosine2 = 1.0 - np.cos(_difference2)
        _sine1[~_valid1] = 0.0
        _cosine1[~_valid1] = 0.0
        _sine2[~_valid2] = 0.0
        _cosine2[~_valid2] = 0.0
        _features.extend((_sine1, _cosine1, _sine2, _cosine2))

    _run = np.maximum(
        _current[:, _todo16_column["previous_boundary_run_length"]], 0.0
    )
    _x_gap = _current[:, _todo16_column["previous_x_gap"]]
    _y_gap = _current[:, _todo16_column["previous_y_gap"]]
    _lag1_x_gap = _lag1[:, _todo16_column["previous_x_gap"]]
    _lag1_y_gap = _lag1[:, _todo16_column["previous_y_gap"]]
    _x_penetration = _current[:, _todo16_column["x_penetration"]]
    _y_penetration = _current[:, _todo16_column["y_penetration"]]
    _lag1_x_penetration = _lag1[:, _todo16_column["x_penetration"]]
    _lag1_y_penetration = _lag1[:, _todo16_column["y_penetration"]]
    _x_crossing = _current[:, _todo16_column["predicted_x_crossing"]]
    _y_crossing = _current[:, _todo16_column["predicted_y_crossing"]]
    _crossing_any = np.maximum(_x_crossing, _y_crossing)
    _lag1_crossing_any = np.maximum(
        _lag1[:, _todo16_column["predicted_x_crossing"]],
        _lag1[:, _todo16_column["predicted_y_crossing"]],
    )
    _crossing_change = _crossing_any - _lag1_crossing_any
    _contact = _current[:, _todo16_column["previous_wall_contact"]]
    _contact_change = (
        _contact - _lag1[:, _todo16_column["previous_wall_contact"]]
    )
    _approach_wall = np.maximum(
        _lag1_x_gap - _x_gap, _lag1_y_gap - _y_gap
    )
    _penetration_delta = np.hypot(
        _x_penetration, _y_penetration
    ) - np.hypot(_lag1_x_penetration, _lag1_y_penetration)
    for _array in (
        _crossing_change,
        _contact_change,
        _approach_wall,
        _penetration_delta,
    ):
        _array[~_valid1] = 0.0

    _past_crossing_age = _todo16_age_before_crossing[_event_indices]
    _past_contact_age = _todo16_age_before_contact[_event_indices]
    _crossing_age = np.where(
        _crossing_any > 0.5, 0.0, _past_crossing_age
    )
    _contact_age = np.where(_contact > 0.5, 0.0, _past_contact_age)
    _crossing_age_valid = (
        (_crossing_any > 0.5)
        | _todo16_crossing_seen_before[_event_indices]
    )
    _contact_age_valid = (
        (_contact > 0.5) | _todo16_contact_seen_before[_event_indices]
    )
    _features.extend(
        (
            _crossing_change,
            _contact_change,
            _approach_wall,
            _penetration_delta,
            _crossing_age,
            _crossing_age_valid.astype(np.float32),
            _contact_age,
            _contact_age_valid.astype(np.float32),
            np.mod(_run, 2.0),
            np.minimum(_run, 8.0),
            _valid1.astype(np.float32),
            _valid2.astype(np.float32),
        )
    )
    _result = np.column_stack(_features).astype(np.float32)
    assert _result.shape[1] == len(_todo16_dynamic_feature_names)
    return np.nan_to_num(_result, nan=0.0, posinf=0.0, neginf=0.0)


def _todo16_design(representation, current_vectors, event_indices):
    """Build the current-only control or compact-dynamics representation."""
    _current = np.asarray(current_vectors, dtype=np.float32).copy()
    if _current.ndim == 1:
        _current = _current.reshape(1, -1)
    # Entry is queried only while the predicted automaton is NORMAL.
    _current[:, _todo16_previous_special_column] = 0.0
    if representation == "current_only":
        return _current
    if representation != "compact_dynamics":
        raise ValueError(f"Unknown TODO16 representation: {representation}")
    return np.concatenate(
        (
            _current,
            _todo16_compact_features(current_vectors, event_indices),
        ),
        axis=1,
    )


# The original object-2 training record uses the observed current object-1
# position through todo7_distance[t].  Replace that current-step quantity with
# a fixed causal teacher: the standard object-1 proposal, except where the
# already-detected deterministic PERIOD2 rule supplies the lag-2 position.
# This teacher is label-free and independent of the entry representation.
_todo16_training_current_matrix = _todo16_base_matrix.copy()
_todo16_p1_teacher_position = _todo9_baseline_position[
    _todo9_p1_event_indices
].copy()
_todo16_p1_loop = np.asarray(_todo9_loop_prediction, dtype=bool)[
    _todo9_p1_event_indices
]
_todo16_p1_teacher_position[_todo16_p1_loop] = _todo9_position_lag2[
    _todo9_p1_event_indices[_todo16_p1_loop]
]
_todo16_training_p2_distance = np.linalg.norm(
    _todo16_p1_teacher_position - _todo9_previous_p2, axis=1
)
for _transition_row, (_data_index, _event_index) in enumerate(
    zip(_todo9_transition_indices, _todo9_p2_event_indices)
):
    _driver_distance = float(
        _todo16_training_p2_distance[_transition_row]
    )
    _baseline = _todo8_update_object(
        todo7_p2[_data_index - 1],
        todo7_angle2[_data_index - 1],
        _driver_distance,
        -1.0,
    )
    _record = _todo9_feature_record(
        _event_index, _baseline, _driver_distance
    )
    _todo16_training_current_matrix[_event_index] = (
        _todo10_feature_vector(_record)
    )

assert np.array_equal(
    _todo16_training_current_matrix[
        _todo9_p2_event_indices, _todo16_column["driver_distance"]
    ],
    _todo16_training_p2_distance.astype(np.float32),
)
assert np.isfinite(_todo16_training_current_matrix).all()

_todo16_design_by_representation = {
    _representation: _todo16_design(
        _representation,
        _todo16_base_matrix,
        np.arange(_todo16_event_count, dtype=int),
    )
    for _representation in TODO16_REPRESENTATIONS
}
_todo16_training_design_by_representation = {
    _representation: _todo16_design(
        _representation,
        _todo16_training_current_matrix,
        np.arange(_todo16_event_count, dtype=int),
    )
    for _representation in TODO16_REPRESENTATIONS
}
_todo16_feature_names = {
    "current_only": tuple(TODO9_FEATURE_COLUMNS),
    "compact_dynamics": (
        tuple(TODO9_FEATURE_COLUMNS) + _todo16_dynamic_feature_names
    ),
}
for _representation in TODO16_REPRESENTATIONS:
    assert _todo16_design_by_representation[_representation].shape[1] == len(
        _todo16_feature_names[_representation]
    )
    assert np.isfinite(
        _todo16_design_by_representation[_representation]
    ).all()


def _todo16_fit_entry_model(excluded_segments, representation):
    _excluded = np.asarray(tuple(excluded_segments), dtype=int)
    _train = _todo15_entry_population & ~np.isin(
        _todo15_event_segments, _excluded
    )
    _target = _todo15_entry_target[_train]
    assert set(np.unique(_target)) == {False, True}
    _model = RandomForestClassifier(
        n_estimators=TODO9_RF_ESTIMATORS,
        max_depth=TODO9_RF_MAX_DEPTH,
        min_samples_leaf=TODO9_RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=TODO16_RANDOM_STATE,
        n_jobs=-1,
    )
    _model.fit(
        _todo16_training_design_by_representation[representation][_train],
        _target,
    )
    return _model


# ---------------------------------------------------------------------------
# 16.B — frozen sequential geometry with dynamic object-2 current features
# ---------------------------------------------------------------------------
def _todo16_add_entry_probabilities(cache, model, representation):
    _result = dict(cache)
    _p1_events = np.asarray(cache["p1_events"], dtype=int)
    _p1_design = _todo16_design_by_representation[representation][
        _p1_events
    ]
    _result["p1_entry_probability"] = _todo11_true_probability(
        model, _p1_design
    )

    _p2_events = np.repeat(np.asarray(cache["p2_events"], dtype=int), 2)
    _p2_vectors = cache["todo15_p2_vectors"].reshape(
        -1, len(TODO9_FEATURE_COLUMNS)
    )
    _p2_design = _todo16_design(
        representation, _p2_vectors, _p2_events
    )
    _result["p2_entry_probability"] = _todo11_true_probability(
        model, _p2_design
    ).reshape(-1, 2)
    return _result


# ---------------------------------------------------------------------------
# 16.C — pairwise nested LOSO threshold selection for each representation
# ---------------------------------------------------------------------------
_todo16_accumulators = {
    (int(_outer), _representation, float(_threshold)): {
        "material_errors": [],
        "exact_errors": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "run_tp": 0,
        "run_fp": 0,
        "run_fn": 0,
    }
    for _outer in _todo11_segments
    for _representation in TODO16_REPRESENTATIONS
    for _threshold in TODO16_ENTRY_THRESHOLDS
}
_todo16_general_pair_models_fitted = 0
_todo16_entry_pair_models_fitted = 0

for _first, _second in combinations(_todo11_segments, 2):
    _general_model = _todo13_fit_general_model((_first, _second))
    _todo16_general_pair_models_fitted += 1
    _entry_models = {
        _representation: _todo16_fit_entry_model(
            (_first, _second), _representation
        )
        for _representation in TODO16_REPRESENTATIONS
    }
    _todo16_entry_pair_models_fitted += len(TODO16_REPRESENTATIONS)

    for _outer, _inner in ((_first, _second), (_second, _first)):
        _common_cache = _todo15_add_continuation_probabilities(
            todo15_geometry_caches[int(_inner)], _general_model
        )
        _stay_threshold = todo11_threshold_by_segment[int(_outer)][1]
        for _representation, _entry_model in _entry_models.items():
            _cache = _todo16_add_entry_probabilities(
                _common_cache, _entry_model, _representation
            )
            for _entry_threshold in TODO16_ENTRY_THRESHOLDS:
                _result = _todo12_simulate_fold(
                    _cache, _entry_threshold, _stay_threshold
                )
                _accumulator = _todo16_accumulators[
                    (
                        int(_outer),
                        _representation,
                        float(_entry_threshold),
                    )
                ]
                _accumulator["material_errors"].append(
                    1.0 - float(_result["material"].mean())
                )
                _accumulator["exact_errors"].append(
                    1.0 - float(_result["exact"].mean())
                )
                _tp, _fp, _fn = _todo12_binary_counts(
                    _result["target"], _result["prediction"]
                )
                _accumulator["tp"] += _tp
                _accumulator["fp"] += _fp
                _accumulator["fn"] += _fn
                _accumulator["run_tp"] += _result["run_tp"]
                _accumulator["run_fp"] += _result["run_fp"]
                _accumulator["run_fn"] += _result["run_fn"]
    del _entry_models, _general_model
    gc.collect()

assert _todo16_general_pair_models_fitted == 55
assert _todo16_entry_pair_models_fitted == 110

_todo16_selection_rows = []
todo16_entry_threshold_by_representation = {
    _representation: {} for _representation in TODO16_REPRESENTATIONS
}
for _outer in _todo11_segments:
    for _representation in TODO16_REPRESENTATIONS:
        _best = None
        for _entry_threshold in TODO16_ENTRY_THRESHOLDS:
            _accumulator = _todo16_accumulators[
                (int(_outer), _representation, float(_entry_threshold))
            ]
            _, _, _f1 = _todo12_f1_from_counts(
                _accumulator["tp"],
                _accumulator["fp"],
                _accumulator["fn"],
            )
            _, _, _run_f1 = _todo12_f1_from_counts(
                _accumulator["run_tp"],
                _accumulator["run_fp"],
                _accumulator["run_fn"],
            )
            _candidate = {
                "segment_id": int(_outer),
                "representation": _representation,
                "entry": float(_entry_threshold),
                "stay": float(
                    todo11_threshold_by_segment[int(_outer)][1]
                ),
                "macro_material_error": float(
                    np.mean(_accumulator["material_errors"])
                ),
                "residual_f1": float(_f1),
                "run_f1": float(_run_f1),
                "macro_exact_error": float(
                    np.mean(_accumulator["exact_errors"])
                ),
            }
            _key = (
                _candidate["macro_material_error"],
                -_candidate["residual_f1"],
                _candidate["macro_exact_error"],
                -_candidate["run_f1"],
                -_candidate["entry"],
            )
            if _best is None or _key < _best[0]:
                _best = (_key, _candidate)
        _chosen = _best[1]
        todo16_entry_threshold_by_representation[_representation][
            int(_outer)
        ] = float(_chosen["entry"])
        _todo16_selection_rows.append(_chosen)

todo16_nested_selection = pd.DataFrame(_todo16_selection_rows).set_index(
    ["segment_id", "representation"]
)
del _todo16_accumulators
gc.collect()


# ---------------------------------------------------------------------------
# 16.D — outer-LOSO sequential evaluation
# ---------------------------------------------------------------------------
todo16_results = {}
_todo16_outer_models_fitted = 0
_todo16_importance_rows = []
for _representation in TODO16_REPRESENTATIONS:
    _arrays = _todo15_empty_result_arrays()
    for _segment_id in _todo11_segments:
        _segment_id = int(_segment_id)
        _entry_model = _todo16_fit_entry_model(
            (_segment_id,), _representation
        )
        _todo16_outer_models_fitted += 1
        _cache = _todo15_add_continuation_probabilities(
            todo15_geometry_caches[_segment_id],
            todo9_residual_models[_segment_id],
        )
        _cache = _todo16_add_entry_probabilities(
            _cache, _entry_model, _representation
        )
        _result = _todo12_simulate_fold(
            _cache,
            todo16_entry_threshold_by_representation[_representation][
                _segment_id
            ],
            todo11_threshold_by_segment[_segment_id][1],
        )
        _rows = _cache["transition_rows"]
        # PERIOD2 geometry must not depend on either learned entry head.
        for _object in ("p1", "p2"):
            _loop = _cache[f"{_object}_loop"]
            _reference = (
                _todo11_p1_prediction if _object == "p1"
                else _todo11_p2_prediction
            )
            np.testing.assert_allclose(
                _result[f"{_object}_prediction"][_loop],
                _reference[_rows[_loop]],
                atol=1e-12, rtol=0.0,
            )
        for _name in (
            "exact",
            "material",
            "position_error",
            "angle_error",
            "distance_error",
            "p1_special",
            "p2_special",
            "p1_prediction",
            "p2_prediction",
            "a1_prediction",
            "a2_prediction",
            "distance_prediction",
        ):
            _arrays[_name][_rows] = _result[_name]
        for _name in ("run_tp", "run_fp", "run_fn"):
            _arrays[_name] += _result[_name]
        _arrays["onset_delays"].extend(_result["onset_delays"])
        _arrays["exit_delays"].extend(_result["exit_delays"])

        if _representation == "compact_dynamics":
            for _feature, _importance in zip(
                _todo16_feature_names[_representation],
                _entry_model.feature_importances_,
            ):
                _todo16_importance_rows.append(
                    {
                        "segment_id": _segment_id,
                        "feature": _feature,
                        "importance": float(_importance),
                    }
                )
        del _entry_model, _cache
        gc.collect()
    todo16_results[_representation] = _arrays

assert _todo16_outer_models_fitted == 22

_todo16_summary_rows = [todo15_model_summary.loc["TODO11"].to_dict()]
_todo16_summary_rows[0]["model"] = "TODO11"
for _representation in TODO16_REPRESENTATIONS:
    _todo16_summary_rows.append(
        _todo15_summarize_result(
            _representation, todo16_results[_representation]
        )
    )
todo16_model_summary = pd.DataFrame(_todo16_summary_rows).set_index("model")

_todo16_segment_rows = []
for _segment_id in _todo11_segments:
    _segment_id = int(_segment_id)
    _mask = _todo9_transition_segments == _segment_id
    _todo16_segment_rows.append(
        {
            "segment_id": _segment_id,
            "current_entry": todo16_entry_threshold_by_representation[
                "current_only"
            ][_segment_id],
            "dynamics_entry": todo16_entry_threshold_by_representation[
                "compact_dynamics"
            ][_segment_id],
            "TODO11_failures": int(
                (~_todo11_sequential_material[_mask]).sum()
            ),
            "current_failures": int(
                (~todo16_results["current_only"]["material"][_mask]).sum()
            ),
            "dynamics_failures": int(
                (
                    ~todo16_results["compact_dynamics"]["material"][_mask]
                ).sum()
            ),
            "TODO11_exact": float(
                _todo11_sequential_exact[_mask].mean()
            ),
            "current_exact": float(
                todo16_results["current_only"]["exact"][_mask].mean()
            ),
            "dynamics_exact": float(
                todo16_results["compact_dynamics"]["exact"][_mask].mean()
            ),
        }
    )
todo16_per_segment = pd.DataFrame(_todo16_segment_rows).set_index(
    "segment_id"
)
todo16_per_segment["dynamics_minus_current_failures"] = (
    todo16_per_segment["dynamics_failures"]
    - todo16_per_segment["current_failures"]
)
todo16_per_segment["dynamics_minus_current_exact"] = (
    todo16_per_segment["dynamics_exact"]
    - todo16_per_segment["current_exact"]
)

_todo16_dynamics_events = np.zeros(_todo16_event_count, dtype=bool)
_todo16_dynamics_events[_todo9_p1_event_indices] = todo16_results[
    "compact_dynamics"
]["p1_special"]
_todo16_dynamics_events[_todo9_p2_event_indices] = todo16_results[
    "compact_dynamics"
]["p2_special"]
_todo16_phase_rows = []
for _phase, _mask in (
    ("entry recall", _todo12_entry_target),
    ("continuation recall", _todo12_continuation_target),
    ("correct exit share", _todo12_exit_target),
):
    _eligible = _mask & ~_todo12_hidden_initialization
    _score = (
        float((~_todo16_dynamics_events[_eligible]).mean())
        if _phase == "correct exit share"
        else float(_todo16_dynamics_events[_eligible].mean())
    )
    _todo16_phase_rows.append(
        {"phase": _phase, "rows": int(_eligible.sum()), "score": _score}
    )
todo16_phase_summary = pd.DataFrame(_todo16_phase_rows).set_index("phase")

_todo16_failure_rows = []
_todo16_dynamics = todo16_results["compact_dynamics"]
for _transition_row in np.flatnonzero(~_todo16_dynamics["material"]):
    _causes = []
    for _event_index, _prediction in (
        (
            _todo9_p1_event_indices[_transition_row],
            _todo16_dynamics["p1_special"][_transition_row],
        ),
        (
            _todo9_p2_event_indices[_transition_row],
            _todo16_dynamics["p2_special"][_transition_row],
        ),
    ):
        if _todo12_hidden_initialization[_event_index]:
            _causes.append("hidden/reset initialization")
        elif _todo12_residual_target[_event_index] and not _prediction:
            _causes.append("missed entry/continuation")
        elif not _todo12_residual_target[_event_index] and _prediction:
            _causes.append("false/late special")
    if not _causes:
        _causes.append("upstream or mixed sequential error")
    _todo16_failure_rows.append(
        {
            "segment_id": int(
                _todo9_transition_segments[_transition_row]
            ),
            "source_row": int(_todo9_transition_source_rows[_transition_row]),
            "cause": " + ".join(sorted(set(_causes))),
        }
    )
todo16_material_failures = pd.DataFrame(
    _todo16_failure_rows, columns=["segment_id", "source_row", "cause"]
)
todo16_material_failure_summary = (
    todo16_material_failures.groupby("cause")
    .agg(
        transitions=("source_row", "size"),
        segments=("segment_id", "nunique"),
    )
    .sort_values("transitions", ascending=False)
)

todo16_feature_importance = (
    pd.DataFrame(_todo16_importance_rows)
    .groupby("feature", as_index=False)
    .agg(
        mean_importance=("importance", "mean"),
        folds_present=("segment_id", "nunique"),
    )
    .sort_values("mean_importance", ascending=False)
    .reset_index(drop=True)
)

_todo16_control_summary = todo16_model_summary.loc["current_only"]
_todo16_dynamics_summary = todo16_model_summary.loc["compact_dynamics"]
TODO16_DYNAMICS_BEATS_CONTROL = bool(
    _todo16_dynamics_summary["material_failures"]
    < _todo16_control_summary["material_failures"]
    and _todo16_dynamics_summary["exact_share"]
    > _todo16_control_summary["exact_share"]
    and _todo16_dynamics_summary["residual_f1"]
    >= _todo16_control_summary["residual_f1"]
)
TODO16_IMPROVED_SEGMENTS = int(
    (
        todo16_per_segment["dynamics_exact"]
        > todo16_per_segment["TODO11_exact"]
    ).sum()
)
TODO16_MATERIAL_IMPROVED_SEGMENTS = int(
    (
        todo16_per_segment["dynamics_failures"]
        < todo16_per_segment["TODO11_failures"]
    ).sum()
)
TODO16_FAILURE_GATE_PASSED = bool(
    _todo16_dynamics_summary["material_failures"]
    <= TODO16_MAXIMUM_MATERIAL_FAILURES
)
TODO16_F1_GATE_PASSED = bool(
    _todo16_dynamics_summary["residual_f1"]
    >= _todo15_todo11_sequential_f1
)
TODO16_GLOBAL_GATE_PASSED = bool(
    _todo16_dynamics_summary["exact_share"]
    > float(_todo11_sequential_exact.mean())
    and _todo16_dynamics_summary["material_share"]
    > float(_todo11_sequential_material.mean())
)
TODO16_SEGMENT_GATE_PASSED = bool(
    TODO16_IMPROVED_SEGMENTS >= TODO16_MINIMUM_IMPROVED_SEGMENTS
)
TODO16_ONE_STEP_GATE_PASSED = bool(
    TODO16_DYNAMICS_BEATS_CONTROL
    and TODO16_FAILURE_GATE_PASSED
    and TODO16_F1_GATE_PASSED
    and TODO16_GLOBAL_GATE_PASSED
    and TODO16_SEGMENT_GATE_PASSED
)
TODO16_STATUS = (
    "compact_dynamics_one_step_gate_passed_recursive_pending"
    if TODO16_ONE_STEP_GATE_PASSED
    else "compact_dynamics_one_step_gate_failed"
)
todo16_manifest = pd.Series(
    {
        "status": TODO16_STATUS,
        "representations": str(TODO16_REPRESENTATIONS),
        "additional dynamics features": len(_todo16_dynamic_feature_names),
        "same entry population and RF architecture": True,
        "history never crosses object/segment": True,
        "current target or actual branch used": False,
        "object2 uses predicted object1 distance": True,
        "entry training p2 teacher": "fixed base/PERIOD2 object1; no current truth",
        "past history": "observed, teacher-forced; not autonomous",
        "evaluation caveat": "conditional on frozen TODO11 stay thresholds",
        "geometry/PERIOD2/continuation frozen": True,
        "general pair models fitted": _todo16_general_pair_models_fitted,
        "entry pair models fitted": _todo16_entry_pair_models_fitted,
        "outer entry models fitted": _todo16_outer_models_fitted,
        "dynamics beats current-only control": TODO16_DYNAMICS_BEATS_CONTROL,
        "material failure gate passed": TODO16_FAILURE_GATE_PASSED,
        "residual F1 gate passed": TODO16_F1_GATE_PASSED,
        "global TODO11 improvement gate passed": TODO16_GLOBAL_GATE_PASSED,
        "exact-improved segments versus TODO11": TODO16_IMPROVED_SEGMENTS,
        "material-improved segments versus TODO11": TODO16_MATERIAL_IMPROVED_SEGMENTS,
        "segment gate passed": TODO16_SEGMENT_GATE_PASSED,
        "one-step gate passed": TODO16_ONE_STEP_GATE_PASSED,
        "recursive rollout run": False,
        "fully blind test preserved": False,
    },
    name="value",
)

assert np.isfinite(_todo16_dynamics["position_error"]).all()
assert np.isfinite(_todo16_dynamics["angle_error"]).all()
assert np.isfinite(_todo16_dynamics["distance_error"]).all()
assert not TODO16_ONE_STEP_GATE_PASSED or TODO16_FAILURE_GATE_PASSED

display(
    Markdown("#### TODO 16 — nested thresholds by representation"),
    todo16_nested_selection.style.format(precision=6),
    Markdown("#### TODO 16 — isolated representation comparison"),
    todo16_model_summary,
    Markdown("#### TODO 16 — per-segment failures"),
    todo16_per_segment.style.format(precision=6),
    Markdown("#### TODO 16 — compact-dynamics phase metrics"),
    todo16_phase_summary.style.format(precision=6),
    Markdown("#### TODO 16 — remaining material failures"),
    todo16_material_failure_summary.style.format(precision=6),
    Markdown("#### TODO 16 — compact feature importance"),
    todo16_feature_importance.head(20).style.format(precision=6),
    Markdown("#### TODO 16 manifest"),
    todo16_manifest.to_frame(),
)
