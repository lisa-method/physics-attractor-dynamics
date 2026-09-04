"""TODO 11 companion: nested-LOSO stateful special-branch selector.

The exact transition geometry from TODO 10 stays frozen.  This experiment
changes only the probability policy for the residual Random Forest: a high
threshold enters the special state and a lower threshold may keep it active.
Thresholds are selected inside every outer fold using pairwise inner models
that exclude both the outer and inner segments.
"""

from itertools import combinations


TODO11_ENTER_THRESHOLDS = tuple(np.round(np.arange(0.50, 0.96, 0.05), 2))
TODO11_STAY_THRESHOLDS = tuple(np.round(np.arange(0.15, 0.81, 0.05), 2))
TODO11_MAXIMUM_MATERIAL_FAILURES = 40
TODO11_SPECIAL_F1_GATE = 0.90
TODO11_MINIMUM_IMPROVED_SEGMENTS = 8
TODO11_RECURSIVE_MEDIAN_GATE = 190
TODO11_RANDOM_STATE = TODO9_RANDOM_STATE

_todo11_segments = np.array(
    sorted(todo9_event_dataset["segment_id"].unique()), dtype=int
)
_todo11_candidate_event_indices = todo9_gate_candidates.index.to_numpy(
    dtype=int
)
# Training still uses the frozen TODO 9 candidate population.  Inference is
# evaluated on every event: an internally active state may legitimately reach
# a row that was not a candidate under the observed-history TODO 9 mask.
_todo11_all_X = pd.get_dummies(
    todo9_event_dataset[
        list(TODO9_NUMERIC_FEATURES + TODO9_CATEGORICAL_FEATURES)
    ],
    columns=list(TODO9_CATEGORICAL_FEATURES),
    dtype=float,
).reindex(columns=TODO9_FEATURE_COLUMNS, fill_value=0.0)
_todo11_all_X = (
    _todo11_all_X.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
)
_todo11_base_candidate = (
    todo9_event_dataset["predicted_x_crossing"].to_numpy(dtype=bool)
    | todo9_event_dataset["predicted_y_crossing"].to_numpy(dtype=bool)
    | todo9_event_dataset["previous_wall_contact"].to_numpy(dtype=bool)
    | todo9_event_dataset["previous_period2_exact"].to_numpy(dtype=bool)
)
_todo11_sequence_indices = [
    np.sort(np.asarray(_indices, dtype=int))
    for _indices in todo9_event_dataset.groupby(
        ["object", "segment_id"], sort=False
    ).indices.values()
]


def _todo11_true_probability(model, feature_frame):
    _true_column = int(np.flatnonzero(model.classes_ == True)[0])
    return model.predict_proba(feature_frame)[:, _true_column]


def _todo11_fast_forest_probability(model, feature_vector):
    """Return P(True) with the same tree-probability average as sklearn."""
    _probability = np.zeros(len(model.classes_), dtype=float)
    for _estimator in model.estimators_:
        _tree = _estimator.tree_
        _node = 0
        while _tree.feature[_node] >= 0:
            _feature = _tree.feature[_node]
            _node = (
                _tree.children_left[_node]
                if feature_vector[_feature] <= _tree.threshold[_node]
                else _tree.children_right[_node]
            )
        _leaf = _tree.value[_node][0]
        _probability += _leaf / _leaf.sum()
    _true_column = int(np.flatnonzero(model.classes_ == True)[0])
    return float(_probability[_true_column] / len(model.estimators_))


def _todo11_selector(probability, enter_threshold, stay_threshold):
    """Scan each object/segment with its own predicted active state."""
    if not 0.0 <= stay_threshold <= enter_threshold <= 1.0:
        raise ValueError("Require 0 <= T_stay <= T_enter <= 1")
    _prediction = np.zeros(len(todo9_event_dataset), dtype=bool)
    for _indices in _todo11_sequence_indices:
        _active = False
        for _row in _indices:
            if _todo9_loop_prediction[_row]:
                _active = True
                continue
            _candidate = bool(_todo11_base_candidate[_row] or _active)
            if not _candidate:
                _active = False
                continue
            _threshold = stay_threshold if _active else enter_threshold
            _active = bool(probability[_row] >= _threshold)
            _prediction[_row] = _active
    return _prediction


# ---------------------------------------------------------------------------
# 11.A — fixed conditional outcomes for threshold selection
# ---------------------------------------------------------------------------
_todo11_formula_angle = _todo10_special_angle(
    todo9_event_dataset["free_angle"].to_numpy(dtype=float),
    _todo9_baseline_angle,
)
_todo11_formula_position = np.empty_like(_todo9_actual_position)
for _row_index, _angle in enumerate(_todo11_formula_angle):
    _limits = _todo8_oriented_limits(_angle)
    _todo11_formula_position[_row_index] = np.clip(
        todo9_event_dataset.loc[
            _row_index, ["free_x", "free_y"]
        ].to_numpy(dtype=float),
        -_limits,
        _limits,
    )

_todo11_baseline_angle_error = np.abs(
    _todo7_wrap_degrees(_todo9_baseline_angle - _todo9_actual_angle)
)
_todo11_baseline_position_error = np.linalg.norm(
    _todo9_baseline_position - _todo9_actual_position, axis=1
)
_todo11_formula_angle_error = np.abs(
    _todo7_wrap_degrees(_todo11_formula_angle - _todo9_actual_angle)
)
_todo11_formula_position_error = np.linalg.norm(
    _todo11_formula_position - _todo9_actual_position, axis=1
)
_todo11_loop_angle_error = np.abs(
    _todo7_wrap_degrees(_todo9_angle_lag2 - _todo9_actual_angle)
)
_todo11_loop_position_error = np.linalg.norm(
    _todo9_position_lag2 - _todo9_actual_position, axis=1
)

_todo11_baseline_exact = (
    (_todo11_baseline_angle_error < TODO8_ANGLE_TOLERANCE)
    & (_todo11_baseline_position_error < TODO8_POSITION_TOLERANCE)
)
_todo11_formula_exact = (
    (_todo11_formula_angle_error < TODO8_ANGLE_TOLERANCE)
    & (_todo11_formula_position_error < TODO8_POSITION_TOLERANCE)
)
_todo11_loop_exact = (
    (_todo11_loop_angle_error < TODO8_ANGLE_TOLERANCE)
    & (_todo11_loop_position_error < TODO8_POSITION_TOLERANCE)
)
_todo11_baseline_material = (
    (_todo11_baseline_angle_error < TODO8_MATERIAL_ANGLE_ERROR)
    & (_todo11_baseline_position_error < TODO8_MATERIAL_POSITION_ERROR)
)
_todo11_formula_material = (
    (_todo11_formula_angle_error < TODO8_MATERIAL_ANGLE_ERROR)
    & (_todo11_formula_position_error < TODO8_MATERIAL_POSITION_ERROR)
)
_todo11_loop_material = (
    (_todo11_loop_angle_error < TODO8_MATERIAL_ANGLE_ERROR)
    & (_todo11_loop_position_error < TODO8_MATERIAL_POSITION_ERROR)
)


def _todo11_conditional_success(predicted_special, kind):
    if kind == "material":
        _baseline = _todo11_baseline_material
        _formula = _todo11_formula_material
        _loop = _todo11_loop_material
    elif kind == "exact":
        _baseline = _todo11_baseline_exact
        _formula = _todo11_formula_exact
        _loop = _todo11_loop_exact
    else:
        raise ValueError(f"Unknown success kind: {kind}")
    _success = np.where(predicted_special, _formula, _baseline).copy()
    _success[_todo9_loop_prediction] = _loop[_todo9_loop_prediction]
    return _success


def _todo11_binary_f1(target, prediction):
    _target = np.asarray(target, dtype=bool)
    _prediction = np.asarray(prediction, dtype=bool)
    _tp = int(np.sum(_target & _prediction))
    _fp = int(np.sum(~_target & _prediction))
    _fn = int(np.sum(_target & ~_prediction))
    _precision = _tp / (_tp + _fp) if _tp + _fp else 0.0
    _recall = _tp / (_tp + _fn) if _tp + _fn else 0.0
    _f1 = (
        2.0 * _precision * _recall / (_precision + _recall)
        if _precision + _recall
        else 0.0
    )
    return _precision, _recall, _f1


# ---------------------------------------------------------------------------
# 11.B — outer probabilities and pairwise nested-LOSO probabilities
# ---------------------------------------------------------------------------
_todo11_outer_probability = np.zeros(len(todo9_event_dataset), dtype=float)
for _segment_id in _todo11_segments:
    _event_mask = todo9_event_dataset["segment_id"].eq(
        _segment_id
    ).to_numpy()
    _model = todo9_residual_models[int(_segment_id)]
    _todo11_outer_probability[_event_mask] = _todo11_true_probability(
        _model, _todo11_all_X.loc[_event_mask]
    )

_todo11_nested_probability = {
    int(_outer): np.zeros(len(todo9_event_dataset), dtype=float)
    for _outer in _todo11_segments
}
_todo11_pair_models_fitted = 0

for _first, _second in combinations(_todo11_segments, 2):
    _train_mask = (
        (_todo9_groups != _first) & (_todo9_groups != _second)
    )
    _pair_model = RandomForestClassifier(
        n_estimators=TODO9_RF_ESTIMATORS,
        max_depth=TODO9_RF_MAX_DEPTH,
        min_samples_leaf=TODO9_RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=TODO11_RANDOM_STATE,
        n_jobs=-1,
    )
    _pair_model.fit(
        todo9_gate_X.loc[_train_mask],
        _todo9_residual_y[_train_mask],
    )
    _todo11_pair_models_fitted += 1

    for _outer, _inner in ((_first, _second), (_second, _first)):
        _inner_event_mask = todo9_event_dataset["segment_id"].eq(
            _inner
        ).to_numpy()
        _todo11_nested_probability[int(_outer)][_inner_event_mask] = (
            _todo11_true_probability(
                _pair_model, _todo11_all_X.loc[_inner_event_mask]
            )
        )

assert _todo11_pair_models_fitted == 55
assert np.isfinite(_todo11_outer_probability).all()
assert np.all(
    (_todo11_outer_probability >= 0.0)
    & (_todo11_outer_probability <= 1.0)
)

# The fast single-row traversal is later used by object 2 and recursive
# rollout.  Verify probabilities, not only hard classes, in every outer fold.
for _segment_id, _model in todo9_residual_models.items():
    _fold_rows = np.flatnonzero(
        todo9_event_dataset["segment_id"].eq(_segment_id).to_numpy()
    )[:4]
    for _row in _fold_rows:
        _frame = _todo11_all_X.iloc[[_row]]
        _reference = float(_todo11_true_probability(_model, _frame)[0])
        _fast = _todo11_fast_forest_probability(
            _model, _frame.iloc[0].to_numpy(dtype=np.float32)
        )
        assert np.isclose(_fast, _reference, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# 11.C — choose enter/stay thresholds strictly inside every outer fold
#
# This is deliberately the pre-registered object-level conditional surrogate:
# it scores the branch choice with each event's causal baseline held fixed.
# The accepted outer result below is nevertheless the full sequential
# object1 -> predicted distance -> object2 one-step simulation.
# ---------------------------------------------------------------------------
_todo11_threshold_rows = []
todo11_threshold_by_segment = {}

for _outer in _todo11_segments:
    _validation_segments = _todo11_segments[_todo11_segments != _outer]
    _validation_mask = todo9_event_dataset["segment_id"].isin(
        _validation_segments
    ).to_numpy()
    _probability = _todo11_nested_probability[int(_outer)]
    _best = None

    for _enter in TODO11_ENTER_THRESHOLDS:
        for _stay in TODO11_STAY_THRESHOLDS:
            if _stay > _enter:
                continue
            _prediction = _todo11_selector(_probability, _enter, _stay)
            _material_success = _todo11_conditional_success(
                _prediction, "material"
            )
            _exact_success = _todo11_conditional_success(
                _prediction, "exact"
            )
            _material_error_by_segment = []
            _exact_error_by_segment = []
            for _inner in _validation_segments:
                _inner_mask = todo9_event_dataset["segment_id"].eq(
                    _inner
                ).to_numpy()
                _material_error_by_segment.append(
                    1.0 - float(_material_success[_inner_mask].mean())
                )
                _exact_error_by_segment.append(
                    1.0 - float(_exact_success[_inner_mask].mean())
                )
            _, _, _f1 = _todo11_binary_f1(
                _todo9_residual_target_all[_validation_mask],
                _prediction[_validation_mask],
            )
            _candidate = {
                "enter": float(_enter),
                "stay": float(_stay),
                "macro_material_error": float(
                    np.mean(_material_error_by_segment)
                ),
                "macro_exact_error": float(np.mean(_exact_error_by_segment)),
                "residual_f1": float(_f1),
            }
            _key = (
                _candidate["macro_material_error"],
                -_candidate["residual_f1"],
                _candidate["macro_exact_error"],
                -_candidate["enter"],
                _candidate["stay"],
            )
            if _best is None or _key < _best[0]:
                _best = (_key, _candidate)

    _chosen = _best[1]
    todo11_threshold_by_segment[int(_outer)] = (
        _chosen["enter"],
        _chosen["stay"],
    )
    _todo11_threshold_rows.append(
        {"segment_id": int(_outer), **_chosen}
    )

todo11_threshold_selection = pd.DataFrame(
    _todo11_threshold_rows
).set_index("segment_id")


# ---------------------------------------------------------------------------
# 11.D — outer-LOSO event and sequential one-step evaluation
# ---------------------------------------------------------------------------
_todo11_event_prediction = np.zeros(len(todo9_event_dataset), dtype=bool)
for _segment_id in _todo11_segments:
    _segment_mask = todo9_event_dataset["segment_id"].eq(
        _segment_id
    ).to_numpy()
    _enter, _stay = todo11_threshold_by_segment[int(_segment_id)]
    _todo11_event_prediction[_segment_mask] = _todo11_selector(
        _todo11_outer_probability, _enter, _stay
    )[_segment_mask]

(
    _todo11_event_precision,
    _todo11_event_recall,
    _todo11_event_f1,
) = _todo11_binary_f1(
    _todo9_residual_target_all, _todo11_event_prediction
)
todo11_event_summary = pd.Series(
    {
        "target residual-special rows": int(
            _todo9_residual_target_all.sum()
        ),
        "predicted residual-special rows": int(
            _todo11_event_prediction.sum()
        ),
        "precision": _todo11_event_precision,
        "recall": _todo11_event_recall,
        "F1": _todo11_event_f1,
        "TODO9 fixed-0.5 F1": float(todo9_residual_gate_summary["F1"]),
        "pairwise inner models fitted": _todo11_pair_models_fitted,
    },
    name="value",
)

_todo11_conditional_angle = _todo9_baseline_angle.copy()
_todo11_conditional_position = _todo9_baseline_position.copy()
_todo11_conditional_angle[_todo11_event_prediction] = (
    _todo11_formula_angle[_todo11_event_prediction]
)
_todo11_conditional_position[_todo11_event_prediction] = (
    _todo11_formula_position[_todo11_event_prediction]
)
_todo11_conditional_angle[_todo9_loop_prediction] = _todo9_angle_lag2[
    _todo9_loop_prediction
]
_todo11_conditional_position[_todo9_loop_prediction] = (
    _todo9_position_lag2[_todo9_loop_prediction]
)

_todo11_p1_prediction = _todo11_conditional_position[
    _todo9_p1_event_indices
].copy()
_todo11_angle1_prediction = _todo11_conditional_angle[
    _todo9_p1_event_indices
].copy()
_todo11_distance_prediction = np.linalg.norm(
    _todo11_p1_prediction - _todo9_previous_p2, axis=1
)

_todo11_p2_prediction = np.empty_like(_todo9_previous_p2)
_todo11_angle2_prediction = np.empty(
    len(_todo9_transition_indices), dtype=float
)
_todo11_p2_probability = np.zeros(
    len(_todo9_transition_indices), dtype=float
)
_todo11_p2_special = np.zeros(
    len(_todo9_transition_indices), dtype=bool
)
_todo11_p2_active_by_segment = {
    int(_segment_id): False for _segment_id in _todo11_segments
}

for _transition_row, (_data_index, _event_index, _driver_distance) in enumerate(
    zip(
        _todo9_transition_indices,
        _todo9_p2_event_indices,
        _todo11_distance_prediction,
    )
):
    _event = todo9_event_dataset.loc[_event_index]
    _segment_id = int(_event["segment_id"])
    _active = _todo11_p2_active_by_segment[_segment_id]
    if bool(_event["predict_period2_from_previous"]):
        _lag2_index = max(int(_data_index) - 2, 0)
        _todo11_p2_prediction[_transition_row] = todo7_p2[_lag2_index]
        _todo11_angle2_prediction[_transition_row] = todo7_angle2[_lag2_index]
        _todo11_p2_active_by_segment[_segment_id] = True
        continue

    _baseline = _todo8_update_object(
        todo7_p2[_data_index - 1],
        todo7_angle2[_data_index - 1],
        float(_driver_distance),
        -1.0,
    )
    _todo11_p2_prediction[_transition_row] = _baseline["position"]
    _todo11_angle2_prediction[_transition_row] = _baseline["angle"]
    _candidate = bool(
        np.any(_baseline["crossing"])
        or _event["previous_wall_contact"]
        or _event["previous_period2_exact"]
        or _active
    )
    if not _candidate:
        _todo11_p2_active_by_segment[_segment_id] = False
        continue

    _record = _todo9_feature_record(
        _event_index, _baseline, _driver_distance
    )
    _vector = _todo10_feature_vector(_record)
    _model = todo9_residual_models[_segment_id]
    _probability = _todo11_fast_forest_probability(_model, _vector)
    _todo11_p2_probability[_transition_row] = _probability
    _enter, _stay = todo11_threshold_by_segment[_segment_id]
    _threshold = _stay if _active else _enter
    if _probability < _threshold:
        _todo11_p2_active_by_segment[_segment_id] = False
        continue

    _todo11_p2_special[_transition_row] = True
    _todo11_p2_active_by_segment[_segment_id] = True
    _special_angle = _todo10_special_angle(
        _baseline["free_angle"], _baseline["angle"]
    )
    _limits = _todo8_oriented_limits(_special_angle)
    _todo11_p2_prediction[_transition_row] = np.clip(
        _baseline["free_position"], -_limits, _limits
    )
    _todo11_angle2_prediction[_transition_row] = _special_angle

_todo11_p1_error = np.linalg.norm(
    _todo11_p1_prediction - _todo9_truth_p1, axis=1
)
_todo11_p2_error = np.linalg.norm(
    _todo11_p2_prediction - _todo9_truth_p2, axis=1
)
_todo11_maximum_position_error = np.maximum(
    _todo11_p1_error, _todo11_p2_error
)
_todo11_angle1_error = np.abs(
    _todo7_wrap_degrees(
        _todo11_angle1_prediction - _todo9_truth_angle1
    )
)
_todo11_angle2_error = np.abs(
    _todo7_wrap_degrees(
        _todo11_angle2_prediction - _todo9_truth_angle2
    )
)
_todo11_maximum_angle_error = np.maximum(
    _todo11_angle1_error, _todo11_angle2_error
)
_todo11_distance_error = np.abs(
    _todo11_distance_prediction - _todo9_truth_distance
)
_todo11_sequential_exact = (
    (_todo11_maximum_position_error < TODO8_POSITION_TOLERANCE)
    & (_todo11_maximum_angle_error < TODO8_ANGLE_TOLERANCE)
    & (_todo11_distance_error < TODO8_DISTANCE_TOLERANCE)
)
_todo11_sequential_material = (
    (_todo11_maximum_position_error < TODO8_MATERIAL_POSITION_ERROR)
    & (_todo11_maximum_angle_error < TODO8_MATERIAL_ANGLE_ERROR)
    & (_todo11_distance_error < TODO8_MATERIAL_DISTANCE_ERROR)
)

todo11_one_step_summary = pd.Series(
    {
        "TODO10 exact complete-state share": float(
            _todo10_sequential_exact.mean()
        ),
        "TODO11 exact complete-state share": float(
            _todo11_sequential_exact.mean()
        ),
        "TODO10 material complete-state share": float(
            _todo10_sequential_material.mean()
        ),
        "TODO11 material complete-state share": float(
            _todo11_sequential_material.mean()
        ),
        "TODO10 material failures": int(
            (~_todo10_sequential_material).sum()
        ),
        "TODO11 material failures": int(
            (~_todo11_sequential_material).sum()
        ),
        "TODO10 maximum-angle MAE": float(
            _todo10_maximum_angle_error.mean()
        ),
        "TODO11 maximum-angle MAE": float(
            _todo11_maximum_angle_error.mean()
        ),
    },
    name="value",
)

_todo11_segment_rows = []
for _segment_id in _todo11_segments:
    _mask = _todo9_transition_segments == _segment_id
    _todo11_segment_rows.append(
        {
            "segment_id": int(_segment_id),
            "transitions": int(_mask.sum()),
            "enter": todo11_threshold_by_segment[int(_segment_id)][0],
            "stay": todo11_threshold_by_segment[int(_segment_id)][1],
            "TODO10_exact": float(_todo10_sequential_exact[_mask].mean()),
            "TODO11_exact": float(_todo11_sequential_exact[_mask].mean()),
            "TODO10_material": float(
                _todo10_sequential_material[_mask].mean()
            ),
            "TODO11_material": float(
                _todo11_sequential_material[_mask].mean()
            ),
        }
    )
todo11_one_step_per_segment = pd.DataFrame(
    _todo11_segment_rows
).set_index("segment_id")

TODO11_IMPROVED_SEGMENTS = int(
    (
        todo11_one_step_per_segment["TODO11_exact"]
        > todo11_one_step_per_segment["TODO10_exact"]
    ).sum()
)
TODO11_FAILURE_GATE_PASSED = bool(
    todo11_one_step_summary["TODO11 material failures"]
    <= TODO11_MAXIMUM_MATERIAL_FAILURES
)
TODO11_F1_GATE_PASSED = bool(
    todo11_event_summary["F1"] >= TODO11_SPECIAL_F1_GATE
)
TODO11_GLOBAL_ONE_STEP_IMPROVED = bool(
    todo11_one_step_summary["TODO11 exact complete-state share"]
    > todo11_one_step_summary["TODO10 exact complete-state share"]
    and todo11_one_step_summary["TODO11 material complete-state share"]
    > todo11_one_step_summary["TODO10 material complete-state share"]
)
TODO11_SEGMENT_GATE_PASSED = bool(
    TODO11_IMPROVED_SEGMENTS >= TODO11_MINIMUM_IMPROVED_SEGMENTS
)
TODO11_ONE_STEP_GATE_PASSED = bool(
    TODO11_FAILURE_GATE_PASSED
    and TODO11_F1_GATE_PASSED
    and TODO11_GLOBAL_ONE_STEP_IMPROVED
    and TODO11_SEGMENT_GATE_PASSED
)


# ---------------------------------------------------------------------------
# 11.E — remaining material-error taxonomy
# ---------------------------------------------------------------------------
_todo11_failure_rows = []
_todo11_p1_special = _todo11_event_prediction[_todo9_p1_event_indices]
for _transition_row in np.flatnonzero(~_todo11_sequential_material):
    _p1_event = _todo9_p1_event_indices[_transition_row]
    _p2_event = _todo9_p2_event_indices[_transition_row]
    _causes = []
    if (
        todo9_event_dataset.at[_p1_event, "event_mode"]
        == "segment_start_special"
        or todo9_event_dataset.at[_p2_event, "event_mode"]
        == "segment_start_special"
    ):
        _causes.append("segment-start hidden state")
    if (
        (_todo11_p1_special[_transition_row] and not _todo10_p1_target_residual[_transition_row])
        or (_todo11_p2_special[_transition_row] and not _todo10_p2_target_residual[_transition_row])
    ):
        _causes.append("stateful gate false positive")
    if (
        (_todo10_p1_target_residual[_transition_row] and not _todo11_p1_special[_transition_row])
        or (_todo10_p2_target_residual[_transition_row] and not _todo11_p2_special[_transition_row])
    ):
        _causes.append("stateful gate false negative")
    if (
        (_todo10_p1_target_loop[_transition_row] and not _todo10_p1_loop_predicted[_transition_row])
        or (_todo10_p2_target_loop[_transition_row] and not _todo10_p2_loop_predicted[_transition_row])
    ):
        _causes.append("period-2 entry not yet observable")
    if not _causes:
        _causes.append("upstream or mixed sequential error")
    _todo11_failure_rows.append(
        {
            "segment_id": int(_todo9_transition_segments[_transition_row]),
            "source_row": int(_todo9_transition_source_rows[_transition_row]),
            "cause": " + ".join(_causes),
            "maximum_position_error": float(
                _todo11_maximum_position_error[_transition_row]
            ),
            "maximum_angle_error": float(
                _todo11_maximum_angle_error[_transition_row]
            ),
        }
    )

todo11_material_failures = pd.DataFrame(
    _todo11_failure_rows,
    columns=[
        "segment_id",
        "source_row",
        "cause",
        "maximum_position_error",
        "maximum_angle_error",
    ],
)
todo11_material_failure_summary = (
    todo11_material_failures.groupby("cause", as_index=True)
    .agg(
        transitions=("source_row", "size"),
        segments=("segment_id", "nunique"),
        maximum_angle_error=("maximum_angle_error", "max"),
        maximum_position_error=("maximum_position_error", "max"),
    )
    .sort_values("transitions", ascending=False)
)


# ---------------------------------------------------------------------------
# 11.F — recursive stateful simulator (only after the outer one-step gate)
# ---------------------------------------------------------------------------
def _todo11_recursive_object_update(
    object_name,
    segment_id,
    previous_position,
    previous_angle,
    driver_distance,
    angle_sign,
    history,
    model,
    enter_threshold,
    stay_threshold,
):
    _baseline = _todo8_update_object(
        previous_position,
        previous_angle,
        driver_distance,
        angle_sign,
    )
    _features = _todo10_recursive_feature_record(
        object_name,
        segment_id,
        previous_position,
        previous_angle,
        driver_distance,
        _baseline,
        history,
    )
    _active = bool(history["special"][-1])
    _period2 = bool(
        _features["previous_period2_exact"]
        and _features["previous_wall_contact"]
        and _features["previous_boundary_run_length"] >= 4
    )
    _probability = 0.0
    if _period2:
        _position = np.asarray(history["position"][-2], dtype=float).copy()
        _angle = float(history["angle"][-2])
        _branch = "corner/loop"
        _mode = "period2_wall_loop"
        _special = True
    else:
        _candidate = bool(
            np.any(_baseline["crossing"])
            or _features["previous_wall_contact"]
            or _features["previous_period2_exact"]
            or _active
        )
        if _candidate:
            _probability = _todo11_fast_forest_probability(
                model, _todo10_feature_vector(_features)
            )
        _threshold = stay_threshold if _active else enter_threshold
        _special = bool(_candidate and _probability >= _threshold)
        if _special:
            _angle = _todo10_special_angle(
                _baseline["free_angle"], _baseline["angle"]
            )
            _limits = _todo8_oriented_limits(_angle)
            _position = np.clip(
                _baseline["free_position"], -_limits, _limits
            )
            _branch = "corner/loop"
            _mode = "wall_loop_special"
        else:
            _position = _baseline["position"].copy()
            _angle = float(_baseline["angle"])
            _branch = str(_baseline["branch"])
            _mode = _branch

    _, _, _wall = _todo10_wall_state(_position, _angle)
    _boundary_run = history["boundary_run"][-1] + 1 if _wall else 0
    history["position"].append(_position.copy())
    history["angle"].append(float(_angle))
    history["branch"].append(_branch)
    history["mode"].append(_mode)
    history["special"].append(bool(_special))
    history["boundary_run"].append(int(_boundary_run))
    return _position, float(_angle), _mode, _probability


def _todo11_rollout_segment(segment_frame, segment_id):
    _segment_frame = segment_frame.sort_index()
    _truth_p1 = _segment_frame[["x1", "y1"]].to_numpy(dtype=float)
    _truth_p2 = _segment_frame[["x2", "y2"]].to_numpy(dtype=float)
    _truth_a1 = _segment_frame["angle1"].to_numpy(dtype=float)
    _truth_a2 = _segment_frame["angle2"].to_numpy(dtype=float)
    _truth_distance = _segment_frame["distance"].to_numpy(dtype=float)
    _model = todo9_residual_models[int(segment_id)]
    _enter, _stay = todo11_threshold_by_segment[int(segment_id)]
    _h1 = _todo10_initial_history(_truth_p1[0], _truth_a1[0])
    _h2 = _todo10_initial_history(_truth_p2[0], _truth_a2[0])
    _distance = [float(_truth_distance[0])]
    _mode1 = ["segment_start"]
    _mode2 = ["segment_start"]
    _probability1 = [0.0]
    _probability2 = [0.0]

    for _ in range(1, len(_segment_frame)):
        _rho_before = float(
            np.linalg.norm(_h2["position"][-1] - _h1["position"][-1])
        )
        _p1, _, _m1, _pr1 = _todo11_recursive_object_update(
            "object1",
            int(segment_id),
            _h1["position"][-1],
            _h1["angle"][-1],
            _rho_before,
            1.0,
            _h1,
            _model,
            _enter,
            _stay,
        )
        _new_distance = float(np.linalg.norm(_p1 - _h2["position"][-1]))
        _, _, _m2, _pr2 = _todo11_recursive_object_update(
            "object2",
            int(segment_id),
            _h2["position"][-1],
            _h2["angle"][-1],
            _new_distance,
            -1.0,
            _h2,
            _model,
            _enter,
            _stay,
        )
        _distance.append(_new_distance)
        _mode1.append(_m1)
        _mode2.append(_m2)
        _probability1.append(_pr1)
        _probability2.append(_pr2)

    _predicted_p1 = np.asarray(_h1["position"])
    _predicted_p2 = np.asarray(_h2["position"])
    _predicted_a1 = np.asarray(_h1["angle"])
    _predicted_a2 = np.asarray(_h2["angle"])
    _predicted_distance = np.asarray(_distance)
    _position_error = np.maximum(
        np.linalg.norm(_predicted_p1[1:] - _truth_p1[1:], axis=1),
        np.linalg.norm(_predicted_p2[1:] - _truth_p2[1:], axis=1),
    )
    _angle_error = np.maximum(
        np.abs(_todo7_wrap_degrees(_predicted_a1[1:] - _truth_a1[1:])),
        np.abs(_todo7_wrap_degrees(_predicted_a2[1:] - _truth_a2[1:])),
    )
    _distance_error = np.abs(_predicted_distance[1:] - _truth_distance[1:])
    _exact_failure = (
        (_position_error >= TODO8_POSITION_TOLERANCE)
        | (_angle_error >= TODO8_ANGLE_TOLERANCE)
        | (_distance_error >= TODO8_DISTANCE_TOLERANCE)
    )
    _material_failure = (
        (_position_error >= TODO8_MATERIAL_POSITION_ERROR)
        | (_angle_error >= TODO8_MATERIAL_ANGLE_ERROR)
        | (_distance_error >= TODO8_MATERIAL_DISTANCE_ERROR)
    )
    return {
        "predicted_positions": np.column_stack(
            (_predicted_p1, _predicted_p2)
        ),
        "truth_positions": np.column_stack((_truth_p1, _truth_p2)),
        "position_error": _position_error,
        "angle_error": _angle_error,
        "distance_error": _distance_error,
        "mode1": _mode1,
        "mode2": _mode2,
        "probability1": np.asarray(_probability1),
        "probability2": np.asarray(_probability2),
        "first_numerical_failure": _todo8_first_failure(_exact_failure),
        "first_material_failure": _todo8_first_failure(_material_failure),
    }


todo11_rollouts = {}
_todo11_rollout_rows = []
_todo11_horizon_rows = []

if TODO11_ONE_STEP_GATE_PASSED:
    for _segment_id, _segment_frame in todo7_df.groupby(
        "segment_id", sort=True
    ):
        _rollout = _todo11_rollout_segment(_segment_frame, _segment_id)
        todo11_rollouts[int(_segment_id)] = _rollout
        _transition_count = len(_segment_frame) - 1
        _first_material = _rollout["first_material_failure"]
        _first_exact = _rollout["first_numerical_failure"]
        _todo11_rollout_rows.append(
            {
                "segment_id": int(_segment_id),
                "transitions": _transition_count,
                "first numerical failure": (
                    _transition_count + 1 if _first_exact is None else _first_exact
                ),
                "first material failure": (
                    _transition_count + 1 if _first_material is None else _first_material
                ),
                "finite": bool(
                    np.isfinite(_rollout["predicted_positions"]).all()
                ),
            }
        )
        for _label, _horizon in [
            *[
                (str(_h), _h)
                for _h in TODO8_ROLLOUT_HORIZONS
                if _h <= _transition_count
            ],
            ("full", _transition_count),
        ]:
            _truth = _rollout["truth_positions"][1 : _horizon + 1]
            _stateful = _rollout["predicted_positions"][1 : _horizon + 1]
            _todo10_positions = todo10_rollouts[int(_segment_id)][
                "predicted_positions"
            ][1 : _horizon + 1]
            _constant = np.broadcast_to(
                _rollout["truth_positions"][0], _truth.shape
            )
            _todo11_horizon_rows.append(
                {
                    "segment_id": int(_segment_id),
                    "horizon": _label,
                    "TODO11 RMSE": float(
                        np.sqrt(np.mean((_stateful - _truth) ** 2))
                    ),
                    "TODO10 RMSE": float(
                        np.sqrt(np.mean((_todo10_positions - _truth) ** 2))
                    ),
                    "constant RMSE": float(
                        np.sqrt(np.mean((_constant - _truth) ** 2))
                    ),
                }
            )

todo11_rollout_summary = pd.DataFrame(_todo11_rollout_rows)
if len(todo11_rollout_summary):
    todo11_rollout_summary = todo11_rollout_summary.set_index("segment_id")
todo11_rollout_horizons = pd.DataFrame(_todo11_horizon_rows)
todo11_rollout_macro = (
    todo11_rollout_horizons.groupby("horizon", sort=False)
    .agg(
        segments=("segment_id", "nunique"),
        TODO11_RMSE=("TODO11 RMSE", "mean"),
        TODO10_RMSE=("TODO10 RMSE", "mean"),
        constant_RMSE=("constant RMSE", "mean"),
    )
    if len(todo11_rollout_horizons)
    else pd.DataFrame()
)

TODO11_RECURSIVE_ROLLOUT_RUN = bool(TODO11_ONE_STEP_GATE_PASSED)
TODO11_MEDIAN_MATERIAL_PREFIX = (
    float(todo11_rollout_summary["first material failure"].median())
    if TODO11_RECURSIVE_ROLLOUT_RUN
    else np.nan
)
TODO11_ROLLOUT_GATE_PASSED = bool(
    TODO11_RECURSIVE_ROLLOUT_RUN
    and TODO11_MEDIAN_MATERIAL_PREFIX >= TODO11_RECURSIVE_MEDIAN_GATE
    and todo11_rollout_macro.loc["100", "TODO11_RMSE"]
    < todo11_rollout_macro.loc["100", "TODO10_RMSE"]
    and todo11_rollout_macro.loc["500", "TODO11_RMSE"]
    < todo11_rollout_macro.loc["500", "TODO10_RMSE"]
    and todo11_rollout_macro.loc["full", "TODO11_RMSE"]
    <= todo11_rollout_macro.loc["full", "TODO10_RMSE"]
)

TODO11_STATUS = (
    "stateful_selector_all_gates_passed"
    if TODO11_ROLLOUT_GATE_PASSED
    else "stateful_selector_one_step_passed_rollout_gate_failed"
    if TODO11_RECURSIVE_ROLLOUT_RUN
    else "stateful_selector_one_step_gate_failed_rollout_closed"
)

todo11_manifest = pd.Series(
    {
        "status": TODO11_STATUS,
        "pairwise inner models": _todo11_pair_models_fitted,
        "outer segments": len(_todo11_segments),
        "thresholds selected with outer labels": False,
        "active state uses previous target special label": False,
        "exact special-angle formula changed": False,
        "maximum material failures": TODO11_MAXIMUM_MATERIAL_FAILURES,
        "material failure gate passed": TODO11_FAILURE_GATE_PASSED,
        "special F1 gate": TODO11_SPECIAL_F1_GATE,
        "special F1 gate passed": TODO11_F1_GATE_PASSED,
        "segments improved": TODO11_IMPROVED_SEGMENTS,
        "minimum improved segments": TODO11_MINIMUM_IMPROVED_SEGMENTS,
        "one-step gate passed": TODO11_ONE_STEP_GATE_PASSED,
        "recursive rollout run": TODO11_RECURSIVE_ROLLOUT_RUN,
        "median first material failure": TODO11_MEDIAN_MATERIAL_PREFIX,
        "recursive median gate": TODO11_RECURSIVE_MEDIAN_GATE,
        "rollout gate passed": TODO11_ROLLOUT_GATE_PASSED,
        "pristine test": False,
    },
    name="value",
)

assert all(
    _stay <= _enter
    for _enter, _stay in todo11_threshold_by_segment.values()
)
assert set(todo11_threshold_by_segment) == set(_todo11_segments)
for _outer, _probability in _todo11_nested_probability.items():
    _inner_mask = todo9_event_dataset["segment_id"].ne(_outer).to_numpy()
    assert np.isfinite(_probability[_inner_mask]).all()
    assert np.all(
        (_probability[_inner_mask] >= 0.0)
        & (_probability[_inner_mask] <= 1.0)
    )
assert not TODO11_RECURSIVE_ROLLOUT_RUN or TODO11_ONE_STEP_GATE_PASSED
if TODO11_RECURSIVE_ROLLOUT_RUN:
    assert todo11_rollout_summary["finite"].all()

display(
    Markdown("#### Nested-LOSO threshold selection"),
    todo11_threshold_selection.style.format(precision=6),
    Markdown("#### Outer-LOSO residual event classification"),
    todo11_event_summary.to_frame(),
    Markdown("#### Sequential outer-LOSO one-step"),
    todo11_one_step_summary.to_frame(),
    todo11_one_step_per_segment.style.format(precision=6),
    Markdown("#### Remaining material failures"),
    todo11_material_failure_summary.style.format(precision=6),
    Markdown("#### Recursive rollout"),
    todo11_rollout_summary.style.format(precision=6)
    if len(todo11_rollout_summary)
    else todo11_rollout_summary,
    todo11_rollout_macro.style.format(precision=6)
    if len(todo11_rollout_macro)
    else todo11_rollout_macro,
    Markdown("#### TODO 11 manifest"),
    todo11_manifest.to_frame(),
)
