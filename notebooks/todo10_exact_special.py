"""TODO 10 companion: exact special-angle law, error audit, and rollout gate.

Executed after :mod:`todo9_hybrid` from ``attractor_todo.ipynb``.  TODO 10
keeps the TODO 9 leave-one-segment-out mode selector fixed and changes only
the previously approximate special-angle magnitude.  Recursive evaluation is
opened only if the preregistered numerical-exact one-step gate is passed.
"""


TODO10_EXACT_COMPLETE_STATE_GATE = TODO9_EXACT_COMPLETE_STATE_GATE
TODO10_MINIMUM_IMPROVED_SEGMENTS = 8
TODO10_ROLLOUT_MEDIAN_MATERIAL_GATE = 178
TODO10_ROLLOUT_HORIZONS = TODO8_ROLLOUT_HORIZONS
TODO10_SPECIAL_GEOMETRY_RATIO = 2.0 / TODO7_BASE_STEP


# ---------------------------------------------------------------------------
# 10.A — exact special-angle identity
# ---------------------------------------------------------------------------
def _todo10_axis_distance(angle_degrees):
    """Unsigned angular distance to the nearest 90-degree axis."""
    _angle = np.asarray(angle_degrees, dtype=float)
    return np.abs((_angle + 45.0) % 90.0 - 45.0)


def _todo10_special_theta(free_angle):
    """Exact magnitude of the special angle found from the event geometry."""
    _phi = np.deg2rad(_todo10_axis_distance(free_angle))
    return np.rad2deg(
        np.arctan2(TODO10_SPECIAL_GEOMETRY_RATIO, np.cos(_phi))
    )


def _todo10_special_angle(free_angle, reference_angle):
    """Choose the member of ``k*90 ± theta(free_angle)`` nearest a reference."""
    _free = np.atleast_1d(np.asarray(free_angle, dtype=float))
    _reference = np.atleast_1d(np.asarray(reference_angle, dtype=float))
    _free, _reference = np.broadcast_arrays(_free, _reference)
    _theta = _todo10_special_theta(_free)
    _grid = np.stack(
        [
            _todo7_wrap_degrees(90.0 * _k + _sign * _theta)
            for _k in range(-2, 3)
            for _sign in (-1.0, 1.0)
        ],
        axis=1,
    )
    _errors = np.abs(
        _todo7_wrap_degrees(_reference[:, None] - _grid)
    )
    _result = _grid[np.arange(len(_grid)), np.argmin(_errors, axis=1)]
    return float(_result[0]) if np.ndim(free_angle) == 0 else _result


_todo10_theta_actual = np.abs(_todo9_actual_axis_residual)
_todo10_theta_formula = _todo10_special_theta(
    todo9_event_dataset["free_angle"].to_numpy(dtype=float)
)
_todo10_wall_special_mask = (
    _todo9_residual_target_all
    & ~todo9_event_dataset["event_mode"]
    .eq("segment_start_special")
    .to_numpy()
)
_todo10_identity_ratio = (
    np.tan(np.deg2rad(_todo10_theta_actual))
    * np.cos(
        np.deg2rad(
            _todo10_axis_distance(
                todo9_event_dataset["free_angle"].to_numpy(dtype=float)
            )
        )
    )
)
_todo10_theta_error = np.abs(
    _todo10_theta_formula - _todo10_theta_actual
)

todo10_angle_identity_summary = pd.Series(
    {
        "wall-special rows": int(_todo10_wall_special_mask.sum()),
        "identity ratio": TODO10_SPECIAL_GEOMETRY_RATIO,
        "mean observed ratio": float(
            _todo10_identity_ratio[_todo10_wall_special_mask].mean()
        ),
        "maximum |observed ratio - 0.2|": float(
            np.max(
                np.abs(
                    _todo10_identity_ratio[_todo10_wall_special_mask]
                    - TODO10_SPECIAL_GEOMETRY_RATIO
                )
            )
        ),
        "theta MAE, degrees": float(
            _todo10_theta_error[_todo10_wall_special_mask].mean()
        ),
        "theta maximum error, degrees": float(
            _todo10_theta_error[_todo10_wall_special_mask].max()
        ),
        "numerically exact theta share": float(
            (
                _todo10_theta_error[_todo10_wall_special_mask]
                < TODO8_ANGLE_TOLERANCE
            ).mean()
        ),
    },
    name="value",
)


# ---------------------------------------------------------------------------
# 10.B — classify the 69 TODO 9 material failures
# ---------------------------------------------------------------------------
_todo10_p2_gate_by_transition = np.zeros(
    len(_todo9_transition_indices), dtype=bool
)
if len(todo9_p2_candidates):
    _todo10_p2_gate_by_transition[
        todo9_p2_candidates.loc[
            _todo9_p2_gate_prediction, "transition_row"
        ].to_numpy(dtype=int)
    ] = True

_todo10_p1_target_residual = _todo9_residual_target_all[
    _todo9_p1_event_indices
]
_todo10_p2_target_residual = _todo9_residual_target_all[
    _todo9_p2_event_indices
]
_todo10_p1_target_loop = _todo9_loop_target[
    _todo9_p1_event_indices
]
_todo10_p2_target_loop = _todo9_loop_target[
    _todo9_p2_event_indices
]
_todo10_p1_loop_predicted = _todo9_loop_prediction[
    _todo9_p1_event_indices
]
_todo10_p2_loop_predicted = _todo9_loop_prediction[
    _todo9_p2_event_indices
]
_todo10_p1_gate = _todo9_residual_oof[_todo9_p1_event_indices]
_todo10_p2_gate = _todo10_p2_gate_by_transition

_todo10_material_failure_rows = []
for _transition_row in np.flatnonzero(~_todo9_sequential_material):
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
        (_todo10_p1_target_loop[_transition_row] and not _todo10_p1_loop_predicted[_transition_row])
        or (_todo10_p2_target_loop[_transition_row] and not _todo10_p2_loop_predicted[_transition_row])
    ):
        _causes.append("period-2 entry not yet observable")
    if (
        (_todo10_p1_gate[_transition_row] and not _todo10_p1_target_residual[_transition_row])
        or (_todo10_p2_gate[_transition_row] and not _todo10_p2_target_residual[_transition_row])
    ):
        _causes.append("residual gate false positive")
    if (
        (_todo10_p1_target_residual[_transition_row] and not _todo10_p1_gate[_transition_row])
        or (_todo10_p2_target_residual[_transition_row] and not _todo10_p2_gate[_transition_row])
    ):
        _causes.append("residual gate false negative")
    if not _causes:
        _causes.append("upstream or mixed sequential error")
    _todo10_material_failure_rows.append(
        {
            "transition_row": int(_transition_row),
            "segment_id": int(_todo9_transition_segments[_transition_row]),
            "source_row": int(_todo9_transition_source_rows[_transition_row]),
            "object1_mode": str(
                todo9_event_dataset.at[_p1_event, "event_mode"]
            ),
            "object2_mode": str(
                todo9_event_dataset.at[_p2_event, "event_mode"]
            ),
            "cause": " + ".join(_causes),
            "maximum_position_error": float(
                _todo9_maximum_position_error[_transition_row]
            ),
            "maximum_angle_error": float(
                _todo9_maximum_angle_error[_transition_row]
            ),
            "distance_error": float(
                _todo9_distance_error[_transition_row]
            ),
        }
    )

todo10_material_failures = pd.DataFrame(_todo10_material_failure_rows)
todo10_material_failure_summary = (
    todo10_material_failures.groupby("cause", as_index=True)
    .agg(
        transitions=("source_row", "size"),
        segments=("segment_id", "nunique"),
        maximum_angle_error=("maximum_angle_error", "max"),
        maximum_position_error=("maximum_position_error", "max"),
    )
    .sort_values("transitions", ascending=False)
)


# ---------------------------------------------------------------------------
# 10.C — fixed-selector LOSO one-step evaluation with the exact formula
# ---------------------------------------------------------------------------
_todo10_conditional_angle = _todo9_baseline_angle.copy()
_todo10_conditional_position = _todo9_baseline_position.copy()
_todo10_formula_override = _todo9_residual_oof & ~_todo9_loop_prediction

if _todo10_formula_override.any():
    _todo10_conditional_angle[_todo10_formula_override] = (
        _todo10_special_angle(
            todo9_event_dataset.loc[
                _todo10_formula_override, "free_angle"
            ].to_numpy(dtype=float),
            _todo9_baseline_angle[_todo10_formula_override],
        )
    )
    _todo10_free_position = todo9_event_dataset.loc[
        _todo10_formula_override, ["free_x", "free_y"]
    ].to_numpy(dtype=float)
    _todo10_limits = np.vstack(
        [
            _todo8_oriented_limits(_angle)
            for _angle in _todo10_conditional_angle[
                _todo10_formula_override
            ]
        ]
    )
    _todo10_conditional_position[_todo10_formula_override] = np.clip(
        _todo10_free_position, -_todo10_limits, _todo10_limits
    )

_todo10_conditional_angle[_todo9_loop_prediction] = _todo9_angle_lag2[
    _todo9_loop_prediction
]
_todo10_conditional_position[_todo9_loop_prediction] = (
    _todo9_position_lag2[_todo9_loop_prediction]
)

_todo10_p1_prediction = _todo10_conditional_position[
    _todo9_p1_event_indices
].copy()
_todo10_angle1_prediction = _todo10_conditional_angle[
    _todo9_p1_event_indices
].copy()
_todo10_distance_prediction = np.linalg.norm(
    _todo10_p1_prediction - _todo9_previous_p2, axis=1
)

_todo10_p2_prediction = np.empty_like(_todo9_previous_p2)
_todo10_angle2_prediction = np.empty(
    len(_todo9_transition_indices), dtype=float
)

for _transition_row, (_data_index, _event_index, _driver_distance) in enumerate(
    zip(
        _todo9_transition_indices,
        _todo9_p2_event_indices,
        _todo10_distance_prediction,
    )
):
    _event = todo9_event_dataset.loc[_event_index]
    if bool(_event["predict_period2_from_previous"]):
        _lag2_index = max(int(_data_index) - 2, 0)
        _todo10_p2_prediction[_transition_row] = todo7_p2[_lag2_index]
        _todo10_angle2_prediction[_transition_row] = todo7_angle2[_lag2_index]
        continue

    _baseline = _todo8_update_object(
        todo7_p2[_data_index - 1],
        todo7_angle2[_data_index - 1],
        float(_driver_distance),
        -1.0,
    )
    _todo10_p2_prediction[_transition_row] = _baseline["position"]
    _todo10_angle2_prediction[_transition_row] = _baseline["angle"]

    if _todo10_p2_gate_by_transition[_transition_row]:
        _special_angle = _todo10_special_angle(
            _baseline["free_angle"], _baseline["angle"]
        )
        _limits = _todo8_oriented_limits(_special_angle)
        _todo10_p2_prediction[_transition_row] = np.clip(
            _baseline["free_position"], -_limits, _limits
        )
        _todo10_angle2_prediction[_transition_row] = _special_angle

_todo10_p1_error = np.linalg.norm(
    _todo10_p1_prediction - _todo9_truth_p1, axis=1
)
_todo10_p2_error = np.linalg.norm(
    _todo10_p2_prediction - _todo9_truth_p2, axis=1
)
_todo10_maximum_position_error = np.maximum(
    _todo10_p1_error, _todo10_p2_error
)
_todo10_angle1_error = np.abs(
    _todo7_wrap_degrees(
        _todo10_angle1_prediction - _todo9_truth_angle1
    )
)
_todo10_angle2_error = np.abs(
    _todo7_wrap_degrees(
        _todo10_angle2_prediction - _todo9_truth_angle2
    )
)
_todo10_maximum_angle_error = np.maximum(
    _todo10_angle1_error, _todo10_angle2_error
)
_todo10_distance_error = np.abs(
    _todo10_distance_prediction - _todo9_truth_distance
)
_todo10_sequential_exact = (
    (_todo10_maximum_position_error < TODO8_POSITION_TOLERANCE)
    & (_todo10_maximum_angle_error < TODO8_ANGLE_TOLERANCE)
    & (_todo10_distance_error < TODO8_DISTANCE_TOLERANCE)
)
_todo10_sequential_material = (
    (_todo10_maximum_position_error < TODO8_MATERIAL_POSITION_ERROR)
    & (_todo10_maximum_angle_error < TODO8_MATERIAL_ANGLE_ERROR)
    & (_todo10_distance_error < TODO8_MATERIAL_DISTANCE_ERROR)
)

todo10_one_step_summary = pd.Series(
    {
        "TODO8 exact complete-state share": float(
            _todo8_exact_complete.mean()
        ),
        "TODO9 approximate-angle exact share": float(
            _todo9_sequential_exact.mean()
        ),
        "TODO10 exact-formula exact share": float(
            _todo10_sequential_exact.mean()
        ),
        "TODO8 material complete-state share": float(
            _todo9_baseline_material.mean()
        ),
        "TODO9 material complete-state share": float(
            _todo9_sequential_material.mean()
        ),
        "TODO10 material complete-state share": float(
            _todo10_sequential_material.mean()
        ),
        "TODO9 maximum-angle MAE": float(
            _todo9_maximum_angle_error.mean()
        ),
        "TODO10 maximum-angle MAE": float(
            _todo10_maximum_angle_error.mean()
        ),
        "TODO10 material failures": int(
            (~_todo10_sequential_material).sum()
        ),
    },
    name="value",
)

_todo10_segment_rows = []
for _segment_id in sorted(np.unique(_todo9_transition_segments)):
    _mask = _todo9_transition_segments == _segment_id
    _todo10_segment_rows.append(
        {
            "segment_id": int(_segment_id),
            "transitions": int(_mask.sum()),
            "TODO8_exact": float(_todo8_exact_complete[_mask].mean()),
            "TODO9_exact": float(_todo9_sequential_exact[_mask].mean()),
            "TODO10_exact": float(_todo10_sequential_exact[_mask].mean()),
            "TODO8_material": float(_todo9_baseline_material[_mask].mean()),
            "TODO9_material": float(_todo9_sequential_material[_mask].mean()),
            "TODO10_material": float(_todo10_sequential_material[_mask].mean()),
        }
    )
todo10_one_step_per_segment = pd.DataFrame(
    _todo10_segment_rows
).set_index("segment_id")

_todo10_remaining_rows = set(
    np.flatnonzero(~_todo10_sequential_material).tolist()
)
_todo10_previous_failure_rows = set(
    todo10_material_failures["transition_row"].tolist()
)
assert _todo10_remaining_rows.issubset(_todo10_previous_failure_rows)
todo10_remaining_material_failures = todo10_material_failures.loc[
    todo10_material_failures["transition_row"].isin(
        _todo10_remaining_rows
    )
].copy()
todo10_remaining_material_failure_summary = (
    todo10_remaining_material_failures.groupby("cause", as_index=True)
    .agg(
        transitions=("source_row", "size"),
        segments=("segment_id", "nunique"),
        maximum_angle_error=("maximum_angle_error", "max"),
        maximum_position_error=("maximum_position_error", "max"),
    )
    .sort_values("transitions", ascending=False)
)

TODO10_EXACT_GATE_PASSED = bool(
    todo10_one_step_summary["TODO10 exact-formula exact share"]
    >= TODO10_EXACT_COMPLETE_STATE_GATE
)
TODO10_IMPROVED_SEGMENTS = int(
    (
        todo10_one_step_per_segment["TODO10_exact"]
        > todo10_one_step_per_segment["TODO8_exact"]
    ).sum()
)
TODO10_SEGMENT_GATE_PASSED = bool(
    TODO10_IMPROVED_SEGMENTS >= TODO10_MINIMUM_IMPROVED_SEGMENTS
)


# ---------------------------------------------------------------------------
# 10.D — recursive hybrid simulator, opened only by the exact one-step gate
# ---------------------------------------------------------------------------
def _todo10_wall_state(position, angle):
    _gap = _todo8_oriented_limits(angle) - np.abs(position)
    _x_wall = bool(abs(_gap[0]) < TODO8_POSITION_TOLERANCE)
    _y_wall = bool(abs(_gap[1]) < TODO8_POSITION_TOLERANCE)
    _mask = "xy" if _x_wall and _y_wall else "x" if _x_wall else "y" if _y_wall else "interior"
    return _gap, _mask, bool(_x_wall or _y_wall)


def _todo10_recursive_feature_record(
    object_name,
    segment_id,
    previous_position,
    previous_angle,
    driver_distance,
    update_result,
    history,
):
    _previous_gap, _wall_mask, _previous_wall = _todo10_wall_state(
        previous_position, previous_angle
    )
    _previous_period2_valid = len(history["position"]) >= 3
    if _previous_period2_valid:
        _period2_position_error = float(
            np.max(
                np.abs(history["position"][-1] - history["position"][-3])
            )
        )
        _period2_angle_error = float(
            abs(
                _todo7_wrap_degrees(
                    history["angle"][-1] - history["angle"][-3]
                )
            )
        )
    else:
        _period2_position_error = np.nan
        _period2_angle_error = np.nan
    _previous_period2_exact = bool(
        _previous_period2_valid
        and _period2_position_error < TODO8_POSITION_TOLERANCE
        and _period2_angle_error < TODO8_ANGLE_TOLERANCE
    )
    _free_x, _free_y = update_result["free_position"]
    return {
        "previous_x": float(previous_position[0]),
        "previous_y": float(previous_position[1]),
        "previous_angle": float(previous_angle),
        "driver_distance": float(driver_distance),
        "free_angle": float(update_result["free_angle"]),
        "step_length": float(update_result["step_length"]),
        "free_x": float(_free_x),
        "free_y": float(_free_y),
        "x_penetration": float(update_result["penetration"][0]),
        "y_penetration": float(update_result["penetration"][1]),
        "x_hit_fraction": float(update_result["hit_fraction"][0]),
        "y_hit_fraction": float(update_result["hit_fraction"][1]),
        "previous_x_gap": float(_previous_gap[0]),
        "previous_y_gap": float(_previous_gap[1]),
        "predicted_x_crossing": bool(update_result["crossing"][0]),
        "predicted_y_crossing": bool(update_result["crossing"][1]),
        "both_axis_proposal": bool(np.all(update_result["crossing"])),
        "previous_wall_contact": _previous_wall,
        "previous_boundary_run_length": int(history["boundary_run"][-1]),
        "previous_special": bool(history["special"][-1]),
        "previous_period2_position_error": _period2_position_error,
        "previous_period2_angle_error": _period2_angle_error,
        "previous_period2_exact": _previous_period2_exact,
        "object": object_name,
        "previous_actual_branch": str(history["branch"][-1]),
        "previous_event_mode": str(history["mode"][-1]),
        "previous_wall_mask": _wall_mask,
        "free_quadrant": (
            "upper_right"
            if _free_x >= 0 and _free_y >= 0
            else "upper_left"
            if _free_x < 0 and _free_y >= 0
            else "lower_left"
            if _free_x < 0 and _free_y < 0
            else "lower_right"
        ),
    }


_todo10_feature_index = {
    _name: _index for _index, _name in enumerate(TODO9_FEATURE_COLUMNS)
}


def _todo10_feature_vector(feature_record):
    """Reproduce TODO 9 get_dummies/reindex for one recursive event."""
    _vector = np.zeros(len(TODO9_FEATURE_COLUMNS), dtype=np.float32)
    for _name in TODO9_NUMERIC_FEATURES:
        _value = float(feature_record[_name])
        if not np.isfinite(_value):
            _value = -1.0
        _vector[_todo10_feature_index[_name]] = _value
    for _name in TODO9_CATEGORICAL_FEATURES:
        _dummy_name = f"{_name}_{feature_record[_name]}"
        if _dummy_name in _todo10_feature_index:
            _vector[_todo10_feature_index[_dummy_name]] = 1.0
    return _vector


def _todo10_fast_forest_prediction(model, feature_vector):
    """Exact sklearn forest vote without per-row pandas/joblib overhead."""
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
    return bool(model.classes_[int(np.argmax(_probability))])


def _todo10_gate_prediction(model, feature_record):
    return _todo10_fast_forest_prediction(
        model, _todo10_feature_vector(feature_record)
    )


# Guard against changing the trained selector while accelerating recursive
# single-row inference.  A fixed sample from every fold must match sklearn.
for _segment_id, _model in todo9_residual_models.items():
    _fold_rows = np.flatnonzero(_todo9_groups == _segment_id)[:8]
    for _row in _fold_rows:
        _vector = todo9_gate_X.iloc[_row].to_numpy(dtype=np.float32)
        assert _todo10_fast_forest_prediction(_model, _vector) == bool(
            _model.predict(todo9_gate_X.iloc[[_row]])[0]
        )


def _todo10_recursive_object_update(
    object_name,
    segment_id,
    previous_position,
    previous_angle,
    driver_distance,
    angle_sign,
    history,
    model,
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
    _period2 = bool(
        _features["previous_period2_exact"]
        and _features["previous_wall_contact"]
        and _features["previous_boundary_run_length"] >= 4
    )
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
            or _features["previous_special"]
            or _features["previous_period2_exact"]
        )
        _special = bool(
            _candidate and _todo10_gate_prediction(model, _features)
        )
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
    return _position, float(_angle), _mode


def _todo10_initial_history(position, angle):
    _, _, _wall = _todo10_wall_state(position, angle)
    return {
        "position": [np.asarray(position, dtype=float).copy()],
        "angle": [float(angle)],
        "branch": ["segment_start"],
        "mode": ["segment_start"],
        "special": [False],
        "boundary_run": [int(_wall)],
    }


def _todo10_rollout_segment(segment_frame, segment_id):
    _segment_frame = segment_frame.sort_index()
    _truth_p1 = _segment_frame[["x1", "y1"]].to_numpy(dtype=float)
    _truth_p2 = _segment_frame[["x2", "y2"]].to_numpy(dtype=float)
    _truth_a1 = _segment_frame["angle1"].to_numpy(dtype=float)
    _truth_a2 = _segment_frame["angle2"].to_numpy(dtype=float)
    _truth_distance = _segment_frame["distance"].to_numpy(dtype=float)
    _model = todo9_residual_models[int(segment_id)]
    _h1 = _todo10_initial_history(_truth_p1[0], _truth_a1[0])
    _h2 = _todo10_initial_history(_truth_p2[0], _truth_a2[0])
    _distance = [float(_truth_distance[0])]
    _mode1 = ["segment_start"]
    _mode2 = ["segment_start"]

    for _ in range(1, len(_segment_frame)):
        _rho_before = float(
            np.linalg.norm(_h2["position"][-1] - _h1["position"][-1])
        )
        _p1, _a1, _m1 = _todo10_recursive_object_update(
            "object1",
            int(segment_id),
            _h1["position"][-1],
            _h1["angle"][-1],
            _rho_before,
            1.0,
            _h1,
            _model,
        )
        _new_distance = float(np.linalg.norm(_p1 - _h2["position"][-1]))
        _, _, _m2 = _todo10_recursive_object_update(
            "object2",
            int(segment_id),
            _h2["position"][-1],
            _h2["angle"][-1],
            _new_distance,
            -1.0,
            _h2,
            _model,
        )
        _distance.append(_new_distance)
        _mode1.append(_m1)
        _mode2.append(_m2)

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
        "first_numerical_failure": _todo8_first_failure(_exact_failure),
        "first_material_failure": _todo8_first_failure(_material_failure),
    }


todo10_rollouts = {}
_todo10_rollout_rows = []
_todo10_horizon_rows = []

if TODO10_EXACT_GATE_PASSED:
    for _segment_id, _segment_frame in todo7_df.groupby(
        "segment_id", sort=True
    ):
        _rollout = _todo10_rollout_segment(_segment_frame, _segment_id)
        todo10_rollouts[int(_segment_id)] = _rollout
        _transition_count = len(_segment_frame) - 1
        _first_material = _rollout["first_material_failure"]
        _first_exact = _rollout["first_numerical_failure"]
        _todo10_rollout_rows.append(
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
                for _h in TODO10_ROLLOUT_HORIZONS
                if _h <= _transition_count
            ],
            ("full", _transition_count),
        ]:
            _truth = _rollout["truth_positions"][1 : _horizon + 1]
            _hybrid = _rollout["predicted_positions"][1 : _horizon + 1]
            _baseline_rollout = todo8_rollouts[int(_segment_id)]["trajectory"]
            _todo8_positions = _baseline_rollout[
                [
                    "predicted_x1",
                    "predicted_y1",
                    "predicted_x2",
                    "predicted_y2",
                ]
            ].to_numpy(dtype=float)[1 : _horizon + 1]
            _constant = np.broadcast_to(
                _rollout["truth_positions"][0], _truth.shape
            )
            _todo10_horizon_rows.append(
                {
                    "segment_id": int(_segment_id),
                    "horizon": _label,
                    "hybrid RMSE": float(
                        np.sqrt(np.mean((_hybrid - _truth) ** 2))
                    ),
                    "TODO8 RMSE": float(
                        np.sqrt(np.mean((_todo8_positions - _truth) ** 2))
                    ),
                    "constant RMSE": float(
                        np.sqrt(np.mean((_constant - _truth) ** 2))
                    ),
                }
            )

todo10_rollout_summary = pd.DataFrame(_todo10_rollout_rows)
if len(todo10_rollout_summary):
    todo10_rollout_summary = todo10_rollout_summary.set_index("segment_id")
todo10_rollout_horizons = pd.DataFrame(_todo10_horizon_rows)
todo10_rollout_macro = (
    todo10_rollout_horizons.groupby("horizon", sort=False)
    .agg(
        segments=("segment_id", "nunique"),
        hybrid_RMSE=("hybrid RMSE", "mean"),
        TODO8_RMSE=("TODO8 RMSE", "mean"),
        constant_RMSE=("constant RMSE", "mean"),
    )
    if len(todo10_rollout_horizons)
    else pd.DataFrame()
)

TODO10_RECURSIVE_ROLLOUT_RUN = bool(TODO10_EXACT_GATE_PASSED)
TODO10_MEDIAN_MATERIAL_PREFIX = (
    float(todo10_rollout_summary["first material failure"].median())
    if TODO10_RECURSIVE_ROLLOUT_RUN
    else np.nan
)
TODO10_RMSE_GATE_100_PASSED = bool(
    TODO10_RECURSIVE_ROLLOUT_RUN
    and todo10_rollout_macro.loc["100", "hybrid_RMSE"]
    < todo10_rollout_macro.loc["100", "TODO8_RMSE"]
)
TODO10_RMSE_GATE_500_PASSED = bool(
    TODO10_RECURSIVE_ROLLOUT_RUN
    and todo10_rollout_macro.loc["500", "hybrid_RMSE"]
    < todo10_rollout_macro.loc["500", "TODO8_RMSE"]
)
TODO10_FULL_BASELINE_GATE_PASSED = bool(
    TODO10_RECURSIVE_ROLLOUT_RUN
    and todo10_rollout_macro.loc["full", "hybrid_RMSE"]
    < todo10_rollout_macro.loc["full", "constant_RMSE"]
)
TODO10_ROLLOUT_GATE_PASSED = bool(
    TODO10_RECURSIVE_ROLLOUT_RUN
    and TODO10_MEDIAN_MATERIAL_PREFIX
    >= TODO10_ROLLOUT_MEDIAN_MATERIAL_GATE
    and TODO10_RMSE_GATE_100_PASSED
    and TODO10_RMSE_GATE_500_PASSED
    and TODO10_FULL_BASELINE_GATE_PASSED
)
TODO10_STATUS = (
    "exact_angle_recovered_all_preregistered_rollout_gates_passed"
    if TODO10_ROLLOUT_GATE_PASSED
    else "exact_angle_recovered_rollout_gate_failed"
    if TODO10_RECURSIVE_ROLLOUT_RUN
    else "exact_angle_recovered_one_step_gate_failed_rollout_closed"
)

todo10_manifest = pd.Series(
    {
        "status": TODO10_STATUS,
        "formula": "tan(theta) * cos(phi) = 2 / base_step = 0.2",
        "formula rows": int(_todo10_wall_special_mask.sum()),
        "formula uses future state": False,
        "LOSO selector changed": False,
        "exact one-step gate": TODO10_EXACT_COMPLETE_STATE_GATE,
        "exact one-step gate passed": TODO10_EXACT_GATE_PASSED,
        "segments improved vs TODO8": TODO10_IMPROVED_SEGMENTS,
        "minimum improved segments": TODO10_MINIMUM_IMPROVED_SEGMENTS,
        "segment gate passed": TODO10_SEGMENT_GATE_PASSED,
        "recursive rollout run": TODO10_RECURSIVE_ROLLOUT_RUN,
        "median first material failure": TODO10_MEDIAN_MATERIAL_PREFIX,
        "rollout median gate": TODO10_ROLLOUT_MEDIAN_MATERIAL_GATE,
        "RMSE gate at horizon 100 passed": TODO10_RMSE_GATE_100_PASSED,
        "RMSE gate at horizon 500 passed": TODO10_RMSE_GATE_500_PASSED,
        "full horizon beats constant baseline": (
            TODO10_FULL_BASELINE_GATE_PASSED
        ),
        "rollout median gate passed": TODO10_ROLLOUT_GATE_PASSED,
        "pristine test": False,
    },
    name="value",
)

assert todo10_angle_identity_summary[
    "numerically exact theta share"
] == 1.0
assert TODO10_RECURSIVE_ROLLOUT_RUN == TODO10_EXACT_GATE_PASSED
assert TODO10_EXACT_GATE_PASSED
assert TODO10_SEGMENT_GATE_PASSED
assert TODO10_ROLLOUT_GATE_PASSED
assert todo10_rollout_summary["finite"].all()

_todo10_figure, _todo10_axes = plt.subplots(1, 3, figsize=(18, 5.2))

_todo10_phi_plot = _todo10_axis_distance(
    todo9_event_dataset.loc[
        _todo10_wall_special_mask, "free_angle"
    ].to_numpy(dtype=float)
)
_todo10_axes[0].scatter(
    _todo10_phi_plot,
    _todo10_theta_actual[_todo10_wall_special_mask],
    s=5,
    alpha=0.25,
    color="tab:blue",
    label="observed special events",
)
_todo10_phi_grid = np.linspace(0.0, 45.0, 300)
_todo10_axes[0].plot(
    _todo10_phi_grid,
    _todo10_special_theta(_todo10_phi_grid),
    color="black",
    linewidth=2.0,
    label="exact formula",
)
_todo10_axes[0].set(
    xlabel="phi: free-angle distance to nearest axis, degrees",
    ylabel="theta, degrees",
    title="Recovered special-angle identity",
)
_todo10_axes[0].grid(alpha=0.2)
_todo10_axes[0].legend()

_todo10_segments = todo10_rollout_summary.index.to_numpy(dtype=int)
_todo10_x = np.arange(len(_todo10_segments))
_todo10_width = 0.38
_todo10_axes[1].bar(
    _todo10_x - _todo10_width / 2,
    todo8_rollout_summary.loc[
        _todo10_segments, "first material failure"
    ].to_numpy(dtype=float),
    width=_todo10_width,
    label="TODO8",
    color="slategray",
)
_todo10_axes[1].bar(
    _todo10_x + _todo10_width / 2,
    todo10_rollout_summary["first material failure"].to_numpy(dtype=float),
    width=_todo10_width,
    label="TODO10 hybrid",
    color="tab:green",
)
_todo10_axes[1].set_xticks(_todo10_x, _todo10_segments)
_todo10_axes[1].set(
    xlabel="segment_id",
    ylabel="first material failure, update",
    title="Recursive survival by segment",
)
_todo10_axes[1].grid(axis="y", alpha=0.2)
_todo10_axes[1].legend()

_todo10_horizon_labels = ["10", "50", "100", "500", "full"]
_todo10_horizon_x = np.arange(len(_todo10_horizon_labels))
for _column, _label, _color in [
    ("hybrid_RMSE", "TODO10 hybrid", "tab:green"),
    ("TODO8_RMSE", "TODO8", "slategray"),
    ("constant_RMSE", "constant position", "tab:orange"),
]:
    _todo10_axes[2].plot(
        _todo10_horizon_x,
        todo10_rollout_macro.loc[
            _todo10_horizon_labels, _column
        ].to_numpy(dtype=float),
        marker="o",
        label=_label,
        color=_color,
    )
_todo10_axes[2].set_xticks(
    _todo10_horizon_x, _todo10_horizon_labels
)
_todo10_axes[2].set(
    xlabel="forecast horizon, updates",
    ylabel="macro coordinate RMSE",
    title="Recursive forecast error",
)
_todo10_axes[2].grid(alpha=0.2)
_todo10_axes[2].legend()

_todo10_figure.tight_layout()

display(
    Markdown("#### Exact special-angle identity"),
    todo10_angle_identity_summary.to_frame(),
    Markdown("#### Причины 69 material failures в TODO 9"),
    todo10_material_failure_summary.style.format(precision=6),
    Markdown("#### Sequential LOSO one-step"),
    todo10_one_step_summary.to_frame(),
    todo10_one_step_per_segment.style.format(precision=6),
    Markdown("#### Оставшиеся 58 material failures после точной формулы"),
    todo10_remaining_material_failure_summary.style.format(precision=6),
    Markdown("#### Полностью recursive rollout"),
    todo10_rollout_summary.style.format(precision=6)
    if len(todo10_rollout_summary)
    else todo10_rollout_summary,
    todo10_rollout_macro.style.format(precision=6)
    if len(todo10_rollout_macro)
    else todo10_rollout_macro,
    Markdown("#### TODO 10 manifest"),
    todo10_manifest.to_frame(),
    _todo10_figure,
)
plt.close(_todo10_figure)
