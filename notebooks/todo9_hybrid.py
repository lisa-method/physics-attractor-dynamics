"""TODO 9 companion: causal boundary event dataset and hybrid one-step probe.

This file is executed from ``attractor_todo.ipynb`` after TODO 8.  It relies on
the named TODO7/TODO8 objects created there and intentionally leaves recursive
rollout closed when the preregistered numerical-exact gate is not met.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.tree import DecisionTreeClassifier


TODO9_RANDOM_STATE = 42
TODO9_TREE_MAX_DEPTH = 4
TODO9_TREE_MIN_SAMPLES_LEAF = 20
TODO9_RF_ESTIMATORS = 160
TODO9_RF_MAX_DEPTH = 12
TODO9_RF_MIN_SAMPLES_LEAF = 5
TODO9_SPECIAL_F1_GATE = 0.80
TODO9_EXACT_COMPLETE_STATE_GATE = 0.9817
TODO9_SPECIAL_MODES = (
    "period2_wall_loop",
    "wall_loop_special",
    "interior_special",
    "segment_start_special",
)


# ---------------------------------------------------------------------------
# 9.A — causal event dataset
# ---------------------------------------------------------------------------
todo9_event_dataset = todo8_event_table.copy().reset_index(drop=True)
todo9_event_dataset["target_special"] = todo9_event_dataset[
    "event_mode"
].isin(TODO9_SPECIAL_MODES)

_todo9_group_keys = ["object", "segment_id"]
todo9_event_dataset["previous_actual_branch"] = (
    todo9_event_dataset.groupby(_todo9_group_keys, sort=False)[
        "actual_branch"
    ]
    .shift(1)
    .fillna("segment_start")
)
todo9_event_dataset["previous_event_mode"] = (
    todo9_event_dataset.groupby(_todo9_group_keys, sort=False)[
        "event_mode"
    ]
    .shift(1)
    .fillna("segment_start")
)
todo9_event_dataset["previous_special"] = todo9_event_dataset[
    "previous_event_mode"
].isin(TODO9_SPECIAL_MODES)

_todo9_previous_wall_run = np.zeros(len(todo9_event_dataset), dtype=int)
for _, _indices in todo9_event_dataset.groupby(
    _todo9_group_keys, sort=False
).indices.items():
    _run_length = 0
    for _row_index in np.sort(_indices):
        if bool(
            todo9_event_dataset.at[
                _row_index, "previous_wall_contact"
            ]
        ):
            _run_length += 1
        else:
            _run_length = 0
        _todo9_previous_wall_run[_row_index] = _run_length
todo9_event_dataset["previous_boundary_run_length"] = (
    _todo9_previous_wall_run
)

_todo9_locations = _todo8_source_to_position.loc[
    todo9_event_dataset["source_row"]
].to_numpy(dtype=int)
_todo9_is_object1 = todo9_event_dataset["object"].eq(
    "object1"
).to_numpy()
_todo9_actual_position = np.where(
    _todo9_is_object1[:, None],
    todo7_p1[_todo9_locations],
    todo7_p2[_todo9_locations],
)
_todo9_actual_angle = np.where(
    _todo9_is_object1,
    todo7_angle1[_todo9_locations],
    todo7_angle2[_todo9_locations],
)


def _todo9_object_state_at(offset):
    _lag_locations = np.maximum(_todo9_locations - offset, 0)
    _position = np.where(
        _todo9_is_object1[:, None],
        todo7_p1[_lag_locations],
        todo7_p2[_lag_locations],
    )
    _angle = np.where(
        _todo9_is_object1,
        todo7_angle1[_lag_locations],
        todo7_angle2[_lag_locations],
    )
    _valid = (
        (_todo9_locations >= offset)
        & (
            todo7_segment[_todo9_locations]
            == todo7_segment[_lag_locations]
        )
    )
    return _position, _angle, _valid


(
    _todo9_position_lag1,
    _todo9_angle_lag1,
    _todo9_lag1_valid,
) = _todo9_object_state_at(1)
(
    _todo9_position_lag2,
    _todo9_angle_lag2,
    _todo9_lag2_valid,
) = _todo9_object_state_at(2)
(
    _todo9_position_lag3,
    _todo9_angle_lag3,
    _todo9_lag3_valid,
) = _todo9_object_state_at(3)

_todo9_previous_period2_position_error = np.max(
    np.abs(_todo9_position_lag1 - _todo9_position_lag3),
    axis=1,
)
_todo9_previous_period2_angle_error = np.abs(
    _todo7_wrap_degrees(
        _todo9_angle_lag1 - _todo9_angle_lag3
    )
)
_todo9_previous_period2_valid = (
    _todo9_lag1_valid & _todo9_lag3_valid
)
_todo9_previous_period2_position_error[
    ~_todo9_previous_period2_valid
] = np.nan
_todo9_previous_period2_angle_error[
    ~_todo9_previous_period2_valid
] = np.nan
todo9_event_dataset["previous_period2_position_error"] = (
    _todo9_previous_period2_position_error
)
todo9_event_dataset["previous_period2_angle_error"] = (
    _todo9_previous_period2_angle_error
)
todo9_event_dataset["previous_period2_exact"] = (
    _todo9_previous_period2_valid
    & (
        _todo9_previous_period2_position_error
        < TODO8_POSITION_TOLERANCE
    )
    & (
        _todo9_previous_period2_angle_error
        < TODO8_ANGLE_TOLERANCE
    )
)

todo9_event_dataset["previous_wall_mask"] = np.select(
    [
        todo9_event_dataset["previous_x_gap"].abs().lt(1e-5)
        & todo9_event_dataset["previous_y_gap"].abs().lt(1e-5),
        todo9_event_dataset["previous_x_gap"].abs().lt(1e-5),
        todo9_event_dataset["previous_y_gap"].abs().lt(1e-5),
    ],
    ["xy", "x", "y"],
    default="interior",
)
todo9_event_dataset["free_quadrant"] = np.select(
    [
        (todo9_event_dataset["free_x"] >= 0)
        & (todo9_event_dataset["free_y"] >= 0),
        (todo9_event_dataset["free_x"] < 0)
        & (todo9_event_dataset["free_y"] >= 0),
        (todo9_event_dataset["free_x"] < 0)
        & (todo9_event_dataset["free_y"] < 0),
    ],
    ["upper_right", "upper_left", "lower_left"],
    default="lower_right",
)
todo9_event_dataset["causal_gate_candidate"] = (
    todo9_event_dataset["predicted_x_crossing"]
    | todo9_event_dataset["predicted_y_crossing"]
    | todo9_event_dataset["previous_wall_contact"]
    | todo9_event_dataset["previous_special"]
    | todo9_event_dataset["previous_period2_exact"]
)

TODO9_NUMERIC_FEATURES = (
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
    "previous_special",
    "previous_period2_position_error",
    "previous_period2_angle_error",
    "previous_period2_exact",
)
TODO9_CATEGORICAL_FEATURES = (
    "object",
    "previous_actual_branch",
    "previous_event_mode",
    "previous_wall_mask",
    "free_quadrant",
)

_todo9_forbidden_feature_names = {
    "actual_branch",
    "event_mode",
    "current_wall_contact",
    "current_x_wall",
    "current_y_wall",
    "lag2_loop_candidate",
    "lag2_position_linf",
    "lag2_angle_error",
    "conditional_position_error",
    "conditional_angle_error",
    "conditional_state_exact",
}
assert not _todo9_forbidden_feature_names.intersection(
    TODO9_NUMERIC_FEATURES + TODO9_CATEGORICAL_FEATURES
)

_todo9_candidate_mask = todo9_event_dataset[
    "causal_gate_candidate"
].to_numpy(dtype=bool)
todo9_gate_candidates = todo9_event_dataset.loc[
    _todo9_candidate_mask
].copy()
todo9_gate_X = pd.get_dummies(
    todo9_gate_candidates[
        list(TODO9_NUMERIC_FEATURES + TODO9_CATEGORICAL_FEATURES)
    ],
    columns=list(TODO9_CATEGORICAL_FEATURES),
    dtype=float,
)
todo9_gate_X = (
    todo9_gate_X.replace([np.inf, -np.inf], np.nan)
    .fillna(-1.0)
)
TODO9_FEATURE_COLUMNS = tuple(todo9_gate_X.columns)
_todo9_groups = todo9_gate_candidates["segment_id"].to_numpy(
    dtype=int
)


# ---------------------------------------------------------------------------
# 9.B — period-2 state machine and quantized special-angle law
# ---------------------------------------------------------------------------
# TODO8 labeled only the nonstandard half of the alternating loop.  The true
# state-machine target includes both phases whenever the complete state repeats
# state i-2 while the object remains on a wall.
todo9_event_dataset["target_period2_state"] = todo9_event_dataset[
    "lag2_loop_candidate"
].astype(bool)
todo9_event_dataset["predict_period2_from_previous"] = (
    todo9_event_dataset["previous_period2_exact"]
    & todo9_event_dataset["previous_wall_contact"]
    & todo9_event_dataset["previous_boundary_run_length"].ge(4)
)

_todo9_loop_target = todo9_event_dataset[
    "target_period2_state"
].to_numpy(dtype=bool)
_todo9_loop_prediction = todo9_event_dataset[
    "predict_period2_from_previous"
].to_numpy(dtype=bool)
(
    _todo9_loop_precision,
    _todo9_loop_recall,
    _todo9_loop_f1,
    _,
) = precision_recall_fscore_support(
    _todo9_loop_target,
    _todo9_loop_prediction,
    labels=[True],
    zero_division=0,
)
todo9_period2_rule_summary = pd.Series(
    {
        "target loop-state rows": int(_todo9_loop_target.sum()),
        "target special-looking phase rows": int(
            (
                _todo9_loop_target
                & todo9_event_dataset["target_special"].to_numpy()
            ).sum()
        ),
        "target standard-looking phase rows": int(
            (
                _todo9_loop_target
                & ~todo9_event_dataset["target_special"].to_numpy()
            ).sum()
        ),
        "predicted rows": int(_todo9_loop_prediction.sum()),
        "precision": float(_todo9_loop_precision[0]),
        "recall": float(_todo9_loop_recall[0]),
        "F1": float(_todo9_loop_f1[0]),
    },
    name="value",
)

_todo9_actual_axis_residual = (
    (_todo9_actual_angle + 45.0) % 90.0
) - 45.0
_todo9_special_mask = todo9_event_dataset[
    "target_special"
].to_numpy(dtype=bool)
todo9_special_theta_summary = pd.Series(
    {
        "mean": float(
            np.mean(np.abs(_todo9_actual_axis_residual[_todo9_special_mask]))
        ),
        "standard deviation": float(
            np.std(np.abs(_todo9_actual_axis_residual[_todo9_special_mask]))
        ),
        "median": float(
            np.median(np.abs(_todo9_actual_axis_residual[_todo9_special_mask]))
        ),
        "q05": float(
            np.quantile(
                np.abs(_todo9_actual_axis_residual[_todo9_special_mask]),
                0.05,
            )
        ),
        "q95": float(
            np.quantile(
                np.abs(_todo9_actual_axis_residual[_todo9_special_mask]),
                0.95,
            )
        ),
    },
    name="degrees",
)


def _todo9_quantize_special_angle(reference_angle, theta):
    _reference = np.asarray(reference_angle, dtype=float)
    _grid = np.asarray(
        [
            _todo7_wrap_degrees(90.0 * _k + _sign * theta)
            for _k in range(-2, 3)
            for _sign in (-1.0, 1.0)
        ],
        dtype=float,
    )
    _errors = np.abs(
        _todo7_wrap_degrees(
            _reference[:, None] - _grid[None, :]
        )
    )
    return _grid[np.argmin(_errors, axis=1)]


_todo9_baseline_angle = np.select(
    [
        todo9_event_dataset["predicted_branch"].eq(
            "vertical reflection"
        ),
        todo9_event_dataset["predicted_branch"].eq(
            "horizontal reflection"
        ),
    ],
    [
        _todo7_wrap_degrees(
            -todo9_event_dataset["free_angle"].to_numpy()
        ),
        _todo7_wrap_degrees(
            180.0
            - todo9_event_dataset["free_angle"].to_numpy()
        ),
    ],
    default=todo9_event_dataset["free_angle"].to_numpy(),
)


# ---------------------------------------------------------------------------
# 9.C — leave-one-segment-out rule probe and residual ML gate
# ---------------------------------------------------------------------------
_todo9_special_y = todo9_gate_candidates[
    "target_special"
].to_numpy(dtype=bool)
_todo9_tree_oof = np.zeros(len(todo9_gate_candidates), dtype=bool)
for _held_out_segment in sorted(np.unique(_todo9_groups)):
    _train_mask = _todo9_groups != _held_out_segment
    _valid_mask = _todo9_groups == _held_out_segment
    _tree = DecisionTreeClassifier(
        max_depth=TODO9_TREE_MAX_DEPTH,
        min_samples_leaf=TODO9_TREE_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=TODO9_RANDOM_STATE,
    )
    _tree.fit(todo9_gate_X.loc[_train_mask], _todo9_special_y[_train_mask])
    _todo9_tree_oof[_valid_mask] = _tree.predict(
        todo9_gate_X.loc[_valid_mask]
    )
(
    _todo9_tree_precision,
    _todo9_tree_recall,
    _todo9_tree_f1,
    _,
) = precision_recall_fscore_support(
    _todo9_special_y,
    _todo9_tree_oof,
    labels=[True],
    zero_division=0,
)
todo9_tree_probe_summary = pd.Series(
    {
        "special precision": float(_todo9_tree_precision[0]),
        "special recall": float(_todo9_tree_recall[0]),
        "special F1": float(_todo9_tree_f1[0]),
        "candidate accuracy": float(
            accuracy_score(_todo9_special_y, _todo9_tree_oof)
        ),
    },
    name="value",
)

# Period-2 continuation is handled by the explicit state machine.  RF predicts
# only the remaining special-angle override.
_todo9_residual_target_all = (
    todo9_event_dataset["target_special"]
    & ~todo9_event_dataset["target_period2_state"]
).to_numpy(dtype=bool)
_todo9_residual_y = _todo9_residual_target_all[_todo9_candidate_mask]
_todo9_residual_oof_candidate = np.zeros(
    len(todo9_gate_candidates), dtype=bool
)
todo9_residual_models = {}
todo9_theta_by_segment = {}
_todo9_rf_fold_rows = []

for _held_out_segment in sorted(np.unique(_todo9_groups)):
    _train_mask = _todo9_groups != _held_out_segment
    _valid_mask = _todo9_groups == _held_out_segment
    _rf = RandomForestClassifier(
        n_estimators=TODO9_RF_ESTIMATORS,
        max_depth=TODO9_RF_MAX_DEPTH,
        min_samples_leaf=TODO9_RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=TODO9_RANDOM_STATE,
        n_jobs=-1,
    )
    _rf.fit(
        todo9_gate_X.loc[_train_mask],
        _todo9_residual_y[_train_mask],
    )
    todo9_residual_models[int(_held_out_segment)] = _rf
    _fold_prediction = _rf.predict(todo9_gate_X.loc[_valid_mask])
    _todo9_residual_oof_candidate[_valid_mask] = _fold_prediction
    (
        _fold_precision,
        _fold_recall,
        _fold_f1,
        _,
    ) = precision_recall_fscore_support(
        _todo9_residual_y[_valid_mask],
        _fold_prediction,
        labels=[True],
        zero_division=0,
    )
    _todo9_rf_fold_rows.append(
        {
            "segment_id": int(_held_out_segment),
            "special_precision": float(_fold_precision[0]),
            "special_recall": float(_fold_recall[0]),
            "special_f1": float(_fold_f1[0]),
        }
    )
    _theta_train_mask = (
        todo9_event_dataset["segment_id"].ne(
            _held_out_segment
        ).to_numpy()
        & _todo9_residual_target_all
    )
    todo9_theta_by_segment[int(_held_out_segment)] = float(
        np.median(
            np.abs(_todo9_actual_axis_residual[_theta_train_mask])
        )
    )

_todo9_residual_oof = np.zeros(len(todo9_event_dataset), dtype=bool)
_todo9_residual_oof[_todo9_candidate_mask] = (
    _todo9_residual_oof_candidate
)
(
    _todo9_rf_precision,
    _todo9_rf_recall,
    _todo9_rf_f1,
    _,
) = precision_recall_fscore_support(
    _todo9_residual_target_all,
    _todo9_residual_oof,
    labels=[True],
    zero_division=0,
)
todo9_residual_gate_summary = pd.Series(
    {
        "target residual-special rows": int(
            _todo9_residual_target_all.sum()
        ),
        "predicted residual-special rows": int(
            _todo9_residual_oof.sum()
        ),
        "precision": float(_todo9_rf_precision[0]),
        "recall": float(_todo9_rf_recall[0]),
        "F1": float(_todo9_rf_f1[0]),
    },
    name="value",
)
todo9_residual_gate_per_segment = pd.DataFrame(
    _todo9_rf_fold_rows
).set_index("segment_id")


# ---------------------------------------------------------------------------
# 9.D — conditional and full sequential OOF one-step evaluation
# ---------------------------------------------------------------------------
_todo9_baseline_position = np.empty_like(_todo9_actual_position)
for _row_index in range(len(todo9_event_dataset)):
    _limits = _todo8_oriented_limits(_todo9_baseline_angle[_row_index])
    _todo9_baseline_position[_row_index] = np.clip(
        todo9_event_dataset.loc[
            _row_index, ["free_x", "free_y"]
        ].to_numpy(dtype=float),
        -_limits,
        _limits,
    )

_todo9_hybrid_angle = _todo9_baseline_angle.copy()
_todo9_hybrid_position = _todo9_baseline_position.copy()

for _held_out_segment in sorted(
    todo9_event_dataset["segment_id"].unique()
):
    _segment_mask = todo9_event_dataset["segment_id"].eq(
        _held_out_segment
    ).to_numpy()
    _quantized_mask = (
        _segment_mask
        & _todo9_residual_oof
        & ~_todo9_loop_prediction
    )
    if _quantized_mask.any():
        _todo9_hybrid_angle[_quantized_mask] = (
            _todo9_quantize_special_angle(
                _todo9_baseline_angle[_quantized_mask],
                todo9_theta_by_segment[int(_held_out_segment)],
            )
        )
        _free_position = todo9_event_dataset.loc[
            _quantized_mask, ["free_x", "free_y"]
        ].to_numpy(dtype=float)
        _limits = np.vstack(
            [
                _todo8_oriented_limits(_angle)
                for _angle in _todo9_hybrid_angle[_quantized_mask]
            ]
        )
        _todo9_hybrid_position[_quantized_mask] = np.clip(
            _free_position, -_limits, _limits
        )

_todo9_hybrid_angle[_todo9_loop_prediction] = _todo9_angle_lag2[
    _todo9_loop_prediction
]
_todo9_hybrid_position[_todo9_loop_prediction] = (
    _todo9_position_lag2[_todo9_loop_prediction]
)

_todo9_baseline_conditional_angle_error = np.abs(
    _todo7_wrap_degrees(
        _todo9_baseline_angle - _todo9_actual_angle
    )
)
_todo9_hybrid_conditional_angle_error = np.abs(
    _todo7_wrap_degrees(
        _todo9_hybrid_angle - _todo9_actual_angle
    )
)
_todo9_baseline_conditional_position_error = np.linalg.norm(
    _todo9_baseline_position - _todo9_actual_position,
    axis=1,
)
_todo9_hybrid_conditional_position_error = np.linalg.norm(
    _todo9_hybrid_position - _todo9_actual_position,
    axis=1,
)
_todo9_baseline_conditional_material = (
    (
        _todo9_baseline_conditional_angle_error
        < TODO8_MATERIAL_ANGLE_ERROR
    )
    & (
        _todo9_baseline_conditional_position_error
        < TODO8_MATERIAL_POSITION_ERROR
    )
)
_todo9_hybrid_conditional_material = (
    (
        _todo9_hybrid_conditional_angle_error
        < TODO8_MATERIAL_ANGLE_ERROR
    )
    & (
        _todo9_hybrid_conditional_position_error
        < TODO8_MATERIAL_POSITION_ERROR
    )
)
todo9_conditional_summary = pd.Series(
    {
        "baseline material state share": float(
            _todo9_baseline_conditional_material.mean()
        ),
        "hybrid material state share": float(
            _todo9_hybrid_conditional_material.mean()
        ),
        "baseline angle MAE": float(
            _todo9_baseline_conditional_angle_error.mean()
        ),
        "hybrid angle MAE": float(
            _todo9_hybrid_conditional_angle_error.mean()
        ),
        "baseline position RMSE": float(
            np.sqrt(
                np.mean(
                    _todo9_baseline_conditional_position_error ** 2
                )
            )
        ),
        "hybrid position RMSE": float(
            np.sqrt(
                np.mean(
                    _todo9_hybrid_conditional_position_error ** 2
                )
            )
        ),
    },
    name="value",
)


def _todo9_feature_record(event_index, update_result, driver_distance):
    _event = todo9_event_dataset.loc[event_index]
    _free_x, _free_y = update_result["free_position"]
    return {
        "event_index": int(event_index),
        "segment_id": int(_event["segment_id"]),
        "previous_x": float(_event["previous_x"]),
        "previous_y": float(_event["previous_y"]),
        "previous_angle": float(_event["previous_angle"]),
        "driver_distance": float(driver_distance),
        "free_angle": float(update_result["free_angle"]),
        "step_length": float(update_result["step_length"]),
        "free_x": float(_free_x),
        "free_y": float(_free_y),
        "x_penetration": float(update_result["penetration"][0]),
        "y_penetration": float(update_result["penetration"][1]),
        "x_hit_fraction": float(update_result["hit_fraction"][0]),
        "y_hit_fraction": float(update_result["hit_fraction"][1]),
        "previous_x_gap": float(_event["previous_x_gap"]),
        "previous_y_gap": float(_event["previous_y_gap"]),
        "predicted_x_crossing": bool(update_result["crossing"][0]),
        "predicted_y_crossing": bool(update_result["crossing"][1]),
        "both_axis_proposal": bool(np.all(update_result["crossing"])),
        "previous_wall_contact": bool(
            _event["previous_wall_contact"]
        ),
        "previous_boundary_run_length": int(
            _event["previous_boundary_run_length"]
        ),
        "previous_special": bool(_event["previous_special"]),
        "previous_period2_position_error": float(
            _event["previous_period2_position_error"]
        ),
        "previous_period2_angle_error": float(
            _event["previous_period2_angle_error"]
        ),
        "previous_period2_exact": bool(
            _event["previous_period2_exact"]
        ),
        "object": str(_event["object"]),
        "previous_actual_branch": str(
            _event["previous_actual_branch"]
        ),
        "previous_event_mode": str(_event["previous_event_mode"]),
        "previous_wall_mask": str(_event["previous_wall_mask"]),
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


_todo9_event_index_by_key = {
    (
        int(_row.segment_id),
        int(_row.source_row),
        str(_row.object),
    ): int(_index)
    for _index, _row in todo9_event_dataset.iterrows()
}
_todo9_transition_indices = np.flatnonzero(todo7_same_segment) + 1
_todo9_transition_segments = todo7_segment[_todo9_transition_indices]
_todo9_transition_source_rows = todo7_df.index.to_numpy(dtype=int)[
    _todo9_transition_indices
]
_todo9_p1_event_indices = np.array(
    [
        _todo9_event_index_by_key[
            (int(_segment), int(_source), "object1")
        ]
        for _segment, _source in zip(
            _todo9_transition_segments,
            _todo9_transition_source_rows,
        )
    ],
    dtype=int,
)
_todo9_p2_event_indices = np.array(
    [
        _todo9_event_index_by_key[
            (int(_segment), int(_source), "object2")
        ]
        for _segment, _source in zip(
            _todo9_transition_segments,
            _todo9_transition_source_rows,
        )
    ],
    dtype=int,
)

# Object 1 has no upstream prediction inside the same row, so its conditional
# OOF prediction is already the correct sequential one-step prediction.
_todo9_p1_prediction = _todo9_hybrid_position[
    _todo9_p1_event_indices
].copy()
_todo9_angle1_prediction = _todo9_hybrid_angle[
    _todo9_p1_event_indices
].copy()
_todo9_previous_p2 = todo7_p2[_todo9_transition_indices - 1]
_todo9_distance_prediction = np.linalg.norm(
    _todo9_p1_prediction - _todo9_previous_p2,
    axis=1,
)

_todo9_p2_prediction = np.empty_like(_todo9_previous_p2)
_todo9_angle2_prediction = np.empty(
    len(_todo9_transition_indices), dtype=float
)
_todo9_p2_base_updates = {}
_todo9_p2_candidate_records = []

for _transition_row, (
    _data_index,
    _event_index,
    _driver_distance,
) in enumerate(
    zip(
        _todo9_transition_indices,
        _todo9_p2_event_indices,
        _todo9_distance_prediction,
    )
):
    _event = todo9_event_dataset.loc[_event_index]
    if bool(_event["predict_period2_from_previous"]):
        _lag2_index = max(int(_data_index) - 2, 0)
        _todo9_p2_prediction[_transition_row] = todo7_p2[
            _lag2_index
        ]
        _todo9_angle2_prediction[_transition_row] = todo7_angle2[
            _lag2_index
        ]
        continue

    _baseline = _todo8_update_object(
        todo7_p2[_data_index - 1],
        todo7_angle2[_data_index - 1],
        float(_driver_distance),
        -1.0,
    )
    _todo9_p2_prediction[_transition_row] = _baseline["position"]
    _todo9_angle2_prediction[_transition_row] = _baseline["angle"]
    _todo9_p2_base_updates[_transition_row] = _baseline
    _causal_candidate = bool(
        np.any(_baseline["crossing"])
        or _event["previous_wall_contact"]
        or _event["previous_special"]
        or _event["previous_period2_exact"]
    )
    if _causal_candidate:
        _record = _todo9_feature_record(
            _event_index,
            _baseline,
            _driver_distance,
        )
        _record["transition_row"] = int(_transition_row)
        _todo9_p2_candidate_records.append(_record)

todo9_p2_candidates = pd.DataFrame(_todo9_p2_candidate_records)
_todo9_p2_gate_prediction = np.zeros(
    len(todo9_p2_candidates), dtype=bool
)
if len(todo9_p2_candidates):
    _todo9_p2_feature_frame = pd.get_dummies(
        todo9_p2_candidates[
            list(TODO9_NUMERIC_FEATURES + TODO9_CATEGORICAL_FEATURES)
        ],
        columns=list(TODO9_CATEGORICAL_FEATURES),
        dtype=float,
    )
    _todo9_p2_feature_frame = (
        _todo9_p2_feature_frame.reindex(
            columns=TODO9_FEATURE_COLUMNS, fill_value=0.0
        )
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-1.0)
    )
    for _segment_id, _candidate_indices in todo9_p2_candidates.groupby(
        "segment_id", sort=True
    ).indices.items():
        _candidate_indices = np.asarray(_candidate_indices, dtype=int)
        _todo9_p2_gate_prediction[_candidate_indices] = (
            todo9_residual_models[int(_segment_id)].predict(
                _todo9_p2_feature_frame.iloc[_candidate_indices]
            )
        )

for _candidate_row in np.flatnonzero(_todo9_p2_gate_prediction):
    _transition_row = int(
        todo9_p2_candidates.iloc[_candidate_row]["transition_row"]
    )
    _segment_id = int(
        todo9_p2_candidates.iloc[_candidate_row]["segment_id"]
    )
    _baseline = _todo9_p2_base_updates[_transition_row]
    _special_angle = float(
        _todo9_quantize_special_angle(
            np.array([_baseline["angle"]]),
            todo9_theta_by_segment[_segment_id],
        )[0]
    )
    _limits = _todo8_oriented_limits(_special_angle)
    _todo9_p2_prediction[_transition_row] = np.clip(
        _baseline["free_position"], -_limits, _limits
    )
    _todo9_angle2_prediction[_transition_row] = _special_angle

_todo9_truth_p1 = todo7_p1[_todo9_transition_indices]
_todo9_truth_p2 = todo7_p2[_todo9_transition_indices]
_todo9_truth_angle1 = todo7_angle1[_todo9_transition_indices]
_todo9_truth_angle2 = todo7_angle2[_todo9_transition_indices]
_todo9_truth_distance = todo7_distance[_todo9_transition_indices]

_todo9_p1_error = np.linalg.norm(
    _todo9_p1_prediction - _todo9_truth_p1, axis=1
)
_todo9_p2_error = np.linalg.norm(
    _todo9_p2_prediction - _todo9_truth_p2, axis=1
)
_todo9_maximum_position_error = np.maximum(
    _todo9_p1_error, _todo9_p2_error
)
_todo9_angle1_error = np.abs(
    _todo7_wrap_degrees(
        _todo9_angle1_prediction - _todo9_truth_angle1
    )
)
_todo9_angle2_error = np.abs(
    _todo7_wrap_degrees(
        _todo9_angle2_prediction - _todo9_truth_angle2
    )
)
_todo9_maximum_angle_error = np.maximum(
    _todo9_angle1_error, _todo9_angle2_error
)
_todo9_distance_error = np.abs(
    _todo9_distance_prediction - _todo9_truth_distance
)

_todo9_sequential_exact = (
    (_todo9_maximum_position_error < TODO8_POSITION_TOLERANCE)
    & (_todo9_maximum_angle_error < TODO8_ANGLE_TOLERANCE)
    & (_todo9_distance_error < TODO8_DISTANCE_TOLERANCE)
)
_todo9_sequential_material = (
    (_todo9_maximum_position_error < TODO8_MATERIAL_POSITION_ERROR)
    & (_todo9_maximum_angle_error < TODO8_MATERIAL_ANGLE_ERROR)
    & (_todo9_distance_error < TODO8_MATERIAL_DISTANCE_ERROR)
)
_todo9_baseline_material = (
    todo8_transition_audit["maximum_position_error"].lt(
        TODO8_MATERIAL_POSITION_ERROR
    )
    & todo8_transition_audit["maximum_angle_error"].lt(
        TODO8_MATERIAL_ANGLE_ERROR
    )
    & todo8_transition_audit["distance_error"].lt(
        TODO8_MATERIAL_DISTANCE_ERROR
    )
).to_numpy()

todo9_sequential_summary = pd.Series(
    {
        "baseline exact complete-state share": float(
            _todo8_exact_complete.mean()
        ),
        "hybrid exact complete-state share": float(
            _todo9_sequential_exact.mean()
        ),
        "baseline material complete-state share": float(
            _todo9_baseline_material.mean()
        ),
        "hybrid material complete-state share": float(
            _todo9_sequential_material.mean()
        ),
        "baseline maximum-angle MAE": float(
            todo8_transition_audit["maximum_angle_error"].mean()
        ),
        "hybrid maximum-angle MAE": float(
            _todo9_maximum_angle_error.mean()
        ),
        "baseline maximum-position RMSE": float(
            np.sqrt(
                np.mean(
                    todo8_transition_audit[
                        "maximum_position_error"
                    ].to_numpy() ** 2
                )
            )
        ),
        "hybrid maximum-position RMSE": float(
            np.sqrt(np.mean(_todo9_maximum_position_error ** 2))
        ),
    },
    name="value",
)

_todo9_segment_rows = []
for _segment_id in sorted(np.unique(_todo9_transition_segments)):
    _segment_mask = _todo9_transition_segments == _segment_id
    _baseline_mask = todo8_transition_audit["segment_id"].eq(
        _segment_id
    ).to_numpy()
    _todo9_segment_rows.append(
        {
            "segment_id": int(_segment_id),
            "transitions": int(_segment_mask.sum()),
            "baseline_exact": float(
                _todo8_exact_complete[_baseline_mask].mean()
            ),
            "hybrid_exact": float(
                _todo9_sequential_exact[_segment_mask].mean()
            ),
            "baseline_material": float(
                _todo9_baseline_material[_baseline_mask].mean()
            ),
            "hybrid_material": float(
                _todo9_sequential_material[_segment_mask].mean()
            ),
            "baseline_angle_MAE": float(
                todo8_transition_audit.loc[
                    _baseline_mask, "maximum_angle_error"
                ].mean()
            ),
            "hybrid_angle_MAE": float(
                _todo9_maximum_angle_error[_segment_mask].mean()
            ),
        }
    )
todo9_sequential_per_segment = pd.DataFrame(
    _todo9_segment_rows
).set_index("segment_id")

TODO9_SPECIAL_GATE_PASSED = bool(
    todo9_residual_gate_summary["F1"] >= TODO9_SPECIAL_F1_GATE
)
TODO9_EXACT_GATE_PASSED = bool(
    todo9_sequential_summary[
        "hybrid exact complete-state share"
    ]
    >= TODO9_EXACT_COMPLETE_STATE_GATE
)
TODO9_MATERIAL_IMPROVED_ALL_SEGMENTS = bool(
    (
        todo9_sequential_per_segment["hybrid_material"]
        > todo9_sequential_per_segment["baseline_material"]
    ).all()
)
TODO9_RECURSIVE_ROLLOUT_RUN = False
TODO9_STATUS = (
    "material_one_step_improved_exact_gate_failed_rollout_closed"
    if TODO9_SPECIAL_GATE_PASSED and not TODO9_EXACT_GATE_PASSED
    else "one_step_probe_failed_rollout_closed"
)

assert TODO9_SPECIAL_GATE_PASSED
assert TODO9_MATERIAL_IMPROVED_ALL_SEGMENTS
assert not TODO9_EXACT_GATE_PASSED
assert not TODO9_RECURSIVE_ROLLOUT_RUN

todo9_manifest = pd.Series(
    {
        "status": TODO9_STATUS,
        "event rows": len(todo9_event_dataset),
        "all segments used through LOSO": True,
        "pristine test": False,
        "future-derived columns used as features": False,
        "period-2 rule F1": float(todo9_period2_rule_summary["F1"]),
        "residual gate F1": float(todo9_residual_gate_summary["F1"]),
        "exact gate passed": TODO9_EXACT_GATE_PASSED,
        "material improved on all segments": (
            TODO9_MATERIAL_IMPROVED_ALL_SEGMENTS
        ),
        "recursive rollout run": TODO9_RECURSIVE_ROLLOUT_RUN,
    },
    name="value",
)

display(
    todo9_period2_rule_summary.to_frame(),
    todo9_special_theta_summary.to_frame(),
    todo9_tree_probe_summary.to_frame(),
    todo9_residual_gate_summary.to_frame(),
    todo9_conditional_summary.to_frame(),
    todo9_sequential_summary.to_frame(),
    todo9_sequential_per_segment.style.format(precision=6),
    todo9_manifest.to_frame(),
)
