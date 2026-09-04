"""TODO 13 companion: TODO 11 entry detector plus TODO 12 continuation.

This is a single-path state machine, not a voting ensemble.  The frozen general
residual RF is queried only while NORMAL; the separately trained continuation
RF is queried only while SPECIAL_ACTIVE or immediately after PERIOD2.
"""

from itertools import combinations


TODO13_ENTRY_THRESHOLDS = TODO12_ENTRY_THRESHOLDS
TODO13_CONTINUATION_THRESHOLDS = TODO12_CONTINUATION_THRESHOLDS
TODO13_MAXIMUM_MATERIAL_FAILURES = TODO12_MAXIMUM_MATERIAL_FAILURES
TODO13_SPECIAL_F1_GATE = TODO12_SPECIAL_F1_GATE
TODO13_RUN_F1_GATE = TODO12_RUN_F1_GATE
TODO13_MAXIMUM_MEDIAN_RUN_DELAY = TODO12_MAXIMUM_MEDIAN_RUN_DELAY
TODO13_MINIMUM_IMPROVED_SEGMENTS = TODO12_MINIMUM_IMPROVED_SEGMENTS
TODO13_RECURSIVE_MEDIAN_GATE = TODO12_RECURSIVE_MEDIAN_GATE


def _todo13_fit_general_model(excluded_segments):
    _excluded = np.asarray(tuple(excluded_segments), dtype=int)
    _train_mask = ~np.isin(_todo9_groups, _excluded)
    _model = RandomForestClassifier(
        n_estimators=TODO9_RF_ESTIMATORS,
        max_depth=TODO9_RF_MAX_DEPTH,
        min_samples_leaf=TODO9_RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=TODO9_RANDOM_STATE,
        n_jobs=-1,
    )
    _model.fit(
        todo9_gate_X.loc[_train_mask],
        _todo9_residual_y[_train_mask],
    )
    return _model


def _todo13_add_general_entry_probabilities(cache, model):
    """Augment a TODO 12 fold cache with causal general-RF entry scores."""
    _augmented = dict(cache)
    _p1_frame = _todo11_all_X.iloc[cache["p1_events"]]
    _augmented["p1_entry_probability"] = _todo12_phase_probability(
        model, _p1_frame, False
    )

    _vectors = []
    _keys = []
    for _local, (_transition_row, _event_index) in enumerate(
        zip(cache["transition_rows"], cache["p2_events"])
    ):
        if cache["p2_loop"][_local]:
            continue
        _data_index = _todo9_transition_indices[_transition_row]
        _previous_p2 = todo7_p2[_data_index - 1]
        _previous_angle = todo7_angle2[_data_index - 1]
        for _scenario in (0, 1):
            _driver_distance = float(
                cache["p2_distance"][_local, _scenario]
            )
            _baseline = _todo8_update_object(
                _previous_p2,
                _previous_angle,
                _driver_distance,
                -1.0,
            )
            _record = _todo9_feature_record(
                _event_index, _baseline, _driver_distance
            )
            _vectors.append(_todo10_feature_vector(_record))
            _keys.append((_local, _scenario))

    _p2_probability = np.zeros_like(
        cache["p2_entry_probability"], dtype=float
    )
    if _vectors:
        _frame = pd.DataFrame(
            np.asarray(_vectors, dtype=np.float32),
            columns=TODO9_FEATURE_COLUMNS,
        )
        _probability = _todo12_phase_probability(model, _frame, False)
        for _index, (_local, _scenario) in enumerate(_keys):
            _p2_probability[_local, _scenario] = _probability[_index]
    _augmented["p2_entry_probability"] = _p2_probability
    return _augmented


# ---------------------------------------------------------------------------
# 13.A — pairwise general entry models; continuation caches stay frozen
# ---------------------------------------------------------------------------
_todo13_nested_caches = {}
_todo13_pair_models_fitted = 0
for _first, _second in combinations(_todo11_segments, 2):
    _general_model = _todo13_fit_general_model((_first, _second))
    _todo13_pair_models_fitted += 1
    for _outer, _inner in ((_first, _second), (_second, _first)):
        _todo13_nested_caches[(int(_outer), int(_inner))] = (
            _todo13_add_general_entry_probabilities(
                _todo12_nested_caches[(int(_outer), int(_inner))],
                _general_model,
            )
        )

assert _todo13_pair_models_fitted == 55


# ---------------------------------------------------------------------------
# 13.B — nested full-sequential threshold selection
# ---------------------------------------------------------------------------
todo13_threshold_by_segment = {}
_todo13_threshold_rows = []
for _outer in _todo11_segments:
    _best = None
    for _entry_threshold in TODO13_ENTRY_THRESHOLDS:
        for _continuation_threshold in TODO13_CONTINUATION_THRESHOLDS:
            _material_errors = []
            _exact_errors = []
            _tp = _fp = _fn = 0
            _run_tp = _run_fp = _run_fn = 0
            for _inner in _todo11_segments:
                if _inner == _outer:
                    continue
                _result = _todo12_simulate_fold(
                    _todo13_nested_caches[(int(_outer), int(_inner))],
                    _entry_threshold,
                    _continuation_threshold,
                )
                _material_errors.append(
                    1.0 - float(_result["material"].mean())
                )
                _exact_errors.append(1.0 - float(_result["exact"].mean()))
                _counts = _todo12_binary_counts(
                    _result["target"], _result["prediction"]
                )
                _tp += _counts[0]
                _fp += _counts[1]
                _fn += _counts[2]
                _run_tp += _result["run_tp"]
                _run_fp += _result["run_fp"]
                _run_fn += _result["run_fn"]
            _, _, _event_f1 = _todo12_f1_from_counts(_tp, _fp, _fn)
            _, _, _run_f1 = _todo12_f1_from_counts(
                _run_tp, _run_fp, _run_fn
            )
            _candidate = {
                "entry": float(_entry_threshold),
                "continuation": float(_continuation_threshold),
                "macro_material_error": float(np.mean(_material_errors)),
                "residual_run_f1": float(_run_f1),
                "macro_exact_error": float(np.mean(_exact_errors)),
                "residual_f1": float(_event_f1),
            }
            _key = (
                _candidate["macro_material_error"],
                -_candidate["residual_run_f1"],
                _candidate["macro_exact_error"],
                -_candidate["residual_f1"],
                -_candidate["entry"],
                -_candidate["continuation"],
            )
            if _best is None or _key < _best[0]:
                _best = (_key, _candidate)
    _chosen = _best[1]
    todo13_threshold_by_segment[int(_outer)] = (
        _chosen["entry"],
        _chosen["continuation"],
    )
    _todo13_threshold_rows.append(
        {"segment_id": int(_outer), **_chosen}
    )

todo13_threshold_selection = pd.DataFrame(
    _todo13_threshold_rows
).set_index("segment_id")


# ---------------------------------------------------------------------------
# 13.C — outer-LOSO full sequential one-step
# ---------------------------------------------------------------------------
todo13_outer_caches = {}
todo13_outer_results = {}
todo13_outer_models = {}
for _segment_id in _todo11_segments:
    _segment_id = int(_segment_id)
    _general_model = todo9_residual_models[_segment_id]
    _continuation_model = todo12_outer_models[_segment_id]["continuation"]
    todo13_outer_models[_segment_id] = {
        "entry": _general_model,
        "continuation": _continuation_model,
    }
    _cache = _todo13_add_general_entry_probabilities(
        todo12_outer_caches[_segment_id], _general_model
    )
    todo13_outer_caches[_segment_id] = _cache
    _entry, _continuation = todo13_threshold_by_segment[_segment_id]
    todo13_outer_results[_segment_id] = _todo12_simulate_fold(
        _cache, _entry, _continuation
    )

_todo13_exact = np.zeros(_todo12_transition_count, dtype=bool)
_todo13_material = np.zeros(_todo12_transition_count, dtype=bool)
_todo13_position_error = np.empty(_todo12_transition_count, dtype=float)
_todo13_angle_error = np.empty(_todo12_transition_count, dtype=float)
_todo13_distance_error = np.empty(_todo12_transition_count, dtype=float)
_todo13_p1_special = np.zeros(_todo12_transition_count, dtype=bool)
_todo13_p2_special = np.zeros(_todo12_transition_count, dtype=bool)
_todo13_all_targets = []
_todo13_all_predictions = []
_todo13_run_tp = _todo13_run_fp = _todo13_run_fn = 0
_todo13_onset_delays = []
_todo13_exit_delays = []

for _segment_id, _result in todo13_outer_results.items():
    _rows = todo13_outer_caches[_segment_id]["transition_rows"]
    _todo13_exact[_rows] = _result["exact"]
    _todo13_material[_rows] = _result["material"]
    _todo13_position_error[_rows] = _result["position_error"]
    _todo13_angle_error[_rows] = _result["angle_error"]
    _todo13_distance_error[_rows] = _result["distance_error"]
    _todo13_p1_special[_rows] = _result["p1_special"]
    _todo13_p2_special[_rows] = _result["p2_special"]
    _todo13_all_targets.append(_result["target"])
    _todo13_all_predictions.append(_result["prediction"])
    _todo13_run_tp += _result["run_tp"]
    _todo13_run_fp += _result["run_fp"]
    _todo13_run_fn += _result["run_fn"]
    _todo13_onset_delays.extend(_result["onset_delays"])
    _todo13_exit_delays.extend(_result["exit_delays"])

_todo13_all_targets = np.concatenate(_todo13_all_targets)
_todo13_all_predictions = np.concatenate(_todo13_all_predictions)
_todo13_tp, _todo13_fp, _todo13_fn = _todo12_binary_counts(
    _todo13_all_targets, _todo13_all_predictions
)
(
    _todo13_precision,
    _todo13_recall,
    _todo13_f1,
) = _todo12_f1_from_counts(_todo13_tp, _todo13_fp, _todo13_fn)
(
    _todo13_run_precision,
    _todo13_run_recall,
    _todo13_run_f1,
) = _todo12_f1_from_counts(
    _todo13_run_tp, _todo13_run_fp, _todo13_run_fn
)
_todo13_median_absolute_onset_delay = float(
    np.median(np.abs(_todo13_onset_delays))
) if _todo13_onset_delays else np.inf
_todo13_median_absolute_exit_delay = float(
    np.median(np.abs(_todo13_exit_delays))
) if _todo13_exit_delays else np.inf

todo13_one_step_summary = pd.Series(
    {
        "TODO10 exact complete-state share": float(
            _todo10_sequential_exact.mean()
        ),
        "TODO11 exact complete-state share": float(
            _todo11_sequential_exact.mean()
        ),
        "TODO12 exact complete-state share": float(_todo12_exact.mean()),
        "TODO13 exact complete-state share": float(_todo13_exact.mean()),
        "TODO10 material failures": int(
            (~_todo10_sequential_material).sum()
        ),
        "TODO11 material failures": int(
            (~_todo11_sequential_material).sum()
        ),
        "TODO12 material failures": int((~_todo12_material).sum()),
        "TODO13 material failures": int((~_todo13_material).sum()),
        "TODO11 residual F1": float(_todo11_event_f1),
        "TODO12 residual F1": float(_todo12_f1),
        "TODO13 residual precision": _todo13_precision,
        "TODO13 residual recall": _todo13_recall,
        "TODO13 residual F1": _todo13_f1,
        "TODO13 residual-run F1": _todo13_run_f1,
        "TODO13 median absolute onset delay": (
            _todo13_median_absolute_onset_delay
        ),
        "TODO13 median absolute exit delay": (
            _todo13_median_absolute_exit_delay
        ),
        "TODO13 maximum-angle MAE": float(_todo13_angle_error.mean()),
    },
    name="value",
)

_todo13_segment_rows = []
for _segment_id in _todo11_segments:
    _segment_id = int(_segment_id)
    _mask = _todo9_transition_segments == _segment_id
    _todo13_segment_rows.append(
        {
            "segment_id": _segment_id,
            "transitions": int(_mask.sum()),
            "entry": todo13_threshold_by_segment[_segment_id][0],
            "continuation": todo13_threshold_by_segment[_segment_id][1],
            "TODO10_exact": float(_todo10_sequential_exact[_mask].mean()),
            "TODO11_exact": float(_todo11_sequential_exact[_mask].mean()),
            "TODO13_exact": float(_todo13_exact[_mask].mean()),
            "TODO11_material": float(
                _todo11_sequential_material[_mask].mean()
            ),
            "TODO13_material": float(_todo13_material[_mask].mean()),
        }
    )
todo13_one_step_per_segment = pd.DataFrame(
    _todo13_segment_rows
).set_index("segment_id")

_todo13_event_predictions = np.zeros(len(todo9_event_dataset), dtype=bool)
_todo13_event_predictions[_todo9_p1_event_indices] = _todo13_p1_special
_todo13_event_predictions[_todo9_p2_event_indices] = _todo13_p2_special
_todo13_phase_rows = []
for _phase_name, _phase_mask in (
    ("entry", _todo12_entry_target),
    ("continuation", _todo12_continuation_target),
    ("exit", _todo12_exit_target),
):
    if _phase_name == "exit":
        _score = float((~_todo13_event_predictions[_phase_mask]).mean())
        _metric = "correct exit share"
    else:
        _score = float(_todo13_event_predictions[_phase_mask].mean())
        _metric = "recall"
    _todo13_phase_rows.append(
        {
            "phase": _phase_name,
            "events": int(_phase_mask.sum()),
            "metric": _metric,
            "value": _score,
        }
    )
todo13_phase_summary = pd.DataFrame(_todo13_phase_rows).set_index("phase")

TODO13_IMPROVED_SEGMENTS = int(
    (
        todo13_one_step_per_segment["TODO13_exact"]
        > todo13_one_step_per_segment["TODO10_exact"]
    ).sum()
)
TODO13_FAILURE_GATE_PASSED = bool(
    int((~_todo13_material).sum()) <= TODO13_MAXIMUM_MATERIAL_FAILURES
)
TODO13_F1_GATE_PASSED = bool(_todo13_f1 >= TODO13_SPECIAL_F1_GATE)
TODO13_RUN_GATE_PASSED = bool(
    _todo13_run_f1 >= TODO13_RUN_F1_GATE
    and _todo13_median_absolute_onset_delay
    <= TODO13_MAXIMUM_MEDIAN_RUN_DELAY
    and _todo13_median_absolute_exit_delay
    <= TODO13_MAXIMUM_MEDIAN_RUN_DELAY
)
TODO13_GLOBAL_GATE_PASSED = bool(
    _todo13_exact.mean() > _todo11_sequential_exact.mean()
    and _todo13_material.mean() > _todo11_sequential_material.mean()
)
TODO13_SEGMENT_GATE_PASSED = bool(
    TODO13_IMPROVED_SEGMENTS >= TODO13_MINIMUM_IMPROVED_SEGMENTS
)
TODO13_ONE_STEP_GATE_PASSED = bool(
    TODO13_FAILURE_GATE_PASSED
    and TODO13_F1_GATE_PASSED
    and TODO13_RUN_GATE_PASSED
    and TODO13_GLOBAL_GATE_PASSED
    and TODO13_SEGMENT_GATE_PASSED
)


# ---------------------------------------------------------------------------
# 13.D — error taxonomy
# ---------------------------------------------------------------------------
_todo13_failure_rows = []
for _transition_row in np.flatnonzero(~_todo13_material):
    _causes = []
    for _event_index, _predicted in (
        (_todo9_p1_event_indices[_transition_row], _todo13_p1_special[_transition_row]),
        (_todo9_p2_event_indices[_transition_row], _todo13_p2_special[_transition_row]),
    ):
        if _todo12_entry_target[_event_index] and not _predicted:
            _causes.append("missed entry")
        elif _todo12_continuation_target[_event_index] and not _predicted:
            _causes.append("missed continuation")
        elif not _todo12_residual_target[_event_index] and _predicted:
            _causes.append("false special / late exit")
        if _todo12_hidden_initialization[_event_index]:
            _causes.append("segment-start hidden state")
    if not _causes:
        _causes.append("period-2 entry or upstream mixed error")
    _todo13_failure_rows.append(
        {
            "transition_row": int(_transition_row),
            "segment_id": int(_todo9_transition_segments[_transition_row]),
            "source_row": int(_todo9_transition_source_rows[_transition_row]),
            "cause": " + ".join(sorted(set(_causes))),
            "maximum_position_error": float(
                _todo13_position_error[_transition_row]
            ),
            "maximum_angle_error": float(
                _todo13_angle_error[_transition_row]
            ),
        }
    )

todo13_material_failures = pd.DataFrame(
    _todo13_failure_rows,
    columns=[
        "transition_row",
        "segment_id",
        "source_row",
        "cause",
        "maximum_position_error",
        "maximum_angle_error",
    ],
)
todo13_material_failure_summary = (
    todo13_material_failures.groupby("cause", as_index=True)
    .agg(
        transitions=("source_row", "size"),
        segments=("segment_id", "nunique"),
        maximum_angle_error=("maximum_angle_error", "max"),
        maximum_position_error=("maximum_position_error", "max"),
    )
    .sort_values("transitions", ascending=False)
)


# ---------------------------------------------------------------------------
# 13.E — recursive single-path rollout after promotion only
# ---------------------------------------------------------------------------
def _todo13_rollout_segment(segment_frame, segment_id):
    _segment_frame = segment_frame.sort_index()
    _truth_p1 = _segment_frame[["x1", "y1"]].to_numpy(dtype=float)
    _truth_p2 = _segment_frame[["x2", "y2"]].to_numpy(dtype=float)
    _truth_a1 = _segment_frame["angle1"].to_numpy(dtype=float)
    _truth_a2 = _segment_frame["angle2"].to_numpy(dtype=float)
    _truth_distance = _segment_frame["distance"].to_numpy(dtype=float)
    _models = todo13_outer_models[int(segment_id)]
    _entry, _continuation = todo13_threshold_by_segment[int(segment_id)]
    _h1 = _todo10_initial_history(_truth_p1[0], _truth_a1[0])
    _h2 = _todo10_initial_history(_truth_p2[0], _truth_a2[0])
    _distance = [float(_truth_distance[0])]

    for _ in range(1, len(_segment_frame)):
        _rho_before = float(
            np.linalg.norm(_h2["position"][-1] - _h1["position"][-1])
        )
        _p1, _, _, _ = _todo12_recursive_object_update(
            "object1",
            int(segment_id),
            _h1["position"][-1],
            _h1["angle"][-1],
            _rho_before,
            1.0,
            _h1,
            _models,
            _entry,
            _continuation,
        )
        _new_distance = float(np.linalg.norm(_p1 - _h2["position"][-1]))
        _todo12_recursive_object_update(
            "object2",
            int(segment_id),
            _h2["position"][-1],
            _h2["angle"][-1],
            _new_distance,
            -1.0,
            _h2,
            _models,
            _entry,
            _continuation,
        )
        _distance.append(_new_distance)

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
        "first_numerical_failure": _todo8_first_failure(_exact_failure),
        "first_material_failure": _todo8_first_failure(_material_failure),
    }


todo13_rollouts = {}
_todo13_rollout_rows = []
_todo13_horizon_rows = []
if TODO13_ONE_STEP_GATE_PASSED:
    for _segment_id, _segment_frame in todo7_df.groupby(
        "segment_id", sort=True
    ):
        _rollout = _todo13_rollout_segment(_segment_frame, _segment_id)
        todo13_rollouts[int(_segment_id)] = _rollout
        _transition_count = len(_segment_frame) - 1
        _first_material = _rollout["first_material_failure"]
        _first_exact = _rollout["first_numerical_failure"]
        _todo13_rollout_rows.append(
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
            _hybrid = _rollout["predicted_positions"][1 : _horizon + 1]
            _todo10_positions = todo10_rollouts[int(_segment_id)][
                "predicted_positions"
            ][1 : _horizon + 1]
            _todo13_horizon_rows.append(
                {
                    "segment_id": int(_segment_id),
                    "horizon": _label,
                    "TODO13 RMSE": float(
                        np.sqrt(np.mean((_hybrid - _truth) ** 2))
                    ),
                    "TODO10 RMSE": float(
                        np.sqrt(np.mean((_todo10_positions - _truth) ** 2))
                    ),
                }
            )

todo13_rollout_summary = pd.DataFrame(_todo13_rollout_rows)
if len(todo13_rollout_summary):
    todo13_rollout_summary = todo13_rollout_summary.set_index("segment_id")
todo13_rollout_horizons = pd.DataFrame(_todo13_horizon_rows)
todo13_rollout_macro = (
    todo13_rollout_horizons.groupby("horizon", sort=False)
    .agg(
        segments=("segment_id", "nunique"),
        TODO13_RMSE=("TODO13 RMSE", "mean"),
        TODO10_RMSE=("TODO10 RMSE", "mean"),
    )
    if len(todo13_rollout_horizons)
    else pd.DataFrame()
)
TODO13_RECURSIVE_ROLLOUT_RUN = bool(TODO13_ONE_STEP_GATE_PASSED)
TODO13_MEDIAN_MATERIAL_PREFIX = (
    float(todo13_rollout_summary["first material failure"].median())
    if TODO13_RECURSIVE_ROLLOUT_RUN
    else np.nan
)
TODO13_ROLLOUT_GATE_PASSED = bool(
    TODO13_RECURSIVE_ROLLOUT_RUN
    and TODO13_MEDIAN_MATERIAL_PREFIX >= TODO13_RECURSIVE_MEDIAN_GATE
    and todo13_rollout_macro.loc["100", "TODO13_RMSE"]
    < todo13_rollout_macro.loc["100", "TODO10_RMSE"]
    and todo13_rollout_macro.loc["500", "TODO13_RMSE"]
    < todo13_rollout_macro.loc["500", "TODO10_RMSE"]
    and todo13_rollout_macro.loc["full", "TODO13_RMSE"]
    <= todo13_rollout_macro.loc["full", "TODO10_RMSE"]
)

TODO13_STATUS = (
    "todo11_entry_continuation_automaton_all_gates_passed"
    if TODO13_ROLLOUT_GATE_PASSED
    else "todo11_entry_one_step_passed_rollout_gate_failed"
    if TODO13_RECURSIVE_ROLLOUT_RUN
    else "todo11_entry_one_step_gate_failed_rollout_closed"
)
todo13_manifest = pd.Series(
    {
        "status": TODO13_STATUS,
        "general pairwise inner models fitted": _todo13_pair_models_fitted,
        "continuation pairwise caches reused": True,
        "one classifier queried per state": True,
        "trajectory ensemble used": False,
        "inner objective uses full sequential state": True,
        "thresholds selected with outer labels": False,
        "material failure gate passed": TODO13_FAILURE_GATE_PASSED,
        "residual F1 gate passed": TODO13_F1_GATE_PASSED,
        "residual-run F1": _todo13_run_f1,
        "run/delay gate passed": TODO13_RUN_GATE_PASSED,
        "global TODO11 improvement gate passed": TODO13_GLOBAL_GATE_PASSED,
        "segments improved versus TODO10": TODO13_IMPROVED_SEGMENTS,
        "segment gate passed": TODO13_SEGMENT_GATE_PASSED,
        "one-step gate passed": TODO13_ONE_STEP_GATE_PASSED,
        "recursive rollout run": TODO13_RECURSIVE_ROLLOUT_RUN,
        "median first material failure": TODO13_MEDIAN_MATERIAL_PREFIX,
        "rollout gate passed": TODO13_ROLLOUT_GATE_PASSED,
        "pristine test": False,
    },
    name="value",
)

assert _todo13_pair_models_fitted == 55
assert set(todo13_threshold_by_segment) == set(_todo11_segments)
assert np.isfinite(_todo13_position_error).all()
assert np.isfinite(_todo13_angle_error).all()
assert np.isfinite(_todo13_distance_error).all()
assert not TODO13_RECURSIVE_ROLLOUT_RUN or TODO13_ONE_STEP_GATE_PASSED
if TODO13_RECURSIVE_ROLLOUT_RUN:
    assert todo13_rollout_summary["finite"].all()

display(
    Markdown("#### Nested TODO11-entry + continuation threshold selection"),
    todo13_threshold_selection.style.format(precision=6),
    Markdown("#### Outer-LOSO full sequential one-step"),
    todo13_one_step_summary.to_frame(),
    todo13_one_step_per_segment.style.format(precision=6),
    Markdown("#### Entry / continuation / exit diagnostics"),
    todo13_phase_summary.style.format(precision=6),
    Markdown("#### Remaining material failures"),
    todo13_material_failure_summary.style.format(precision=6),
    Markdown("#### Recursive rollout"),
    todo13_rollout_summary.style.format(precision=6)
    if len(todo13_rollout_summary)
    else todo13_rollout_summary,
    todo13_rollout_macro.style.format(precision=6)
    if len(todo13_rollout_macro)
    else todo13_rollout_macro,
    Markdown("#### TODO 13 manifest"),
    todo13_manifest.to_frame(),
)
