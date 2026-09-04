"""TODO 12 companion: separate entry and continuation/exit automata.

The deterministic transition map and exact special-angle law remain frozen.
Two Random Forests with the TODO 9 architecture estimate different causal
questions: entering residual-special from NORMAL, and staying residual-special
versus exiting after SPECIAL_ACTIVE.  Thresholds are selected by a full
sequential object1 -> predicted distance -> object2 nested-LOSO simulation.
"""

from itertools import combinations


TODO12_ENTRY_THRESHOLDS = tuple(np.round(np.arange(0.50, 0.96, 0.05), 2))
TODO12_CONTINUATION_THRESHOLDS = tuple(
    np.round(np.arange(0.15, 0.81, 0.05), 2)
)
TODO12_MAXIMUM_MATERIAL_FAILURES = 40
TODO12_SPECIAL_F1_GATE = 0.915
TODO12_RUN_F1_GATE = 0.80
TODO12_MAXIMUM_MEDIAN_RUN_DELAY = 2.0
TODO12_MINIMUM_IMPROVED_SEGMENTS = 8
TODO12_RECURSIVE_MEDIAN_GATE = 190
TODO12_RANDOM_STATE = TODO9_RANDOM_STATE

_todo12_event_segments = todo9_event_dataset["segment_id"].to_numpy(
    dtype=int
)
_todo12_residual_target = np.asarray(
    _todo9_residual_target_all, dtype=bool
)
_todo12_period2_target = np.asarray(_todo9_loop_target, dtype=bool)
# PERIOD2 has a separate deterministic output, but its end passes through the
# continuation/exit head instead of being mislabeled as a fresh entry.
_todo12_active_target = (
    _todo12_residual_target | _todo12_period2_target
)
_todo12_hidden_initialization = todo9_event_dataset["event_mode"].eq(
    "segment_start_special"
).to_numpy()
_todo12_previous_active_target = np.zeros(
    len(todo9_event_dataset), dtype=bool
)
for _indices in _todo11_sequence_indices:
    _todo12_previous_active_target[_indices[1:]] = (
        _todo12_active_target[_indices[:-1]]
    )

_todo12_entry_target = (
    _todo12_residual_target & ~_todo12_previous_active_target
)
_todo12_continuation_target = (
    _todo12_residual_target & _todo12_previous_active_target
)
_todo12_exit_target = (
    ~_todo12_residual_target
    & _todo12_previous_active_target
    & ~np.asarray(_todo9_loop_prediction, dtype=bool)
    & ~_todo12_period2_target
)
_todo12_entry_population = (
    _todo11_base_candidate
    & ~_todo12_previous_active_target
    & ~np.asarray(_todo9_loop_prediction, dtype=bool)
    & ~_todo12_period2_target
    & ~_todo12_hidden_initialization
)
_todo12_continuation_population = (
    _todo12_previous_active_target
    & ~np.asarray(_todo9_loop_prediction, dtype=bool)
    & ~_todo12_period2_target
    & ~_todo12_hidden_initialization
)

_todo12_previous_special_column = TODO9_FEATURE_COLUMNS.index(
    "previous_special"
)


def _todo12_phase_frame(frame, active):
    """Use the automaton's state, not the observed previous-special flag."""
    _frame = frame.copy()
    _frame["previous_special"] = float(active)
    return _frame


def _todo12_fit_phase_models(excluded_segments):
    """Fit fixed-architecture entry and continuation models."""
    _excluded = np.asarray(tuple(excluded_segments), dtype=int)
    _available = ~np.isin(_todo12_event_segments, _excluded)
    _masks = {
        "entry": _available & _todo12_entry_population,
        "continuation": _available & _todo12_continuation_population,
    }
    _models = {}
    for _phase, _mask in _masks.items():
        _target = _todo12_residual_target[_mask]
        assert set(np.unique(_target)) == {False, True}
        _model = RandomForestClassifier(
            n_estimators=TODO9_RF_ESTIMATORS,
            max_depth=TODO9_RF_MAX_DEPTH,
            min_samples_leaf=TODO9_RF_MIN_SAMPLES_LEAF,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=TODO12_RANDOM_STATE,
            n_jobs=-1,
        )
        _model.fit(
            _todo12_phase_frame(
                _todo11_all_X.loc[_mask], _phase == "continuation"
            ),
            _target,
        )
        _models[_phase] = _model
    return _models


def _todo12_phase_probability(model, frame, active):
    return _todo11_true_probability(
        model, _todo12_phase_frame(frame, active)
    )


def _todo12_vector_probability(model, vector, active):
    _vector = np.asarray(vector, dtype=np.float32).copy()
    _vector[_todo12_previous_special_column] = float(active)
    return _todo11_fast_forest_probability(model, _vector)


def _todo12_build_fold_cache(segment_id, models):
    """Cache both possible object-1 branches and resulting object-2 inputs."""
    _transition_rows = np.flatnonzero(
        _todo9_transition_segments == int(segment_id)
    )
    _count = len(_transition_rows)
    _p1_events = _todo9_p1_event_indices[_transition_rows]
    _p2_events = _todo9_p2_event_indices[_transition_rows]
    _data_indices = _todo9_transition_indices[_transition_rows]

    _p1_base_position = _todo9_baseline_position[_p1_events].copy()
    _p1_base_angle = _todo9_baseline_angle[_p1_events].copy()
    _p1_special_position = _todo11_formula_position[_p1_events].copy()
    _p1_special_angle = _todo11_formula_angle[_p1_events].copy()
    _p1_loop = _todo9_loop_prediction[_p1_events]
    _p1_base_position[_p1_loop] = _todo9_position_lag2[
        _p1_events[_p1_loop]
    ]
    _p1_special_position[_p1_loop] = _p1_base_position[_p1_loop]
    _p1_base_angle[_p1_loop] = _todo9_angle_lag2[_p1_events[_p1_loop]]
    _p1_special_angle[_p1_loop] = _p1_base_angle[_p1_loop]
    _p1_frame = _todo11_all_X.iloc[_p1_events]
    _p1_entry_probability = _todo12_phase_probability(
        models["entry"], _p1_frame, False
    )
    _p1_continuation_probability = _todo12_phase_probability(
        models["continuation"], _p1_frame, True
    )

    _p2_base_position = np.empty((_count, 2, 2), dtype=float)
    _p2_special_position = np.empty((_count, 2, 2), dtype=float)
    _p2_base_angle = np.empty((_count, 2), dtype=float)
    _p2_special_angle = np.empty((_count, 2), dtype=float)
    _p2_distance = np.empty((_count, 2), dtype=float)
    _p2_candidate = np.zeros((_count, 2), dtype=bool)
    _p2_vectors = []
    _p2_vector_keys = []
    _p2_loop = _todo9_loop_prediction[_p2_events]

    for _local, (_transition_row, _data_index, _event_index) in enumerate(
        zip(_transition_rows, _data_indices, _p2_events)
    ):
        _event = todo9_event_dataset.loc[_event_index]
        _previous_p2 = todo7_p2[_data_index - 1]
        if _p2_loop[_local]:
            _lag2_index = max(int(_data_index) - 2, 0)
            _loop_position = todo7_p2[_lag2_index]
            _loop_angle = todo7_angle2[_lag2_index]
            for _scenario, _p1_position in enumerate(
                (_p1_base_position[_local], _p1_special_position[_local])
            ):
                _p2_distance[_local, _scenario] = np.linalg.norm(
                    _p1_position - _previous_p2
                )
                _p2_base_position[_local, _scenario] = _loop_position
                _p2_special_position[_local, _scenario] = _loop_position
                _p2_base_angle[_local, _scenario] = _loop_angle
                _p2_special_angle[_local, _scenario] = _loop_angle
            continue

        for _scenario, _p1_position in enumerate(
            (_p1_base_position[_local], _p1_special_position[_local])
        ):
            _driver_distance = float(
                np.linalg.norm(_p1_position - _previous_p2)
            )
            _p2_distance[_local, _scenario] = _driver_distance
            _baseline = _todo8_update_object(
                _previous_p2,
                todo7_angle2[_data_index - 1],
                _driver_distance,
                -1.0,
            )
            _p2_base_position[_local, _scenario] = _baseline["position"]
            _p2_base_angle[_local, _scenario] = _baseline["angle"]
            _special_angle = _todo10_special_angle(
                _baseline["free_angle"], _baseline["angle"]
            )
            _limits = _todo8_oriented_limits(_special_angle)
            _p2_special_position[_local, _scenario] = np.clip(
                _baseline["free_position"], -_limits, _limits
            )
            _p2_special_angle[_local, _scenario] = _special_angle
            _p2_candidate[_local, _scenario] = bool(
                np.any(_baseline["crossing"])
                or _event["previous_wall_contact"]
                or _event["previous_period2_exact"]
            )
            _record = _todo9_feature_record(
                _event_index, _baseline, _driver_distance
            )
            _p2_vectors.append(_todo10_feature_vector(_record))
            _p2_vector_keys.append((_local, _scenario))

    _p2_entry_probability = np.zeros((_count, 2), dtype=float)
    _p2_continuation_probability = np.zeros((_count, 2), dtype=float)
    if _p2_vectors:
        _vector_frame = pd.DataFrame(
            np.asarray(_p2_vectors, dtype=np.float32),
            columns=TODO9_FEATURE_COLUMNS,
        )
        _entry_probability = _todo12_phase_probability(
            models["entry"], _vector_frame, False
        )
        _continuation_probability = _todo12_phase_probability(
            models["continuation"], _vector_frame, True
        )
        for _index, (_local, _scenario) in enumerate(_p2_vector_keys):
            _p2_entry_probability[_local, _scenario] = (
                _entry_probability[_index]
            )
            _p2_continuation_probability[_local, _scenario] = (
                _continuation_probability[_index]
            )

    return {
        "segment_id": int(segment_id),
        "transition_rows": _transition_rows,
        "p1_events": _p1_events,
        "p2_events": _p2_events,
        "p1_loop": _p1_loop,
        "p1_candidate": _todo11_base_candidate[_p1_events],
        "p1_entry_probability": _p1_entry_probability,
        "p1_continuation_probability": _p1_continuation_probability,
        "p1_base_position": _p1_base_position,
        "p1_special_position": _p1_special_position,
        "p1_base_angle": _p1_base_angle,
        "p1_special_angle": _p1_special_angle,
        "p2_loop": _p2_loop,
        "p2_candidate": _p2_candidate,
        "p2_entry_probability": _p2_entry_probability,
        "p2_continuation_probability": _p2_continuation_probability,
        "p2_base_position": _p2_base_position,
        "p2_special_position": _p2_special_position,
        "p2_base_angle": _p2_base_angle,
        "p2_special_angle": _p2_special_angle,
        "p2_distance": _p2_distance,
    }


def _todo12_binary_counts(target, prediction):
    _target = np.asarray(target, dtype=bool)
    _prediction = np.asarray(prediction, dtype=bool)
    return (
        int(np.sum(_target & _prediction)),
        int(np.sum(~_target & _prediction)),
        int(np.sum(_target & ~_prediction)),
    )


def _todo12_f1_from_counts(true_positive, false_positive, false_negative):
    _precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    _recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    _f1 = (
        2.0 * _precision * _recall / (_precision + _recall)
        if _precision + _recall
        else 0.0
    )
    return _precision, _recall, _f1


def _todo12_runs(mask):
    """Return inclusive [start, end] intervals for contiguous True values."""
    _mask = np.asarray(mask, dtype=bool)
    _padded = np.pad(_mask.astype(np.int8), (1, 1))
    _changes = np.diff(_padded)
    _starts = np.flatnonzero(_changes == 1)
    _ends = np.flatnonzero(_changes == -1) - 1
    return list(zip(_starts.tolist(), _ends.tolist()))


def _todo12_run_metrics(target, prediction):
    """One-to-one run matching by IoU >= 0.5 with deterministic ties."""
    _true_runs = _todo12_runs(target)
    _predicted_runs = _todo12_runs(prediction)
    _pairs = []
    for _true_index, (_true_start, _true_end) in enumerate(_true_runs):
        for _pred_index, (_pred_start, _pred_end) in enumerate(
            _predicted_runs
        ):
            _intersection = max(
                0,
                min(_true_end, _pred_end)
                - max(_true_start, _pred_start)
                + 1,
            )
            _union = (
                _true_end
                - _true_start
                + 1
                + _pred_end
                - _pred_start
                + 1
                - _intersection
            )
            _iou = _intersection / _union if _union else 0.0
            if _iou >= 0.5:
                _pairs.append(
                    (
                        -_iou,
                        _pred_start,
                        _true_start,
                        _true_index,
                        _pred_index,
                    )
                )
    _matched_true = set()
    _matched_prediction = set()
    _onset_delays = []
    _exit_delays = []
    for _, _, _, _true_index, _pred_index in sorted(_pairs):
        if (
            _true_index in _matched_true
            or _pred_index in _matched_prediction
        ):
            continue
        _matched_true.add(_true_index)
        _matched_prediction.add(_pred_index)
        _true_start, _true_end = _true_runs[_true_index]
        _pred_start, _pred_end = _predicted_runs[_pred_index]
        _onset_delays.append(_pred_start - _true_start)
        _exit_delays.append(_pred_end - _true_end)
    return {
        "tp": len(_matched_true),
        "fp": len(_predicted_runs) - len(_matched_prediction),
        "fn": len(_true_runs) - len(_matched_true),
        "onset_delays": _onset_delays,
        "exit_delays": _exit_delays,
    }


def _todo12_simulate_fold(cache, entry_threshold, continuation_threshold):
    """Full teacher-forced one-step scan with self-predicted mode state."""
    _count = len(cache["transition_rows"])
    _p1_prediction = np.empty((_count, 2), dtype=float)
    _p2_prediction = np.empty((_count, 2), dtype=float)
    _a1_prediction = np.empty(_count, dtype=float)
    _a2_prediction = np.empty(_count, dtype=float)
    _distance_prediction = np.empty(_count, dtype=float)
    _p1_special = np.zeros(_count, dtype=bool)
    _p2_special = np.zeros(_count, dtype=bool)
    _active1 = False
    _active2 = False

    for _local in range(_count):
        if cache["p1_loop"][_local]:
            _scenario = 0
            _active1 = True
        else:
            if _active1:
                _p1_special[_local] = bool(
                    cache["p1_continuation_probability"][_local]
                    >= continuation_threshold
                )
            else:
                _p1_special[_local] = bool(
                    cache["p1_candidate"][_local]
                    and cache["p1_entry_probability"][_local]
                    >= entry_threshold
                )
            _active1 = bool(_p1_special[_local])
            _scenario = int(_p1_special[_local])

        _p1_prediction[_local] = (
            cache["p1_special_position"][_local]
            if _scenario
            else cache["p1_base_position"][_local]
        )
        _a1_prediction[_local] = (
            cache["p1_special_angle"][_local]
            if _scenario
            else cache["p1_base_angle"][_local]
        )
        _distance_prediction[_local] = cache["p2_distance"][
            _local, _scenario
        ]

        if cache["p2_loop"][_local]:
            _p2_choice = False
            _active2 = True
        elif _active2:
            _p2_choice = bool(
                cache["p2_continuation_probability"][_local, _scenario]
                >= continuation_threshold
            )
            _p2_special[_local] = _p2_choice
            _active2 = _p2_choice
        else:
            _p2_choice = bool(
                cache["p2_candidate"][_local, _scenario]
                and cache["p2_entry_probability"][_local, _scenario]
                >= entry_threshold
            )
            _p2_special[_local] = _p2_choice
            _active2 = _p2_choice

        _p2_prediction[_local] = (
            cache["p2_special_position"][_local, _scenario]
            if _p2_choice
            else cache["p2_base_position"][_local, _scenario]
        )
        _a2_prediction[_local] = (
            cache["p2_special_angle"][_local, _scenario]
            if _p2_choice
            else cache["p2_base_angle"][_local, _scenario]
        )

    _rows = cache["transition_rows"]
    _position_error = np.maximum(
        np.linalg.norm(_p1_prediction - _todo9_truth_p1[_rows], axis=1),
        np.linalg.norm(_p2_prediction - _todo9_truth_p2[_rows], axis=1),
    )
    _angle_error = np.maximum(
        np.abs(
            _todo7_wrap_degrees(
                _a1_prediction - _todo9_truth_angle1[_rows]
            )
        ),
        np.abs(
            _todo7_wrap_degrees(
                _a2_prediction - _todo9_truth_angle2[_rows]
            )
        ),
    )
    _distance_error = np.abs(
        _distance_prediction - _todo9_truth_distance[_rows]
    )
    _exact = (
        (_position_error < TODO8_POSITION_TOLERANCE)
        & (_angle_error < TODO8_ANGLE_TOLERANCE)
        & (_distance_error < TODO8_DISTANCE_TOLERANCE)
    )
    _material = (
        (_position_error < TODO8_MATERIAL_POSITION_ERROR)
        & (_angle_error < TODO8_MATERIAL_ANGLE_ERROR)
        & (_distance_error < TODO8_MATERIAL_DISTANCE_ERROR)
    )
    _p1_target = _todo12_residual_target[cache["p1_events"]].copy()
    _p2_target = _todo12_residual_target[cache["p2_events"]].copy()
    _eligible_event = np.concatenate(
        (
            ~_todo12_hidden_initialization[cache["p1_events"]],
            ~_todo12_hidden_initialization[cache["p2_events"]],
        )
    )
    _target = np.concatenate((_p1_target, _p2_target))[_eligible_event]
    _prediction = np.concatenate((_p1_special, _p2_special))[
        _eligible_event
    ]
    _p1_run_target = _p1_target.copy()
    _p2_run_target = _p2_target.copy()
    _p1_run_prediction = _p1_special.copy()
    _p2_run_prediction = _p2_special.copy()
    _p1_hidden = _todo12_hidden_initialization[cache["p1_events"]]
    _p2_hidden = _todo12_hidden_initialization[cache["p2_events"]]
    _p1_run_target[_p1_hidden] = False
    _p2_run_target[_p2_hidden] = False
    _p1_run_prediction[_p1_hidden] = False
    _p2_run_prediction[_p2_hidden] = False
    _run1 = _todo12_run_metrics(_p1_run_target, _p1_run_prediction)
    _run2 = _todo12_run_metrics(_p2_run_target, _p2_run_prediction)
    return {
        "exact": _exact,
        "material": _material,
        "position_error": _position_error,
        "angle_error": _angle_error,
        "distance_error": _distance_error,
        "p1_special": _p1_special,
        "p2_special": _p2_special,
        "target": _target,
        "prediction": _prediction,
        "run_tp": _run1["tp"] + _run2["tp"],
        "run_fp": _run1["fp"] + _run2["fp"],
        "run_fn": _run1["fn"] + _run2["fn"],
        "onset_delays": _run1["onset_delays"] + _run2["onset_delays"],
        "exit_delays": _run1["exit_delays"] + _run2["exit_delays"],
        "p1_prediction": _p1_prediction,
        "p2_prediction": _p2_prediction,
        "a1_prediction": _a1_prediction,
        "a2_prediction": _a2_prediction,
        "distance_prediction": _distance_prediction,
    }


# ---------------------------------------------------------------------------
# 12.A — pairwise inner folds and full-sequential threshold selection
# ---------------------------------------------------------------------------
_todo12_nested_caches = {}
_todo12_pair_models_fitted = 0
for _first, _second in combinations(_todo11_segments, 2):
    _models = _todo12_fit_phase_models((_first, _second))
    _todo12_pair_models_fitted += 2
    _todo12_nested_caches[(int(_first), int(_second))] = (
        _todo12_build_fold_cache(int(_second), _models)
    )
    _todo12_nested_caches[(int(_second), int(_first))] = (
        _todo12_build_fold_cache(int(_first), _models)
    )

assert _todo12_pair_models_fitted == 110

todo12_threshold_by_segment = {}
_todo12_threshold_rows = []
for _outer in _todo11_segments:
    _best = None
    for _entry_threshold in TODO12_ENTRY_THRESHOLDS:
        for _continuation_threshold in TODO12_CONTINUATION_THRESHOLDS:
            _material_errors = []
            _exact_errors = []
            _tp = _fp = _fn = 0
            _run_tp = _run_fp = _run_fn = 0
            for _inner in _todo11_segments:
                if _inner == _outer:
                    continue
                _result = _todo12_simulate_fold(
                    _todo12_nested_caches[(int(_outer), int(_inner))],
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
            _, _, _f1 = _todo12_f1_from_counts(_tp, _fp, _fn)
            _, _, _run_f1 = _todo12_f1_from_counts(
                _run_tp, _run_fp, _run_fn
            )
            _candidate = {
                "entry": float(_entry_threshold),
                "continuation": float(_continuation_threshold),
                "macro_material_error": float(np.mean(_material_errors)),
                "residual_f1": float(_f1),
                "run_f1": float(_run_f1),
                "macro_exact_error": float(np.mean(_exact_errors)),
            }
            _key = (
                _candidate["macro_material_error"],
                -_candidate["run_f1"],
                _candidate["macro_exact_error"],
                -_candidate["residual_f1"],
                -_candidate["entry"],
                -_candidate["continuation"],
            )
            if _best is None or _key < _best[0]:
                _best = (_key, _candidate)
    _chosen = _best[1]
    todo12_threshold_by_segment[int(_outer)] = (
        _chosen["entry"],
        _chosen["continuation"],
    )
    _todo12_threshold_rows.append(
        {"segment_id": int(_outer), **_chosen}
    )

todo12_threshold_selection = pd.DataFrame(
    _todo12_threshold_rows
).set_index("segment_id")


# ---------------------------------------------------------------------------
# 12.B — outer-LOSO full sequential one-step evaluation
# ---------------------------------------------------------------------------
todo12_outer_models = {}
todo12_outer_caches = {}
todo12_outer_results = {}
for _segment_id in _todo11_segments:
    _models = _todo12_fit_phase_models((_segment_id,))
    todo12_outer_models[int(_segment_id)] = _models
    _cache = _todo12_build_fold_cache(int(_segment_id), _models)
    todo12_outer_caches[int(_segment_id)] = _cache
    _entry, _continuation = todo12_threshold_by_segment[int(_segment_id)]
    todo12_outer_results[int(_segment_id)] = _todo12_simulate_fold(
        _cache, _entry, _continuation
    )

_todo12_transition_count = len(_todo9_transition_indices)
_todo12_exact = np.zeros(_todo12_transition_count, dtype=bool)
_todo12_material = np.zeros(_todo12_transition_count, dtype=bool)
_todo12_position_error = np.empty(_todo12_transition_count, dtype=float)
_todo12_angle_error = np.empty(_todo12_transition_count, dtype=float)
_todo12_distance_error = np.empty(_todo12_transition_count, dtype=float)
_todo12_p1_special = np.zeros(_todo12_transition_count, dtype=bool)
_todo12_p2_special = np.zeros(_todo12_transition_count, dtype=bool)
_todo12_all_targets = []
_todo12_all_predictions = []
_todo12_run_tp = _todo12_run_fp = _todo12_run_fn = 0
_todo12_onset_delays = []
_todo12_exit_delays = []

for _segment_id, _result in todo12_outer_results.items():
    _rows = todo12_outer_caches[_segment_id]["transition_rows"]
    _todo12_exact[_rows] = _result["exact"]
    _todo12_material[_rows] = _result["material"]
    _todo12_position_error[_rows] = _result["position_error"]
    _todo12_angle_error[_rows] = _result["angle_error"]
    _todo12_distance_error[_rows] = _result["distance_error"]
    _todo12_p1_special[_rows] = _result["p1_special"]
    _todo12_p2_special[_rows] = _result["p2_special"]
    _todo12_all_targets.append(_result["target"])
    _todo12_all_predictions.append(_result["prediction"])
    _todo12_run_tp += _result["run_tp"]
    _todo12_run_fp += _result["run_fp"]
    _todo12_run_fn += _result["run_fn"]
    _todo12_onset_delays.extend(_result["onset_delays"])
    _todo12_exit_delays.extend(_result["exit_delays"])

_todo12_all_targets = np.concatenate(_todo12_all_targets)
_todo12_all_predictions = np.concatenate(_todo12_all_predictions)
_todo12_tp, _todo12_fp, _todo12_fn = _todo12_binary_counts(
    _todo12_all_targets, _todo12_all_predictions
)
(
    _todo12_precision,
    _todo12_recall,
    _todo12_f1,
) = _todo12_f1_from_counts(_todo12_tp, _todo12_fp, _todo12_fn)
(
    _todo12_run_precision,
    _todo12_run_recall,
    _todo12_run_f1,
) = _todo12_f1_from_counts(
    _todo12_run_tp, _todo12_run_fp, _todo12_run_fn
)
_todo12_median_absolute_onset_delay = float(
    np.median(np.abs(_todo12_onset_delays))
) if _todo12_onset_delays else np.inf
_todo12_median_absolute_exit_delay = float(
    np.median(np.abs(_todo12_exit_delays))
) if _todo12_exit_delays else np.inf

todo12_one_step_summary = pd.Series(
    {
        "TODO10 exact complete-state share": float(
            _todo10_sequential_exact.mean()
        ),
        "TODO11 exact complete-state share": float(
            _todo11_sequential_exact.mean()
        ),
        "TODO12 exact complete-state share": float(_todo12_exact.mean()),
        "TODO10 material complete-state share": float(
            _todo10_sequential_material.mean()
        ),
        "TODO11 material complete-state share": float(
            _todo11_sequential_material.mean()
        ),
        "TODO12 material complete-state share": float(
            _todo12_material.mean()
        ),
        "TODO10 material failures": int(
            (~_todo10_sequential_material).sum()
        ),
        "TODO11 material failures": int(
            (~_todo11_sequential_material).sum()
        ),
        "TODO12 material failures": int((~_todo12_material).sum()),
        "TODO11 residual F1": float(_todo11_event_f1),
        "TODO12 residual precision": _todo12_precision,
        "TODO12 residual recall": _todo12_recall,
        "TODO12 residual F1": _todo12_f1,
        "TODO12 residual-run F1": _todo12_run_f1,
        "TODO12 median absolute onset delay": (
            _todo12_median_absolute_onset_delay
        ),
        "TODO12 median absolute exit delay": (
            _todo12_median_absolute_exit_delay
        ),
        "TODO11 maximum-angle MAE": float(
            _todo11_maximum_angle_error.mean()
        ),
        "TODO12 maximum-angle MAE": float(_todo12_angle_error.mean()),
    },
    name="value",
)

_todo12_segment_rows = []
for _segment_id in _todo11_segments:
    _mask = _todo9_transition_segments == _segment_id
    _todo12_segment_rows.append(
        {
            "segment_id": int(_segment_id),
            "transitions": int(_mask.sum()),
            "entry": todo12_threshold_by_segment[int(_segment_id)][0],
            "continuation": todo12_threshold_by_segment[int(_segment_id)][1],
            "TODO10_exact": float(_todo10_sequential_exact[_mask].mean()),
            "TODO11_exact": float(_todo11_sequential_exact[_mask].mean()),
            "TODO12_exact": float(_todo12_exact[_mask].mean()),
            "TODO11_material": float(
                _todo11_sequential_material[_mask].mean()
            ),
            "TODO12_material": float(_todo12_material[_mask].mean()),
        }
    )
todo12_one_step_per_segment = pd.DataFrame(
    _todo12_segment_rows
).set_index("segment_id")

_todo12_event_predictions = np.zeros(len(todo9_event_dataset), dtype=bool)
_todo12_event_predictions[_todo9_p1_event_indices] = _todo12_p1_special
_todo12_event_predictions[_todo9_p2_event_indices] = _todo12_p2_special

_todo12_phase_rows = []
for _phase_name, _phase_mask in (
    ("entry", _todo12_entry_target),
    ("continuation", _todo12_continuation_target),
    ("exit", _todo12_exit_target),
):
    if _phase_name == "exit":
        _score = float(
            (~_todo12_event_predictions[_phase_mask]).mean()
        )
        _metric = "correct exit share"
    else:
        _score = float(_todo12_event_predictions[_phase_mask].mean())
        _metric = "recall"
    _todo12_phase_rows.append(
        {
            "phase": _phase_name,
            "events": int(_phase_mask.sum()),
            "metric": _metric,
            "value": _score,
        }
    )
todo12_phase_summary = pd.DataFrame(_todo12_phase_rows).set_index("phase")

TODO12_IMPROVED_SEGMENTS = int(
    (
        todo12_one_step_per_segment["TODO12_exact"]
        > todo12_one_step_per_segment["TODO10_exact"]
    ).sum()
)
TODO12_FAILURE_GATE_PASSED = bool(
    int((~_todo12_material).sum()) <= TODO12_MAXIMUM_MATERIAL_FAILURES
)
TODO12_F1_GATE_PASSED = bool(_todo12_f1 >= TODO12_SPECIAL_F1_GATE)
TODO12_RUN_GATE_PASSED = bool(
    _todo12_run_f1 >= TODO12_RUN_F1_GATE
    and _todo12_median_absolute_onset_delay
    <= TODO12_MAXIMUM_MEDIAN_RUN_DELAY
    and _todo12_median_absolute_exit_delay
    <= TODO12_MAXIMUM_MEDIAN_RUN_DELAY
)
TODO12_GLOBAL_GATE_PASSED = bool(
    _todo12_exact.mean() > _todo11_sequential_exact.mean()
    and _todo12_material.mean() > _todo11_sequential_material.mean()
)
TODO12_SEGMENT_GATE_PASSED = bool(
    TODO12_IMPROVED_SEGMENTS >= TODO12_MINIMUM_IMPROVED_SEGMENTS
)
TODO12_ONE_STEP_GATE_PASSED = bool(
    TODO12_FAILURE_GATE_PASSED
    and TODO12_F1_GATE_PASSED
    and TODO12_RUN_GATE_PASSED
    and TODO12_GLOBAL_GATE_PASSED
    and TODO12_SEGMENT_GATE_PASSED
)


# ---------------------------------------------------------------------------
# 12.C — failure taxonomy
# ---------------------------------------------------------------------------
_todo12_failure_rows = []
for _transition_row in np.flatnonzero(~_todo12_material):
    _p1_event = _todo9_p1_event_indices[_transition_row]
    _p2_event = _todo9_p2_event_indices[_transition_row]
    _causes = []
    for _event_index, _predicted in (
        (_p1_event, _todo12_p1_special[_transition_row]),
        (_p2_event, _todo12_p2_special[_transition_row]),
    ):
        if _todo12_entry_target[_event_index] and not _predicted:
            _causes.append("missed entry")
        elif _todo12_continuation_target[_event_index] and not _predicted:
            _causes.append("missed continuation")
        elif not _todo12_residual_target[_event_index] and _predicted:
            _causes.append("false special / late exit")
        if todo9_event_dataset.at[_event_index, "event_mode"] == "segment_start_special":
            _causes.append("segment-start hidden state")
    if not _causes:
        _causes.append("period-2 entry or upstream mixed error")
    _todo12_failure_rows.append(
        {
            "transition_row": int(_transition_row),
            "segment_id": int(_todo9_transition_segments[_transition_row]),
            "source_row": int(_todo9_transition_source_rows[_transition_row]),
            "cause": " + ".join(sorted(set(_causes))),
            "maximum_position_error": float(
                _todo12_position_error[_transition_row]
            ),
            "maximum_angle_error": float(
                _todo12_angle_error[_transition_row]
            ),
        }
    )

todo12_material_failures = pd.DataFrame(
    _todo12_failure_rows,
    columns=[
        "transition_row",
        "segment_id",
        "source_row",
        "cause",
        "maximum_position_error",
        "maximum_angle_error",
    ],
)
todo12_material_failure_summary = (
    todo12_material_failures.groupby("cause", as_index=True)
    .agg(
        transitions=("source_row", "size"),
        segments=("segment_id", "nunique"),
        maximum_angle_error=("maximum_angle_error", "max"),
        maximum_position_error=("maximum_position_error", "max"),
    )
    .sort_values("transitions", ascending=False)
)


# ---------------------------------------------------------------------------
# 12.D — recursive simulator, opened only by the complete one-step gate
# ---------------------------------------------------------------------------
def _todo12_recursive_object_update(
    object_name,
    segment_id,
    previous_position,
    previous_angle,
    driver_distance,
    angle_sign,
    history,
    models,
    entry_threshold,
    continuation_threshold,
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
        )
        if _active:
            _model = models["continuation"]
            _threshold = continuation_threshold
            _eligible = True
        else:
            _model = models["entry"]
            _threshold = entry_threshold
            _eligible = _candidate
        if _eligible:
            _probability = _todo12_vector_probability(
                _model, _todo10_feature_vector(_features), _active
            )
        _special = bool(_eligible and _probability >= _threshold)
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


def _todo12_rollout_segment(segment_frame, segment_id):
    _segment_frame = segment_frame.sort_index()
    _truth_p1 = _segment_frame[["x1", "y1"]].to_numpy(dtype=float)
    _truth_p2 = _segment_frame[["x2", "y2"]].to_numpy(dtype=float)
    _truth_a1 = _segment_frame["angle1"].to_numpy(dtype=float)
    _truth_a2 = _segment_frame["angle2"].to_numpy(dtype=float)
    _truth_distance = _segment_frame["distance"].to_numpy(dtype=float)
    _models = todo12_outer_models[int(segment_id)]
    _entry, _continuation = todo12_threshold_by_segment[int(segment_id)]
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


todo12_rollouts = {}
_todo12_rollout_rows = []
_todo12_horizon_rows = []
if TODO12_ONE_STEP_GATE_PASSED:
    for _segment_id, _segment_frame in todo7_df.groupby(
        "segment_id", sort=True
    ):
        _rollout = _todo12_rollout_segment(_segment_frame, _segment_id)
        todo12_rollouts[int(_segment_id)] = _rollout
        _transition_count = len(_segment_frame) - 1
        _first_material = _rollout["first_material_failure"]
        _first_exact = _rollout["first_numerical_failure"]
        _todo12_rollout_rows.append(
            {
                "segment_id": int(_segment_id),
                "transitions": _transition_count,
                "first numerical failure": (
                    _transition_count + 1
                    if _first_exact is None
                    else _first_exact
                ),
                "first material failure": (
                    _transition_count + 1
                    if _first_material is None
                    else _first_material
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
            _automaton = _rollout["predicted_positions"][1 : _horizon + 1]
            _todo10_positions = todo10_rollouts[int(_segment_id)][
                "predicted_positions"
            ][1 : _horizon + 1]
            _todo12_horizon_rows.append(
                {
                    "segment_id": int(_segment_id),
                    "horizon": _label,
                    "TODO12 RMSE": float(
                        np.sqrt(np.mean((_automaton - _truth) ** 2))
                    ),
                    "TODO10 RMSE": float(
                        np.sqrt(np.mean((_todo10_positions - _truth) ** 2))
                    ),
                }
            )

todo12_rollout_summary = pd.DataFrame(_todo12_rollout_rows)
if len(todo12_rollout_summary):
    todo12_rollout_summary = todo12_rollout_summary.set_index("segment_id")
todo12_rollout_horizons = pd.DataFrame(_todo12_horizon_rows)
todo12_rollout_macro = (
    todo12_rollout_horizons.groupby("horizon", sort=False)
    .agg(
        segments=("segment_id", "nunique"),
        TODO12_RMSE=("TODO12 RMSE", "mean"),
        TODO10_RMSE=("TODO10 RMSE", "mean"),
    )
    if len(todo12_rollout_horizons)
    else pd.DataFrame()
)

TODO12_RECURSIVE_ROLLOUT_RUN = bool(TODO12_ONE_STEP_GATE_PASSED)
TODO12_MEDIAN_MATERIAL_PREFIX = (
    float(todo12_rollout_summary["first material failure"].median())
    if TODO12_RECURSIVE_ROLLOUT_RUN
    else np.nan
)
TODO12_ROLLOUT_GATE_PASSED = bool(
    TODO12_RECURSIVE_ROLLOUT_RUN
    and TODO12_MEDIAN_MATERIAL_PREFIX >= TODO12_RECURSIVE_MEDIAN_GATE
    and todo12_rollout_macro.loc["100", "TODO12_RMSE"]
    < todo12_rollout_macro.loc["100", "TODO10_RMSE"]
    and todo12_rollout_macro.loc["500", "TODO12_RMSE"]
    < todo12_rollout_macro.loc["500", "TODO10_RMSE"]
    and todo12_rollout_macro.loc["full", "TODO12_RMSE"]
    <= todo12_rollout_macro.loc["full", "TODO10_RMSE"]
)

TODO12_STATUS = (
    "entry_continuation_automaton_all_gates_passed"
    if TODO12_ROLLOUT_GATE_PASSED
    else "entry_continuation_one_step_passed_rollout_gate_failed"
    if TODO12_RECURSIVE_ROLLOUT_RUN
    else "entry_continuation_one_step_gate_failed_rollout_closed"
)

todo12_manifest = pd.Series(
    {
        "status": TODO12_STATUS,
        "pairwise inner models fitted": _todo12_pair_models_fitted,
        "outer models fitted": 2 * len(_todo11_segments),
        "inner objective uses full sequential state": True,
        "thresholds selected with outer labels": False,
        "automaton state uses current/previous target at inference": False,
        "exact special-angle formula changed": False,
        "trajectory ensemble used": False,
        "material failure gate passed": TODO12_FAILURE_GATE_PASSED,
        "residual F1 gate passed": TODO12_F1_GATE_PASSED,
        "residual-run F1": _todo12_run_f1,
        "residual-run F1 gate": TODO12_RUN_F1_GATE,
        "run/delay gate passed": TODO12_RUN_GATE_PASSED,
        "global TODO11 improvement gate passed": TODO12_GLOBAL_GATE_PASSED,
        "segments improved versus TODO10": TODO12_IMPROVED_SEGMENTS,
        "segment gate passed": TODO12_SEGMENT_GATE_PASSED,
        "one-step gate passed": TODO12_ONE_STEP_GATE_PASSED,
        "recursive rollout run": TODO12_RECURSIVE_ROLLOUT_RUN,
        "median first material failure": TODO12_MEDIAN_MATERIAL_PREFIX,
        "rollout gate passed": TODO12_ROLLOUT_GATE_PASSED,
        "pristine test": False,
    },
    name="value",
)

assert set(todo12_threshold_by_segment) == set(_todo11_segments)
assert np.isfinite(_todo12_position_error).all()
assert np.isfinite(_todo12_angle_error).all()
assert np.isfinite(_todo12_distance_error).all()
assert not TODO12_RECURSIVE_ROLLOUT_RUN or TODO12_ONE_STEP_GATE_PASSED
if TODO12_RECURSIVE_ROLLOUT_RUN:
    assert todo12_rollout_summary["finite"].all()

todo12_label_summary = pd.Series(
    {
        "entry targets": int(_todo12_entry_target.sum()),
        "continuation targets": int(_todo12_continuation_target.sum()),
        "exit targets": int(_todo12_exit_target.sum()),
        "entry targets outside causal candidate": int(
            (_todo12_entry_target & ~_todo11_base_candidate).sum()
        ),
        "entry training rows": int(_todo12_entry_population.sum()),
        "continuation training rows": int(
            _todo12_continuation_population.sum()
        ),
    },
    name="events",
)

display(
    Markdown("#### Automaton target populations"),
    todo12_label_summary.to_frame(),
    Markdown("#### Nested full-sequential threshold selection"),
    todo12_threshold_selection.style.format(precision=6),
    Markdown("#### Outer-LOSO full sequential one-step"),
    todo12_one_step_summary.to_frame(),
    todo12_one_step_per_segment.style.format(precision=6),
    Markdown("#### Entry / continuation / exit diagnostics"),
    todo12_phase_summary.style.format(precision=6),
    Markdown("#### Remaining material failures"),
    todo12_material_failure_summary.style.format(precision=6),
    Markdown("#### Recursive rollout"),
    todo12_rollout_summary.style.format(precision=6)
    if len(todo12_rollout_summary)
    else todo12_rollout_summary,
    todo12_rollout_macro.style.format(precision=6)
    if len(todo12_rollout_macro)
    else todo12_rollout_macro,
    Markdown("#### TODO 12 manifest"),
    todo12_manifest.to_frame(),
)
