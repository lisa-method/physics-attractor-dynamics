"""TODO 15 companion: history-aware NORMAL -> SPECIAL entry selector.

The deterministic geometry, PERIOD2 priority, and TODO 11 continuation policy
stay frozen.  Only the entry classifier receives a short causal history.  The
history length and entry threshold are selected by full sequential nested LOSO.
"""

import gc
from itertools import combinations


TODO15_HISTORY_LENGTHS = (2, 4, 8)
TODO15_ENTRY_THRESHOLDS = tuple(np.round(np.arange(0.35, 0.86, 0.05), 2))
TODO15_MAXIMUM_MATERIAL_FAILURES = 40
TODO15_MINIMUM_IMPROVED_SEGMENTS = 8
TODO15_RANDOM_STATE = TODO9_RANDOM_STATE


# ---------------------------------------------------------------------------
# 15.A — compact causal history, never crossing an object/segment boundary
# ---------------------------------------------------------------------------
_todo15_history_names = (
    "driver_distance",
    "previous_angle_sin",
    "previous_angle_cos",
    "free_angle_sin",
    "free_angle_cos",
    "step_length",
    "free_x",
    "free_y",
    "x_penetration",
    "y_penetration",
    "x_hit_fraction",
    "y_hit_fraction",
    "x_hit_fraction_finite",
    "y_hit_fraction_finite",
    "previous_x_gap",
    "previous_y_gap",
    "predicted_x_crossing",
    "predicted_y_crossing",
    "both_axis_proposal",
    "previous_wall_contact",
    "previous_boundary_run_length",
    "previous_boundary_run_parity",
    "previous_boundary_run_capped8",
    "previous_special",
    "previous_period2_position_error",
    "previous_period2_angle_error",
    "previous_period2_exact",
)

_todo15_free_angle_radians = np.deg2rad(
    todo9_event_dataset["free_angle"].to_numpy(dtype=float)
)
_todo15_previous_angle_radians = np.deg2rad(
    todo9_event_dataset["previous_angle"].to_numpy(dtype=float)
)
_todo15_x_hit = todo9_event_dataset["x_hit_fraction"].to_numpy(dtype=float)
_todo15_y_hit = todo9_event_dataset["y_hit_fraction"].to_numpy(dtype=float)
_todo15_history_signals = np.column_stack(
    [
        todo9_event_dataset["driver_distance"].to_numpy(dtype=float),
        np.sin(_todo15_previous_angle_radians),
        np.cos(_todo15_previous_angle_radians),
        np.sin(_todo15_free_angle_radians),
        np.cos(_todo15_free_angle_radians),
        todo9_event_dataset["step_length"].to_numpy(dtype=float),
        todo9_event_dataset["free_x"].to_numpy(dtype=float),
        todo9_event_dataset["free_y"].to_numpy(dtype=float),
        todo9_event_dataset["x_penetration"].to_numpy(dtype=float),
        todo9_event_dataset["y_penetration"].to_numpy(dtype=float),
        np.where(np.isfinite(_todo15_x_hit), _todo15_x_hit, -1.0),
        np.where(np.isfinite(_todo15_y_hit), _todo15_y_hit, -1.0),
        np.isfinite(_todo15_x_hit).astype(float),
        np.isfinite(_todo15_y_hit).astype(float),
        todo9_event_dataset["previous_x_gap"].to_numpy(dtype=float),
        todo9_event_dataset["previous_y_gap"].to_numpy(dtype=float),
        todo9_event_dataset["predicted_x_crossing"].to_numpy(dtype=float),
        todo9_event_dataset["predicted_y_crossing"].to_numpy(dtype=float),
        todo9_event_dataset["both_axis_proposal"].to_numpy(dtype=float),
        todo9_event_dataset["previous_wall_contact"].to_numpy(dtype=float),
        todo9_event_dataset["previous_boundary_run_length"].to_numpy(
            dtype=float
        ),
        np.mod(
            todo9_event_dataset["previous_boundary_run_length"].to_numpy(
                dtype=float
            ),
            2.0,
        ),
        np.minimum(
            todo9_event_dataset["previous_boundary_run_length"].to_numpy(
                dtype=float
            ),
            8.0,
        ),
        todo9_event_dataset["previous_special"].to_numpy(dtype=float),
        todo9_event_dataset["previous_period2_position_error"].to_numpy(
            dtype=float
        ),
        todo9_event_dataset["previous_period2_angle_error"].to_numpy(
            dtype=float
        ),
        todo9_event_dataset["previous_period2_exact"].to_numpy(dtype=float),
    ]
).astype(np.float32)
_todo15_history_signals = np.nan_to_num(
    _todo15_history_signals, nan=-1.0, posinf=-1.0, neginf=-1.0
)

_todo15_history_width = len(_todo15_history_names) + 1
_todo15_max_history = max(TODO15_HISTORY_LENGTHS)
_todo15_history_block = np.zeros(
    (
        len(todo9_event_dataset),
        _todo15_max_history * _todo15_history_width,
    ),
    dtype=np.float32,
)
_todo15_history_feature_names = []
assert not todo9_event_dataset.duplicated(
    ["object", "segment_id", "segment_step"]
).any()
for _lag in range(1, _todo15_max_history + 1):
    _start = (_lag - 1) * _todo15_history_width
    for _indices in _todo11_sequence_indices:
        if len(_indices) <= _lag:
            continue
        _current = _indices[_lag:]
        _previous = _indices[:-_lag]
        _todo15_history_block[
            _current, _start : _start + len(_todo15_history_names)
        ] = _todo15_history_signals[_previous]
        _todo15_history_block[
            _current, _start + len(_todo15_history_names)
        ] = 1.0
    _todo15_history_feature_names.extend(
        [f"lag{_lag}__{_name}" for _name in _todo15_history_names]
        + [f"lag{_lag}__valid"]
    )

_todo15_current_matrix = _todo11_all_X.to_numpy(dtype=np.float32)
_todo15_previous_special_column = TODO9_FEATURE_COLUMNS.index(
    "previous_special"
)


def _todo15_entry_design(history_length, current_vectors, event_indices):
    """Combine a current causal proposal with frozen earlier-row features."""
    _current = np.asarray(current_vectors, dtype=np.float32).copy()
    if _current.ndim == 1:
        _current = _current.reshape(1, -1)
    # The entry head is queried only while the predicted automaton is NORMAL.
    # Do not let the observed previous-special flag contradict that state.
    _current[:, _todo15_previous_special_column] = 0.0
    _event_indices = np.asarray(event_indices, dtype=int)
    _history_columns = history_length * _todo15_history_width
    return np.concatenate(
        (
            _current,
            _todo15_history_block[_event_indices, :_history_columns],
        ),
        axis=1,
    )


_todo15_design_by_k = {
    _history_length: _todo15_entry_design(
        _history_length,
        _todo15_current_matrix,
        np.arange(len(todo9_event_dataset), dtype=int),
    )
    for _history_length in TODO15_HISTORY_LENGTHS
}
_todo15_entry_target = np.asarray(_todo12_residual_target, dtype=bool)
_todo15_entry_population = np.asarray(
    _todo12_entry_population, dtype=bool
)
_todo15_event_segments = todo9_event_dataset["segment_id"].to_numpy(
    dtype=int
)

for _indices in _todo11_sequence_indices:
    _sequence = todo9_event_dataset.loc[
        _indices, ["object", "segment_id", "segment_step"]
    ]
    assert _sequence["object"].nunique() == 1
    assert _sequence["segment_id"].nunique() == 1
    assert np.all(np.diff(_sequence["segment_step"].to_numpy()) == 1)
    _first = int(_indices[0])
    for _lag in range(1, _todo15_max_history + 1):
        _start = (_lag - 1) * _todo15_history_width
        assert _todo15_history_block[
            _first, _start + len(_todo15_history_names)
        ] == 0.0


def _todo15_fit_entry_model(excluded_segments, history_length):
    _excluded = np.asarray(tuple(excluded_segments), dtype=int)
    _train = (
        _todo15_entry_population
        & ~np.isin(_todo15_event_segments, _excluded)
    )
    _target = _todo15_entry_target[_train]
    assert set(np.unique(_target)) == {False, True}
    _model = RandomForestClassifier(
        n_estimators=TODO9_RF_ESTIMATORS,
        max_depth=TODO9_RF_MAX_DEPTH,
        min_samples_leaf=TODO9_RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=TODO15_RANDOM_STATE,
        n_jobs=-1,
    )
    _model.fit(_todo15_design_by_k[history_length][_train], _target)
    return _model


def _todo15_feature_names(history_length):
    _history_columns = history_length * _todo15_history_width
    return tuple(TODO9_FEATURE_COLUMNS) + tuple(
        _todo15_history_feature_names[:_history_columns]
    )


# ---------------------------------------------------------------------------
# 15.B — frozen two-scenario sequential geometry and dynamic object-2 inputs
# ---------------------------------------------------------------------------
todo15_geometry_caches = {}
for _segment_id in _todo11_segments:
    _cache = dict(todo13_outer_caches[int(_segment_id)])
    _count = len(_cache["transition_rows"])
    _p2_vectors = np.zeros(
        (_count, 2, len(TODO9_FEATURE_COLUMNS)), dtype=np.float32
    )
    for _local, (_transition_row, _event_index) in enumerate(
        zip(_cache["transition_rows"], _cache["p2_events"])
    ):
        if _cache["p2_loop"][_local]:
            _p2_vectors[_local, :, :] = _todo15_current_matrix[_event_index]
            continue
        _data_index = _todo9_transition_indices[_transition_row]
        for _scenario in (0, 1):
            _driver_distance = float(
                _cache["p2_distance"][_local, _scenario]
            )
            _baseline = _todo8_update_object(
                todo7_p2[_data_index - 1],
                todo7_angle2[_data_index - 1],
                _driver_distance,
                -1.0,
            )
            assert np.allclose(
                _baseline["position"],
                _cache["p2_base_position"][_local, _scenario],
                atol=1e-12,
                rtol=0.0,
            )
            _record = _todo9_feature_record(
                _event_index, _baseline, _driver_distance
            )
            _p2_vectors[_local, _scenario] = _todo10_feature_vector(_record)
    _cache["todo15_p2_vectors"] = _p2_vectors
    todo15_geometry_caches[int(_segment_id)] = _cache


def _todo15_add_continuation_probabilities(cache, model):
    """Use the frozen TODO 11 general RF only while already active."""
    _result = dict(cache)
    _p1_frame = _todo11_all_X.iloc[cache["p1_events"]]
    _result["p1_continuation_probability"] = _todo11_true_probability(
        model, _p1_frame
    )
    _flat = cache["todo15_p2_vectors"].reshape(
        -1, len(TODO9_FEATURE_COLUMNS)
    )
    _flat_frame = pd.DataFrame(_flat, columns=TODO9_FEATURE_COLUMNS)
    _result["p2_continuation_probability"] = (
        _todo11_true_probability(model, _flat_frame).reshape(-1, 2)
    )
    return _result


def _todo15_add_entry_probabilities(cache, model, history_length):
    """Add history-entry probabilities for both possible object-1 branches."""
    _result = dict(cache)
    _p1_events = np.asarray(cache["p1_events"], dtype=int)
    _result["p1_entry_probability"] = _todo11_true_probability(
        model, _todo15_design_by_k[history_length][_p1_events]
    )
    _p2_events = np.repeat(np.asarray(cache["p2_events"], dtype=int), 2)
    _flat = cache["todo15_p2_vectors"].reshape(
        -1, len(TODO9_FEATURE_COLUMNS)
    )
    _p2_design = _todo15_entry_design(
        history_length, _flat, _p2_events
    )
    _result["p2_entry_probability"] = _todo11_true_probability(
        model, _p2_design
    ).reshape(-1, 2)
    return _result


# ---------------------------------------------------------------------------
# 15.C — full-sequential pairwise nested LOSO selection of K and T_entry
# ---------------------------------------------------------------------------
_todo15_accumulators = {
    (int(_outer), int(_history_length), float(_threshold)): {
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
    for _history_length in TODO15_HISTORY_LENGTHS
    for _threshold in TODO15_ENTRY_THRESHOLDS
}
_todo15_general_pair_models_fitted = 0
_todo15_history_pair_models_fitted = 0

for _first, _second in combinations(_todo11_segments, 2):
    _general_model = _todo13_fit_general_model((_first, _second))
    _todo15_general_pair_models_fitted += 1
    _history_models = {
        _history_length: _todo15_fit_entry_model(
            (_first, _second), _history_length
        )
        for _history_length in TODO15_HISTORY_LENGTHS
    }
    _todo15_history_pair_models_fitted += len(TODO15_HISTORY_LENGTHS)

    for _outer, _inner in ((_first, _second), (_second, _first)):
        _common_cache = _todo15_add_continuation_probabilities(
            todo15_geometry_caches[int(_inner)], _general_model
        )
        _stay_threshold = todo11_threshold_by_segment[int(_outer)][1]
        for _history_length, _history_model in _history_models.items():
            _cache = _todo15_add_entry_probabilities(
                _common_cache, _history_model, _history_length
            )
            for _entry_threshold in TODO15_ENTRY_THRESHOLDS:
                _result = _todo12_simulate_fold(
                    _cache, _entry_threshold, _stay_threshold
                )
                _accumulator = _todo15_accumulators[
                    (
                        int(_outer),
                        int(_history_length),
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

    del _history_models, _general_model
    gc.collect()

assert _todo15_general_pair_models_fitted == 55
assert _todo15_history_pair_models_fitted == 165

_todo15_candidate_rows = []
_todo15_best_by_segment_and_k = {}
todo15_selected_history_by_segment = {}
todo15_entry_threshold_by_segment = {}

for _outer in _todo11_segments:
    _outer_best = None
    for _history_length in TODO15_HISTORY_LENGTHS:
        _history_best = None
        for _entry_threshold in TODO15_ENTRY_THRESHOLDS:
            _accumulator = _todo15_accumulators[
                (int(_outer), int(_history_length), float(_entry_threshold))
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
                "history_length": int(_history_length),
                "entry": float(_entry_threshold),
                "stay": float(todo11_threshold_by_segment[int(_outer)][1]),
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
                _candidate["history_length"],
                -_candidate["entry"],
            )
            if _history_best is None or _key < _history_best[0]:
                _history_best = (_key, _candidate)
        _chosen_for_k = _history_best[1]
        _todo15_best_by_segment_and_k[
            (int(_outer), int(_history_length))
        ] = _chosen_for_k
        _todo15_candidate_rows.append(_chosen_for_k)
        if _outer_best is None or _history_best[0] < _outer_best[0]:
            _outer_best = _history_best

    _chosen = _outer_best[1]
    todo15_selected_history_by_segment[int(_outer)] = int(
        _chosen["history_length"]
    )
    todo15_entry_threshold_by_segment[int(_outer)] = float(
        _chosen["entry"]
    )

todo15_nested_selection = pd.DataFrame(_todo15_candidate_rows)
todo15_nested_selection["selected_for_outer"] = [
    todo15_selected_history_by_segment[int(_row.segment_id)]
    == int(_row.history_length)
    for _row in todo15_nested_selection.itertuples(index=False)
]
todo15_nested_selection = todo15_nested_selection.set_index(
    ["segment_id", "history_length"]
)

del _todo15_accumulators
gc.collect()


# ---------------------------------------------------------------------------
# 15.D — outer-LOSO sequential evaluation for every K and nested choice
# ---------------------------------------------------------------------------
_todo15_transition_count = len(_todo9_transition_indices)
_todo15_outer_models_fitted = 0
_todo15_selected_importance_rows = []
todo15_results_by_k = {}


def _todo15_empty_result_arrays():
    return {
        "exact": np.zeros(_todo15_transition_count, dtype=bool),
        "material": np.zeros(_todo15_transition_count, dtype=bool),
        "position_error": np.empty(_todo15_transition_count, dtype=float),
        "angle_error": np.empty(_todo15_transition_count, dtype=float),
        "distance_error": np.empty(_todo15_transition_count, dtype=float),
        "p1_special": np.zeros(_todo15_transition_count, dtype=bool),
        "p2_special": np.zeros(_todo15_transition_count, dtype=bool),
        "p1_prediction": np.empty((_todo15_transition_count, 2), dtype=float),
        "p2_prediction": np.empty((_todo15_transition_count, 2), dtype=float),
        "a1_prediction": np.empty(_todo15_transition_count, dtype=float),
        "a2_prediction": np.empty(_todo15_transition_count, dtype=float),
        "distance_prediction": np.empty(_todo15_transition_count, dtype=float),
        "run_tp": 0,
        "run_fp": 0,
        "run_fn": 0,
        "onset_delays": [],
        "exit_delays": [],
    }


for _history_length in TODO15_HISTORY_LENGTHS:
    _arrays = _todo15_empty_result_arrays()
    for _segment_id in _todo11_segments:
        _segment_id = int(_segment_id)
        _model = _todo15_fit_entry_model((_segment_id,), _history_length)
        _todo15_outer_models_fitted += 1
        _cache = _todo15_add_continuation_probabilities(
            todo15_geometry_caches[_segment_id],
            todo9_residual_models[_segment_id],
        )
        _cache = _todo15_add_entry_probabilities(
            _cache, _model, _history_length
        )
        _selection = _todo15_best_by_segment_and_k[
            (_segment_id, int(_history_length))
        ]
        _result = _todo12_simulate_fold(
            _cache, _selection["entry"], _selection["stay"]
        )
        _rows = _cache["transition_rows"]
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

        if (
            todo15_selected_history_by_segment[_segment_id]
            == _history_length
        ):
            for _feature, _importance in zip(
                _todo15_feature_names(_history_length),
                _model.feature_importances_,
            ):
                _todo15_selected_importance_rows.append(
                    {
                        "segment_id": _segment_id,
                        "feature": _feature,
                        "importance": float(_importance),
                    }
                )
        del _model, _cache
        gc.collect()
    todo15_results_by_k[int(_history_length)] = _arrays

assert _todo15_outer_models_fitted == 33


def _todo15_collect_event_scores(p1_special, p2_special):
    _target_parts = []
    _prediction_parts = []
    for _segment_id in _todo11_segments:
        _rows = np.flatnonzero(_todo9_transition_segments == _segment_id)
        _p1_events = _todo9_p1_event_indices[_rows]
        _p2_events = _todo9_p2_event_indices[_rows]
        _eligible = np.concatenate(
            (
                ~_todo12_hidden_initialization[_p1_events],
                ~_todo12_hidden_initialization[_p2_events],
            )
        )
        _target_parts.append(
            np.concatenate(
                (
                    _todo12_residual_target[_p1_events],
                    _todo12_residual_target[_p2_events],
                )
            )[_eligible]
        )
        _prediction_parts.append(
            np.concatenate((p1_special[_rows], p2_special[_rows]))[
                _eligible
            ]
        )
    return np.concatenate(_target_parts), np.concatenate(_prediction_parts)


_todo15_baseline_target, _todo15_baseline_prediction = (
    _todo15_collect_event_scores(_todo11_p1_special, _todo11_p2_special)
)
_todo15_baseline_counts = _todo12_binary_counts(
    _todo15_baseline_target, _todo15_baseline_prediction
)
_, _, _todo15_todo11_sequential_f1 = _todo12_f1_from_counts(
    *_todo15_baseline_counts
)


def _todo15_summarize_result(label, arrays):
    _target, _prediction = _todo15_collect_event_scores(
        arrays["p1_special"], arrays["p2_special"]
    )
    _tp, _fp, _fn = _todo12_binary_counts(_target, _prediction)
    _precision, _recall, _f1 = _todo12_f1_from_counts(_tp, _fp, _fn)
    _run_precision, _run_recall, _run_f1 = _todo12_f1_from_counts(
        arrays["run_tp"], arrays["run_fp"], arrays["run_fn"]
    )
    return {
        "model": label,
        "exact_share": float(arrays["exact"].mean()),
        "material_share": float(arrays["material"].mean()),
        "material_failures": int((~arrays["material"]).sum()),
        "residual_precision": float(_precision),
        "residual_recall": float(_recall),
        "residual_f1": float(_f1),
        "run_f1": float(_run_f1),
        "median_absolute_onset_delay": (
            float(np.median(np.abs(arrays["onset_delays"])))
            if arrays["onset_delays"]
            else np.nan
        ),
        "median_absolute_exit_delay": (
            float(np.median(np.abs(arrays["exit_delays"])))
            if arrays["exit_delays"]
            else np.nan
        ),
    }


_todo15_summary_rows = [
    {
        "model": "TODO11",
        "exact_share": float(_todo11_sequential_exact.mean()),
        "material_share": float(_todo11_sequential_material.mean()),
        "material_failures": int((~_todo11_sequential_material).sum()),
        "residual_precision": float(
            _todo15_baseline_counts[0]
            / max(1, _todo15_baseline_counts[0] + _todo15_baseline_counts[1])
        ),
        "residual_recall": float(
            _todo15_baseline_counts[0]
            / max(1, _todo15_baseline_counts[0] + _todo15_baseline_counts[2])
        ),
        "residual_f1": float(_todo15_todo11_sequential_f1),
        "run_f1": np.nan,
        "median_absolute_onset_delay": np.nan,
        "median_absolute_exit_delay": np.nan,
    }
]
for _history_length in TODO15_HISTORY_LENGTHS:
    _todo15_summary_rows.append(
        _todo15_summarize_result(
            f"history K={_history_length}",
            todo15_results_by_k[int(_history_length)],
        )
    )

_todo15_adaptive = _todo15_empty_result_arrays()
for _segment_id in _todo11_segments:
    _segment_id = int(_segment_id)
    _history_length = todo15_selected_history_by_segment[_segment_id]
    _source = todo15_results_by_k[_history_length]
    _rows = np.flatnonzero(_todo9_transition_segments == _segment_id)
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
        _todo15_adaptive[_name][_rows] = _source[_name][_rows]

# Recompute run metrics for the per-segment nested history choice.
for _segment_id in _todo11_segments:
    _rows = np.flatnonzero(_todo9_transition_segments == _segment_id)
    for _events, _prediction in (
        (
            _todo9_p1_event_indices[_rows],
            _todo15_adaptive["p1_special"][_rows],
        ),
        (
            _todo9_p2_event_indices[_rows],
            _todo15_adaptive["p2_special"][_rows],
        ),
    ):
        _target = _todo12_residual_target[_events].copy()
        _prediction = _prediction.copy()
        _hidden = _todo12_hidden_initialization[_events]
        _target[_hidden] = False
        _prediction[_hidden] = False
        _run = _todo12_run_metrics(_target, _prediction)
        for _name in ("tp", "fp", "fn"):
            _todo15_adaptive[f"run_{_name}"] += _run[_name]
        _todo15_adaptive["onset_delays"].extend(_run["onset_delays"])
        _todo15_adaptive["exit_delays"].extend(_run["exit_delays"])

_todo15_summary_rows.append(
    _todo15_summarize_result("nested adaptive K", _todo15_adaptive)
)
todo15_model_summary = pd.DataFrame(_todo15_summary_rows).set_index("model")


# ---------------------------------------------------------------------------
# 15.E — segment, phase, failure, importance and frozen gate reports
# ---------------------------------------------------------------------------
_todo15_segment_rows = []
for _segment_id in _todo11_segments:
    _mask = _todo9_transition_segments == _segment_id
    _todo15_segment_rows.append(
        {
            "segment_id": int(_segment_id),
            "history_length": todo15_selected_history_by_segment[
                int(_segment_id)
            ],
            "entry": todo15_entry_threshold_by_segment[int(_segment_id)],
            "stay": todo11_threshold_by_segment[int(_segment_id)][1],
            "TODO11_exact": float(_todo11_sequential_exact[_mask].mean()),
            "TODO15_exact": float(_todo15_adaptive["exact"][_mask].mean()),
            "TODO11_material": float(
                _todo11_sequential_material[_mask].mean()
            ),
            "TODO15_material": float(
                _todo15_adaptive["material"][_mask].mean()
            ),
            "TODO11_failures": int(
                (~_todo11_sequential_material[_mask]).sum()
            ),
            "TODO15_failures": int(
                (~_todo15_adaptive["material"][_mask]).sum()
            ),
        }
    )
todo15_per_segment = pd.DataFrame(_todo15_segment_rows).set_index("segment_id")

_todo15_event_predictions = np.zeros(len(todo9_event_dataset), dtype=bool)
_todo15_event_predictions[_todo9_p1_event_indices] = _todo15_adaptive[
    "p1_special"
]
_todo15_event_predictions[_todo9_p2_event_indices] = _todo15_adaptive[
    "p2_special"
]
_todo15_phase_rows = []
for _phase, _mask in (
    ("entry recall", _todo12_entry_target),
    ("continuation recall", _todo12_continuation_target),
    ("correct exit share", _todo12_exit_target),
):
    _eligible = _mask & ~_todo12_hidden_initialization
    _score = (
        float((~_todo15_event_predictions[_eligible]).mean())
        if _phase == "correct exit share"
        else float(_todo15_event_predictions[_eligible].mean())
    )
    _todo15_phase_rows.append(
        {"phase": _phase, "rows": int(_eligible.sum()), "score": _score}
    )
todo15_phase_summary = pd.DataFrame(_todo15_phase_rows).set_index("phase")

_todo15_failure_rows = []
for _transition_row in np.flatnonzero(~_todo15_adaptive["material"]):
    _causes = []
    for _event_index, _prediction in (
        (
            _todo9_p1_event_indices[_transition_row],
            _todo15_adaptive["p1_special"][_transition_row],
        ),
        (
            _todo9_p2_event_indices[_transition_row],
            _todo15_adaptive["p2_special"][_transition_row],
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
    _todo15_failure_rows.append(
        {
            "segment_id": int(_todo9_transition_segments[_transition_row]),
            "source_row": int(_todo9_transition_source_rows[_transition_row]),
            "cause": " + ".join(sorted(set(_causes))),
            "maximum_position_error": float(
                _todo15_adaptive["position_error"][_transition_row]
            ),
            "maximum_angle_error": float(
                _todo15_adaptive["angle_error"][_transition_row]
            ),
        }
    )
todo15_material_failures = pd.DataFrame(_todo15_failure_rows)
todo15_material_failure_summary = (
    todo15_material_failures.groupby("cause")
    .agg(
        transitions=("source_row", "size"),
        segments=("segment_id", "nunique"),
        maximum_position_error=("maximum_position_error", "max"),
        maximum_angle_error=("maximum_angle_error", "max"),
    )
    .sort_values("transitions", ascending=False)
)

todo15_feature_importance = (
    pd.DataFrame(_todo15_selected_importance_rows)
    .groupby("feature", as_index=False)
    .agg(
        mean_importance=("importance", "mean"),
        folds_present=("segment_id", "nunique"),
    )
    .sort_values("mean_importance", ascending=False)
    .reset_index(drop=True)
)

TODO15_IMPROVED_SEGMENTS = int(
    (todo15_per_segment["TODO15_exact"] > todo15_per_segment["TODO11_exact"])
    .sum()
)
_todo15_adaptive_summary = todo15_model_summary.loc["nested adaptive K"]
TODO15_FAILURE_GATE_PASSED = bool(
    _todo15_adaptive_summary["material_failures"]
    <= TODO15_MAXIMUM_MATERIAL_FAILURES
)
TODO15_F1_GATE_PASSED = bool(
    _todo15_adaptive_summary["residual_f1"]
    >= _todo15_todo11_sequential_f1
)
TODO15_GLOBAL_GATE_PASSED = bool(
    _todo15_adaptive_summary["exact_share"]
    > float(_todo11_sequential_exact.mean())
    and _todo15_adaptive_summary["material_share"]
    > float(_todo11_sequential_material.mean())
)
TODO15_SEGMENT_GATE_PASSED = bool(
    TODO15_IMPROVED_SEGMENTS >= TODO15_MINIMUM_IMPROVED_SEGMENTS
)
TODO15_ONE_STEP_GATE_PASSED = bool(
    TODO15_FAILURE_GATE_PASSED
    and TODO15_F1_GATE_PASSED
    and TODO15_GLOBAL_GATE_PASSED
    and TODO15_SEGMENT_GATE_PASSED
)
TODO15_STATUS = (
    "history_entry_one_step_gate_passed_recursive_pending"
    if TODO15_ONE_STEP_GATE_PASSED
    else "history_entry_one_step_gate_failed"
)
todo15_manifest = pd.Series(
    {
        "status": TODO15_STATUS,
        "history lengths selected nested LOSO": str(TODO15_HISTORY_LENGTHS),
        "history never crosses object/segment": True,
        "current target or current actual branch in history": False,
        "object2 uses predicted object1 distance": True,
        "continuation model and stay thresholds frozen from TODO11": True,
        "general pair models fitted": _todo15_general_pair_models_fitted,
        "history pair models fitted": _todo15_history_pair_models_fitted,
        "history outer models fitted": _todo15_outer_models_fitted,
        "material failure gate passed": TODO15_FAILURE_GATE_PASSED,
        "residual F1 gate passed": TODO15_F1_GATE_PASSED,
        "global TODO11 improvement gate passed": TODO15_GLOBAL_GATE_PASSED,
        "segments improved versus TODO11": TODO15_IMPROVED_SEGMENTS,
        "segment gate passed": TODO15_SEGMENT_GATE_PASSED,
        "one-step gate passed": TODO15_ONE_STEP_GATE_PASSED,
        "recursive rollout run": False,
        "fully blind test preserved": False,
    },
    name="value",
)

assert set(todo15_selected_history_by_segment) == set(
    _todo11_segments.tolist()
)
assert set(todo15_entry_threshold_by_segment) == set(
    _todo11_segments.tolist()
)
assert np.isfinite(_todo15_adaptive["position_error"]).all()
assert np.isfinite(_todo15_adaptive["angle_error"]).all()
assert np.isfinite(_todo15_adaptive["distance_error"]).all()
assert not TODO15_ONE_STEP_GATE_PASSED or TODO15_FAILURE_GATE_PASSED

display(
    Markdown("#### TODO 15 — nested history and threshold selection"),
    todo15_nested_selection.style.format(precision=6),
    Markdown("#### TODO 15 — outer-LOSO model comparison"),
    todo15_model_summary.style.format(precision=6),
    Markdown("#### TODO 15 — per-segment adaptive result"),
    todo15_per_segment.style.format(precision=6),
    Markdown("#### TODO 15 — entry/continuation/exit"),
    todo15_phase_summary.style.format(precision=6),
    Markdown("#### TODO 15 — remaining material failures"),
    todo15_material_failure_summary.style.format(precision=6),
    Markdown("#### TODO 15 — selected-model feature importance"),
    todo15_feature_importance.head(20).style.format(precision=6),
    Markdown("#### TODO 15 manifest"),
    todo15_manifest.to_frame(),
)
