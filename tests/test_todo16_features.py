"""Fast tests of the real TODO16 feature builder, without running RF sweeps."""

import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "notebooks/todo16_compact_dynamics.py"
COLUMNS = (
    "previous_x", "previous_y", "previous_angle", "driver_distance",
    "free_angle", "step_length", "free_x", "free_y", "x_penetration",
    "y_penetration", "x_hit_fraction", "y_hit_fraction", "previous_x_gap",
    "previous_y_gap", "predicted_x_crossing", "predicted_y_crossing",
    "both_axis_proposal", "previous_wall_contact",
    "previous_boundary_run_length", "previous_special",
)


def feature_namespace(mutate=None, continuous=False):
    """Execute the actual pre-fit script section on three short trajectories."""
    matrix = np.zeros((12, len(COLUMNS)), dtype=np.float32)
    index = {name: i for i, name in enumerate(COLUMNS)}
    for offset in (0, 4, 8):
        matrix[offset:offset + 4, index["driver_distance"]] = [10, 11, 13, 16]
        matrix[offset:offset + 4, index["step_length"]] = [10, 12, 14, 16]
        matrix[offset:offset + 4, index["previous_angle"]] = [179, -179, -177, -175]
        matrix[offset:offset + 4, index["free_angle"]] = [178, -178, -174, -170]
        matrix[offset:offset + 4, index["previous_x_gap"]] = [4, 3, 2, 1]
        matrix[offset:offset + 4, index["previous_y_gap"]] = [8, 8, 8, 8]
    matrix[1, index["predicted_x_crossing"]] = 1
    matrix[1, index["previous_wall_contact"]] = 1
    matrix[:, index["previous_special"]] = 1
    if mutate is not None:
        mutate(matrix, index)
    frame = pd.DataFrame({
        "object": ["object1"] * 4 + ["object2"] * 4 + ["object1"] * 4,
        "segment_id": [0] * 8 + [1] * 4,
        "segment_step": list(range(1, 5)) * 3,
    })
    sequences = [np.arange(i, i + 4) for i in (0, 4, 8)]
    if continuous:
        frame["object"] = "object1"
        frame["segment_id"] = 0
        frame["segment_step"] = np.arange(1, 13)
        sequences = [np.arange(12)]
    namespace = {
        "np": np, "pd": pd, "TODO11_ENTER_THRESHOLDS": (0.5, 0.6),
        "todo11_threshold_by_segment": {0: (0.6, 0.4), 1: (0.6, 0.3)},
        "TODO9_RANDOM_STATE": 42, "TODO9_FEATURE_COLUMNS": COLUMNS,
        "todo9_event_dataset": frame,
        "_todo11_sequence_indices": sequences,
        "_todo15_current_matrix": matrix,
        "_todo7_wrap_degrees": lambda angles: (angles + 180) % 360 - 180,
    }
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "_todo16_training_current_matrix"
            for target in node.targets
        ):
            break
        prefix.append(node)
    else:
        raise AssertionError("TODO16 feature section boundary is missing")
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace, matrix, index


class Todo16FeatureTests(unittest.TestCase):
    def test_feature_count_and_current_only_control(self):
        ns, matrix, index = feature_namespace()
        before = matrix.copy()
        rows = np.arange(len(matrix))
        current = ns["_todo16_design"]("current_only", matrix, rows)
        dynamics = ns["_todo16_design"]("compact_dynamics", matrix, rows)
        self.assertEqual(len(ns["_todo16_dynamic_feature_names"]), 31)
        self.assertEqual(dynamics.shape, (12, len(COLUMNS) + 31))
        np.testing.assert_array_equal(current, dynamics[:, :len(COLUMNS)])
        np.testing.assert_array_equal(matrix, before)
        self.assertTrue((current[:, index["previous_special"]] == 0).all())
        self.assertTrue(np.isfinite(dynamics).all())

    def test_lags_and_ages_reset_at_object_and_segment_boundaries(self):
        ns, matrix, _ = feature_namespace()
        features = ns["_todo16_compact_features"](matrix, np.arange(12))
        col = {name: i for i, name in enumerate(ns["_todo16_dynamic_feature_names"])}
        for row in (0, 4, 8):
            self.assertEqual(features[row, col["lag1_valid"]], 0)
            self.assertEqual(features[row, col["lag2_valid"]], 0)
            self.assertEqual(features[row, col["delta1__driver_distance"]], 0)
            self.assertEqual(features[row, col["age_since_crossing_valid"]], 0)
        self.assertEqual(features[2, col["age_since_crossing_capped8"]], 1)
        self.assertEqual(features[3, col["age_since_contact_capped8"]], 2)
        self.assertEqual(features[2, col["approach_wall_max"]], 1)
        self.assertEqual(features[2, col["delta2__driver_distance"]], 1)

    def test_angles_wrap_and_column_names_match_values(self):
        ns, matrix, _ = feature_namespace()
        features = ns["_todo16_compact_features"](matrix, np.arange(12))
        col = {name: i for i, name in enumerate(ns["_todo16_dynamic_feature_names"])}
        expected = {
            "delta1_previous_angle_sin": np.sin(np.deg2rad(2)),
            "delta2_previous_angle_sin": np.sin(np.deg2rad(4)),
            "delta1_free_angle_sin": np.sin(np.deg2rad(4)),
            "delta2_free_angle_sin": np.sin(np.deg2rad(8)),
        }
        for name, value in expected.items():
            self.assertAlmostEqual(features[2, col[name]], value, places=6)

    def test_dynamic_current_proposal_and_future_invariance(self):
        ns, matrix, index = feature_namespace()
        proposal = matrix[[2]].copy()
        proposal[0, index["driver_distance"]] = 20
        features = ns["_todo16_compact_features"](proposal, [2])
        col = {name: i for i, name in enumerate(ns["_todo16_dynamic_feature_names"])}
        self.assertEqual(features[0, col["delta1__driver_distance"]], 9)

        def mutate_future(values, columns):
            values[3, :] = 100
            values[4:, :] = 200

        changed, _, _ = feature_namespace(mutate_future)
        np.testing.assert_array_equal(
            features, changed["_todo16_compact_features"](proposal, [2])
        )

    def test_event_ages_are_capped_and_current_crossing_resets_age(self):
        ns, matrix, index = feature_namespace(continuous=True)
        features = ns["_todo16_compact_features"](matrix, np.arange(12))
        col = {name: i for i, name in enumerate(ns["_todo16_dynamic_feature_names"])}
        self.assertEqual(features[11, col["age_since_crossing_capped8"]], 8)
        self.assertEqual(features[11, col["age_since_contact_capped8"]], 8)
        self.assertEqual(features[11, col["age_since_crossing_valid"]], 1)
        proposal = matrix[[11]].copy()
        proposal[0, index["predicted_y_crossing"]] = 1
        changed = ns["_todo16_compact_features"](proposal, [11])
        self.assertEqual(changed[0, col["age_since_crossing_capped8"]], 0)


if __name__ == "__main__":
    unittest.main()
