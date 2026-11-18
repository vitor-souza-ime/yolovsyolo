"""Structural regression tests for the benchmark source; no robot/GPU required."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = ROOT / "main.py"
SOURCE = MAIN_FILE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(MAIN_FILE))


def get_function_source(name):
    matches = [
        node for node in TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one top-level function named {name}; found {len(matches)}.")
    return ast.get_source_segment(SOURCE, matches[0])


class BenchmarkSourceIntegrityTests(unittest.TestCase):
    def test_main_has_valid_python_syntax(self):
        self.assertIsNotNone(TREE)

    def test_only_one_performance_plot_function_exists(self):
        definitions = [
            node for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_performance_plots"
        ]
        self.assertEqual(len(definitions), 1)

    def test_json_export_uses_nested_confidence_statistics(self):
        saver = get_function_source("save_benchmark_results")
        self.assertIn('data.get("confidence_stats", {})', saver)
        self.assertNotIn('"max_confidence": data["max_confidence"]', saver)
        self.assertIn("allow_nan=False", saver)

    def test_report_uses_collected_metric_names(self):
        report = get_function_source("generate_final_report")
        self.assertIn('"confidence_mean"', report)
        self.assertIn('"confidence_max"', report)
        self.assertNotIn('"avg_confidence":', report)

    def test_failed_camera_capture_does_not_create_synthetic_frame(self):
        fallback = get_function_source("capture_image_alternative")
        self.assertIn("raise RuntimeError", fallback)
        self.assertNotIn("Image.new", fallback)


if __name__ == "__main__":
    unittest.main()
