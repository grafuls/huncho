"""Check leakage prevention and observed label semantics without downloads."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

file = Path(__file__).resolve().parents[1] / "prepare_kev_calibration.py"
spec = importlib.util.spec_from_file_location("prepare_kev_calibration", file)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
file = file.with_name("collect_kev_calibration.py")
spec = importlib.util.spec_from_file_location("collect_kev_calibration", file)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
file = file.with_name("evaluate_kev_calibration.py")
spec = importlib.util.spec_from_file_location("evaluate_kev_calibration", file)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class CalibrationDataTests(unittest.TestCase):
    def test_split_selection_is_reproducible_and_excludes_overlapping_inputs(self):
        fit = [{"sentence": text, "label": i % 2} for i, text in enumerate(["shared", "fit-a", "fit-b", "fit-c"])]
        evaluation = [{"sentence": text, "label": i % 2} for i, text in enumerate(["shared", "eval-a", "eval-b"])]
        selected = data.select_rows("choice", fit, evaluation, 3, 2, 7)
        self.assertEqual(selected, data.select_rows("choice", fit, evaluation, 3, 2, 7))
        self.assertEqual([i for i, _ in selected[0]], [1, 2, 3])
        self.assertFalse({data.content_key("choice", row) for _, row in selected[0]} & {data.content_key("choice", row) for row in evaluation})
        with self.assertRaises(ValueError):
            data.select_rows("choice", fit, evaluation, 4, 2, 7)

    def test_hidden_and_malformed_labels_are_rejected(self):
        for label in [-1, True, "1"]:
            with self.assertRaises(ValueError):
                data.question("choice", {"sentence": "review", "label": label})
        with self.assertRaises(ValueError):
            data.question("noul", {"passage": "passage", "question": "question", "answer": "false"})
        for label in [5, -1, False]:
            with self.assertRaises(ValueError):
                data.question("score", {"text": "review", "label": label})

    def test_observed_targets_match_wire_candidate_labels(self):
        self.assertEqual(data.question("choice", {"sentence": "review", "label": 0})[2], "negative")
        self.assertEqual(data.question("noul", {"passage": "passage", "question": "question", "answer": False})[2], "no")
        for label in range(5):
            _, prompt, target = data.question("score", {"text": "review", "label": label})
            self.assertEqual(target, str(label))
            self.assertEqual(prompt["criteria"][int(target)], f"{label + 1} star" + ("s" if label else ""))

    def test_fitting_target_uses_prompt_order_and_preserves_observed_heldout_label(self):
        record = {"id": "case", "source": {}, "request": {"model": "kev-4b", "questions": {"question": {"type": "choice", "criteria": {"positive": None, "negative": None}}}}, "targets": {"question": "negative"}}
        response = {"model": "kev-4b", "extensions": {"raw_logits": {"question": [2.0, 1.0]}}, "answers": {"question": {"type": "choice", "probabilities": {"negative": 0.3, "positive": 0.7}}}}
        row, case = collector.extract(record, response)
        self.assertEqual(row["target"], 1)
        self.assertEqual(case["targets"], {"question": "negative"})
        response["extensions"]["raw_logits"]["question"] = [float("nan"), 1.0]
        with self.assertRaises(ValueError):
            collector.extract(record, response)

    def test_kev_noul_logit_order_is_no_yes_and_must_match_the_wire_response(self):
        record = {"id": "noul", "source": {}, "request": {"model": "kev-4b", "questions": {"question": {"type": "noul"}}}, "targets": {"question": "yes"}}
        response = {"model": "kev-4b", "extensions": {"raw_logits": {"question": [1., 3.]}}, "answers": {"question": {"type": "noul", "noul": .7}}}
        row, _ = collector.extract(record, response)
        self.assertEqual(row["labels"], ["no", "yes"])
        self.assertEqual(row["target"], 1)
        response["answers"]["question"]["noul"] = .3
        with self.assertRaises(ValueError):
            collector.extract(record, response)

    def test_reused_fitting_logits_are_reindexed_from_pinned_observed_labels(self):
        record = {"id": "noul", "source": {"row": 7}, "request": {"questions": {"question": {"type": "noul"}}}, "targets": {"question": "yes"}}
        old = {"id": "noul", "source": {"row": 7}, "qtype": "noul", "target": 0, "logits": [1., 3.]}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "fit.jsonl"
            file.write_text(json.dumps(old) + "\n")
            row = collector.reuse_fitting(file, [record])[0]
            self.assertEqual(row["target"], 1)
            self.assertEqual(row["logits"], old["logits"])
            old["source"] = {"row": 8}
            file.write_text(json.dumps(old) + "\n")
            with self.assertRaises(ValueError):
                collector.reuse_fitting(file, [record])

    def test_outcome_ece_uses_observed_accuracy_and_temperature_precedence(self):
        # Same confidence, one correct and one wrong: observed accuracy is .5.
        values = [{"confidence": .8, "correct": correct, "nll": 1., "brier": .4} for correct in (0, 1)]
        self.assertAlmostEqual(evaluator.metrics(values)["ece"], .3)
        entry = {"temperature": 2., "per_type_temperatures": {"score": 3.}, "temperature_by_options": {"score:3-5": 4.}}
        self.assertEqual(evaluator.temperature(entry, "choice", 2), 2.)
        self.assertEqual(evaluator.temperature(entry, "score", 8), 3.)
        self.assertEqual(evaluator.temperature(entry, "score", 5), 4.)

    def test_paired_evaluation_keeps_accuracy_and_handles_extreme_logits(self):
        rows = [{"logits": [1000., -1000.], "target": target, "qtype": "choice"} for target in (0, 1)]
        report = evaluator.comparison(rows, {"temperature": 1.}, {"temperature": 2.}, repetitions=32)
        self.assertEqual(report["baseline"]["accuracy"], .5)
        self.assertEqual(report["refit"]["accuracy"], .5)
        self.assertEqual(report["baseline"]["nll"], 1000.)
        self.assertEqual(report["refit"]["nll"], 500.)
        self.assertEqual(report, evaluator.comparison(rows, {"temperature": 1.}, {"temperature": 2.}, repetitions=32))


if __name__ == "__main__":
    unittest.main()
