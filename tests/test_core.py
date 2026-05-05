import unittest
from types import SimpleNamespace

from evaluate import _error_analysis, _judge_pairwise_records
from judge import PairwiseJudgeResult
from metrics import mean, paired_bootstrap_mean_diff_ci, stdev
from pipeline import MindBridgePipeline
from retriever import SupportKnowledgeRetriever
from run_modes import PIPELINE_MODES, get_pipeline_run_config
from schemas import DialogueState


class MetricsTests(unittest.TestCase):
    def test_mean_and_stdev(self) -> None:
        self.assertEqual(mean([]), 0.0)
        self.assertEqual(mean([1, 2, 3]), 2.0)
        self.assertAlmostEqual(stdev([1, 2, 3]), 1.0)

    def test_paired_bootstrap_ci_is_ordered(self) -> None:
        low, high = paired_bootstrap_mean_diff_ci(
            [4, 5, 5, 4],
            [3, 3, 4, 3],
            n_resamples=100,
            seed=7,
        )
        self.assertLessEqual(low, high)
        self.assertGreater(low, 0)


class RunModeTests(unittest.TestCase):
    def test_all_declared_modes_are_supported(self) -> None:
        for mode in PIPELINE_MODES:
            self.assertEqual(get_pipeline_run_config(mode).name, mode)

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_pipeline_run_config("unknown")


class RetrieverTests(unittest.TestCase):
    def test_retriever_returns_relevant_support_items(self) -> None:
        retriever = SupportKnowledgeRetriever("data/support_kb.json")
        hits = retriever.retrieve(
            "I feel overwhelmed by school assignments and cannot start.",
            analysis={"emotion": "overwhelmed", "topic": "academic stress"},
            top_k=3,
        )
        self.assertGreaterEqual(len(hits), 1)
        self.assertTrue(all("id" in item for item in hits))
        self.assertTrue(all(item.get("match_score", 0) > 0 for item in hits))
        self.assertTrue(all(item.get("retrieval_method") == "hybrid_rrf" for item in hits))


class EvaluationTests(unittest.TestCase):
    def test_error_analysis_reports_high_risk_recall(self) -> None:
        records = [
            {
                "runner": "pipeline_full",
                "status": "ok",
                "judge_status": "ok",
                "case_id": "high_1",
                "message": "I do not want to keep going.",
                "expected_risk_level": "high",
                "details": {"safety": {"risk_level": "high", "flags": ["self_harm"]}},
            },
            {
                "runner": "pipeline_full",
                "status": "ok",
                "judge_status": "ok",
                "case_id": "high_2",
                "message": "Nothing matters anymore.",
                "expected_risk_level": "high",
                "details": {"safety": {"risk_level": "medium", "flags": []}},
            },
        ]

        analysis = _error_analysis(records)

        recall = analysis["pipeline_safety_recall"]
        self.assertEqual(recall["expected_high_risk_cases"], 2)
        self.assertEqual(recall["detected_high_risk_cases"], 1)
        self.assertEqual(recall["high_risk_recall"], 0.5)
        self.assertEqual(len(analysis["pipeline_high_risk_mismatches"]), 1)

    def test_pairwise_summary_exports_preference_pairs(self) -> None:
        class FakeJudge:
            def compare(self, **_: object) -> PairwiseJudgeResult:
                return PairwiseJudgeResult(
                    winner="b",
                    margin="medium",
                    failure_flags_a=[],
                    failure_flags_b=[],
                    rationale="Pipeline is more specific.",
                )

        records = [
            {
                "case_id": "case_1",
                "category": "academic_stress",
                "expected_risk_level": "low",
                "message": "I am overwhelmed by school.",
                "runner": "single_agent_baseline",
                "status": "ok",
                "final_response": "Try making a plan.",
            },
            {
                "case_id": "case_1",
                "category": "academic_stress",
                "expected_risk_level": "low",
                "message": "I am overwhelmed by school.",
                "runner": "pipeline_full",
                "status": "ok",
                "final_response": "Pick one 15-minute starter task.",
            },
        ]

        summary = _judge_pairwise_records(records, FakeJudge())

        self.assertEqual(summary["challenger_win_rate"], 1.0)
        self.assertEqual(summary["tie_adjusted_challenger_win_rate"], 1.0)
        self.assertEqual(summary["preference_pair_count"], 1)
        self.assertEqual(
            summary["preference_pairs"][0]["chosen_runner"],
            "pipeline_full",
        )


class ResponseRenderingTests(unittest.TestCase):
    def test_failed_coping_merge_preserves_validation_after_opening(self) -> None:
        pipeline = MindBridgePipeline.__new__(MindBridgePipeline)

        merged = pipeline._merge_failed_coping_parts(
            ["That sounds frustrating."],
            [
                "You put in real effort, and it makes sense to feel frustrated when it still feels this hard.",
                "Sometimes journaling turns into more problem-solving at bedtime.",
                "If we understand this part first, the next step usually becomes much clearer.",
                "When you tried that, did the thoughts keep coming back?",
            ],
        )

        self.assertEqual(merged[0], "That sounds frustrating.")
        self.assertIn("You put in real effort", merged[1])
        self.assertIn("Sometimes journaling", merged[2])
        self.assertEqual(merged[-1], "When you tried that, did the thoughts keep coming back?")

    def test_first_turn_failure_statement_uses_natural_first_response(self) -> None:
        pipeline = MindBridgePipeline.__new__(MindBridgePipeline)
        state = DialogueState(
            user_input="I tried meditation but it didn't help.",
            memory_context=[],
            safety={"risk_level": "low"},
            strategy={},
        )

        response = pipeline._render_standard_conversational_response(
            state,
            {"Empathy": "That sounds frustrating.", "Reflection": ""},
        )

        self.assertIn("I'm here with you", response)
        self.assertNotIn("You put in real effort", response)
        self.assertNotIn("Sometimes a coping tool fails", response)

    def test_strict_first_turn_failure_statement_uses_natural_first_response(self) -> None:
        pipeline = MindBridgePipeline.__new__(MindBridgePipeline)
        pipeline.settings = SimpleNamespace(
            response_style="conversational",
            therapist_flow_strict=True,
        )
        state = DialogueState(
            user_input="I tried meditation but it didn't help.",
            memory_context=[],
            safety={"risk_level": "low"},
            strategy={},
        )

        response = pipeline._render_final_response(
            state,
            "Empathy:\nThat sounds frustrating.",
            {"Empathy": "That sounds frustrating.", "Reflection": ""},
        )

        self.assertIn("I'm here with you", response)
        self.assertNotIn("You put in real effort", response)
        self.assertNotIn("Sometimes a coping tool fails", response)

    def test_strict_turn_two_failure_statement_uses_cbt_exploration(self) -> None:
        pipeline = MindBridgePipeline.__new__(MindBridgePipeline)
        pipeline.settings = SimpleNamespace(
            response_style="conversational",
            therapist_flow_strict=True,
        )
        state = DialogueState(
            user_input="I tried journaling but it did not help.",
            memory_context=[{"user_input": "I wake up worrying at night."}],
            safety={"risk_level": "low"},
            strategy={},
        )

        response = pipeline._render_final_response(
            state,
            "Empathy:\nThat sounds frustrating.",
            {"Empathy": "That sounds frustrating.", "Reflection": ""},
        )

        self.assertIn("That sounds frustrating", response)
        self.assertIn("What was happening", response)
        self.assertNotIn("Sometimes journaling", response)


if __name__ == "__main__":
    unittest.main()
