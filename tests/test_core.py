import unittest

from metrics import mean, paired_bootstrap_mean_diff_ci, stdev
from retriever import SupportKnowledgeRetriever
from run_modes import PIPELINE_MODES, get_pipeline_run_config


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


if __name__ == "__main__":
    unittest.main()
