import numpy as np

from ...descriptors.fingerprints import MorganFP
from ....utils.testing.base import QSPRTestCase
from ....utils.testing.check_mixins import DataPrepCheckMixIn
from ....utils.testing.path_mixins import DataSetsPathMixIn
from .activity_cliff import ActivityCliffSplit
from .knn_failure import KNNFailureSplit
from .substructure_distance import SubstructureDistanceSplit


class TestBiasedSplits(DataSetsPathMixIn, QSPRTestCase, DataPrepCheckMixIn):
    """Tests for the biased split family.

    Each splitter inherits from DataSplit, Randomized, DataSetDependent and yields
    one (train_indices, test_indices) fold per (intended_bias, repeat) pair.
    Effective bias for each fold is stored on `self.effectiveBiases`.
    """

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def _checkSweep(self, dataset, name, expected_n_folds):
        """Run checkSplit, then verify per-fold metadata was populated."""
        self.checkSplit(dataset, name)
        splitter = dataset.getSplit(name)
        self.assertEqual(len(splitter.effectiveBiases), expected_n_folds)
        for entry in splitter.effectiveBiases:
            self.assertIn("intended_bias", entry)
            self.assertIn("effective_bias", entry)
            self.assertIn("repeat", entry)
            self.assertIn("seed", entry)
            self.assertTrue(0.0 <= entry["effective_bias"] <= 1.0)
            self.assertTrue(0.0 <= entry["intended_bias"] <= 1.0)

    def testActivityCliffSplit(self):
        """Sweep over three intended biases, verify shape and effective bias."""
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        dataset.addSplit(
            ActivityCliffSplit(
                similarity_threshold=0.7,
                activity_threshold=1,
                intended_biases=[0.0, 0.5, 1.0],
                n_repeats=1,
                test_fraction=0.1,
                seed=42,
            ),
            name="ActivityCliffSplit_sweep",
        )
        self._checkSweep(dataset, "ActivityCliffSplit_sweep", expected_n_folds=3)

    def testActivityCliffSplitSingleBias(self):
        """Default single-fold behaviour, matching the basic test pattern."""
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        dataset.addSplit(
            ActivityCliffSplit(
                similarity_threshold=0.7,
                activity_threshold=1,
                test_fraction=0.1,
                seed=42,
            ),
            name="ActivityCliffSplit_single",
        )
        self._checkSweep(dataset, "ActivityCliffSplit_single", expected_n_folds=1)

    def testActivityCliffSplitWithRepeats(self):
        """Two biases x two repeats = four folds."""
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        dataset.addSplit(
            ActivityCliffSplit(
                similarity_threshold=0.7,
                activity_threshold=1,
                intended_biases=[0.0, 1.0],
                n_repeats=2,
                test_fraction=0.1,
                seed=42,
            ),
            name="ActivityCliffSplit_repeats",
        )
        self._checkSweep(dataset, "ActivityCliffSplit_repeats", expected_n_folds=4)

    def testKNNFailureSplit(self):
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        dataset.addSplit(
            KNNFailureSplit(
                similarity_threshold=0.5,
                activity_threshold=1,
                n_neighbors=3,
                intended_biases=[0.0, 0.5, 1.0],
                n_repeats=1,
                test_fraction=0.1,
                seed=42,
            ),
            name="KNNFailureSplit_sweep",
        )
        self._checkSweep(dataset, "KNNFailureSplit_sweep", expected_n_folds=3)

    def testKNNFailureSplitSingleBias(self):
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        dataset.addSplit(
            KNNFailureSplit(
                similarity_threshold=0.5,
                activity_threshold=1,
                n_neighbors=3,
                test_fraction=0.1,
                seed=42,
            ),
            name="KNNFailureSplit_single",
        )
        self._checkSweep(dataset, "KNNFailureSplit_single", expected_n_folds=1)

    def testSubstructureDistanceSplit(self):
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        dataset.addSplit(
            SubstructureDistanceSplit(
                similarity_threshold=0.6,
                intended_biases=[0.0, 0.5, 1.0],
                n_repeats=1,
                test_fraction=0.1,
                seed=42,
            ),
            name="SubstructureDistanceSplit_sweep",
        )
        self._checkSweep(dataset, "SubstructureDistanceSplit_sweep", expected_n_folds=3)

    def testSubstructureDistanceSplitSingleBias(self):
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        dataset.addSplit(
            SubstructureDistanceSplit(
                similarity_threshold=0.6,
                test_fraction=0.1,
                seed=42,
            ),
            name="SubstructureDistanceSplit_single",
        )
        self._checkSweep(
            dataset, "SubstructureDistanceSplit_single", expected_n_folds=1
        )

    def testReproducibility(self):
        """Two splitter instances with the same seed produce the same folds."""
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])

        splitter_a = ActivityCliffSplit(
            similarity_threshold=0.7,
            activity_threshold=1,
            intended_biases=[0.5],
            test_fraction=0.1,
            seed=42,
        )
        splitter_b = ActivityCliffSplit(
            similarity_threshold=0.7,
            activity_threshold=1,
            intended_biases=[0.5],
            test_fraction=0.1,
            seed=42,
        )
        dataset.addSplit(splitter_a, name="run_a")
        dataset.addSplit(splitter_b, name="run_b")
        self.checkSplit(dataset, "run_a")
        self.checkSplit(dataset, "run_b")

        bias_a = dataset.getSplit("run_a").effectiveBiases[0]["effective_bias"]
        bias_b = dataset.getSplit("run_b").effectiveBiases[0]["effective_bias"]
        self.assertEqual(bias_a, bias_b)

    def testActivityCliffPostHocVerification(self):
        """evaluateQuestionOnTest matches the stored effective_bias."""
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        splitter = ActivityCliffSplit(
            similarity_threshold=0.7,
            activity_threshold=1,
            intended_biases=[0.5],
            test_fraction=0.1,
            seed=42,
        )
        dataset.addSplit(splitter, name="cliff_verify")
        self.checkSplit(dataset, "cliff_verify")

        stored = dataset.getSplit("cliff_verify")
        train_ids, test_ids = dataset.getSplit("cliff_verify", as_type="ids")[0]

        df_index = dataset.getDF().index
        train_positions = np.where(df_index.isin(train_ids))[0]
        test_positions = np.where(df_index.isin(test_ids))[0]

        recomputed = stored.evaluateQuestionOnTest(train_positions, test_positions)
        self.assertAlmostEqual(
            recomputed,
            stored.effectiveBiases[0]["effective_bias"],
            places=6,
        )

    def testSubstructureDistancePostHocVerification(self):
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP()])
        splitter = SubstructureDistanceSplit(
            similarity_threshold=0.6,
            intended_biases=[0.5],
            test_fraction=0.1,
            seed=42,
        )
        dataset.addSplit(splitter, name="substruct_verify")
        self.checkSplit(dataset, "substruct_verify")

        stored = dataset.getSplit("substruct_verify")
        train_ids, test_ids = dataset.getSplit("substruct_verify", as_type="ids")[0]

        df_index = dataset.getDF().index
        train_positions = np.where(df_index.isin(train_ids))[0]
        test_positions = np.where(df_index.isin(test_ids))[0]

        recomputed = stored.evaluateQuestionOnTest(train_positions, test_positions)
        self.assertAlmostEqual(
            recomputed,
            stored.effectiveBiases[0]["effective_bias"],
            places=6,
        )
