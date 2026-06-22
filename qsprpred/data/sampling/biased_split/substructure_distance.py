import os
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
import networkx as nx
from PIL import Image

from .molecularnetwork import (
    smiles_to_ecfp4_bitvect,
    compute_similarity_matrix,
    visualise_molnet_split,
)
from ...tables.interfaces.data_set_dependent import DataSetDependent
from ...tables.interfaces.qspr_data_set import QSPRDataSet
from ....utils.interfaces.randomized import Randomized
from ..splits import DataSplit

UNASSIGNED_NODE = 0
TRAIN_NODE = 1
TEST_NODE = 2


class SubstructureDistanceSplitter:
    def __init__(self, similarity_threshold, test_fraction=0.2):
        self.similarity_threshold = similarity_threshold
        self.test_fraction = test_fraction

    def split_for_intended_bias(
        self, smiless, tversky_matrix, activity_values, intended_bias, random_seed
    ):
        if not (0.0 <= intended_bias <= 1.0):
            raise ValueError(f"intended_bias must be in [0, 1], got {intended_bias}")

        rng = np.random.default_rng(random_seed)
        n_molecules = len(smiless)
        target_test_size = int(self.test_fraction * n_molecules)
        n_isolated_test_target = int(intended_bias * target_test_size)

        components = self.find_components(tversky_matrix, self.similarity_threshold)
        assignment = self.walk_components(
            components, n_molecules, n_isolated_test_target, rng
        )

        unassigned_indices = np.where(assignment == UNASSIGNED_NODE)[0]
        n_random_fill = target_test_size - int((assignment == TEST_NODE).sum())
        if n_random_fill > 0 and len(unassigned_indices) > 0:
            n_to_sample = min(n_random_fill, len(unassigned_indices))
            random_test_indices = rng.choice(
                unassigned_indices, size=n_to_sample, replace=False
            )
            assignment[random_test_indices] = TEST_NODE

        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE

        train_indices = np.where(assignment == TRAIN_NODE)[0]
        test_indices = np.where(assignment == TEST_NODE)[0]

        question_results = self.evaluate_substructure_question(
            test_indices, train_indices, tversky_matrix, self.similarity_threshold
        )
        effective_bias = self.effective_bias_from_question_results(question_results)

        return train_indices, test_indices, effective_bias

    def split(self, smiless, activity_values, intended_biases, n_repeats):
        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        tversky_matrix = compute_similarity_matrix(fps_bitvect, method="tversky")
        for intended_bias in intended_biases:
            for repeat_index in range(n_repeats):
                train_indices, test_indices, effective_bias = (
                    self.split_for_intended_bias(
                        smiless=smiless,
                        tversky_matrix=tversky_matrix,
                        activity_values=activity_values,
                        intended_bias=intended_bias,
                        random_seed=repeat_index,
                    )
                )
                yield train_indices, test_indices, effective_bias, intended_bias, repeat_index

    @staticmethod
    def find_components(tversky_matrix, similarity_threshold):
        adj_matrix = np.triu(tversky_matrix, k=1)
        adj_matrix[adj_matrix < similarity_threshold] = 0
        similarity_graph = nx.from_numpy_array(adj_matrix)
        return sorted(nx.connected_components(similarity_graph), key=len, reverse=True)

    @staticmethod
    def walk_components(components, n_molecules, n_isolated_test_target, rng):
        assignment = np.full(n_molecules, UNASSIGNED_NODE, dtype=np.int8)
        remaining_budget = n_isolated_test_target
        unused_components = list(components)
        while True:
            fitting = [c for c in unused_components if len(c) <= remaining_budget]
            if not fitting:
                break
            max_size = max(len(c) for c in fitting)
            largest = [c for c in fitting if len(c) == max_size]
            chosen = largest[int(rng.integers(len(largest)))]
            for molecule_index in chosen:
                assignment[molecule_index] = TEST_NODE
            unused_components.remove(chosen)
            remaining_budget -= len(chosen)
        return assignment

    @staticmethod
    def evaluate_substructure_question(
        test_indices, train_indices, tversky_matrix, similarity_threshold
    ):
        if len(test_indices) == 0:
            return np.array([], dtype=float)
        if len(train_indices) == 0:
            return np.ones(len(test_indices), dtype=float)
        similarity_test_vs_train = tversky_matrix[np.ix_(test_indices, train_indices)]
        max_train_similarity = similarity_test_vs_train.max(axis=1)
        is_isolated = max_train_similarity < similarity_threshold
        return is_isolated.astype(float)

    @staticmethod
    def effective_bias_from_question_results(question_results):
        if question_results.size == 0:
            return 0.0
        return float(question_results.mean())

    @staticmethod
    def build_visualization_network(
        smiless, activity_values, tversky_matrix, similarity_threshold
    ):
        adj_matrix = np.triu(tversky_matrix, k=1)
        adj_matrix[adj_matrix < similarity_threshold] = 0
        G = nx.from_numpy_array(adj_matrix)
        node_attrs = {
            n: {"smiles": smi, "activity": act}
            for n, (smi, act) in enumerate(zip(smiless, activity_values))
        }
        nx.set_node_attributes(G, node_attrs)
        G.graph["activity_label"] = "activity"
        G.graph["activity_threshold"] = np.inf
        G.graph["similarity_threshold"] = similarity_threshold
        G.graph["similarity_fp"] = "2048bit ECFP4"
        G.graph["similarity_distance"] = "tversky"
        return G

    def visualise_splits(
        self,
        smiless,
        activity_values,
        intended_biases,
        n_repeats,
        output_path,
        duration=500,
    ):
        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        tversky_matrix = compute_similarity_matrix(fps_bitvect, method="tversky")
        G = self.build_visualization_network(
            smiless, activity_values, tversky_matrix, self.similarity_threshold
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            frame_index = 0
            for intended_bias in intended_biases:
                for repeat_index in range(n_repeats):
                    train_idx, test_idx, effective_bias = self.split_for_intended_bias(
                        smiless=smiless,
                        tversky_matrix=tversky_matrix,
                        activity_values=activity_values,
                        intended_bias=intended_bias,
                        random_seed=repeat_index,
                    )
                    p = os.path.join(tmpdir, f"frame_{frame_index:04d}.png")
                    visualise_molnet_split(
                        G,
                        train_idx,
                        test_idx,
                        effective_bias,
                        intended_bias,
                        filepath=p,
                        cliff=False,
                    )
                    paths.append(p)
                    frame_index += 1
            frames = [Image.open(p) for p in paths]
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )


class SubstructureDistanceSplit(DataSplit, Randomized, DataSetDependent):
    """QSPRPred-compatible wrapper around SubstructureDistanceSplitter.

    Yields one (train_indices, test_indices) fold per (intended_bias, repeat) pair.
    Effective bias per fold is stored on `self.effectiveBiases` as a list of
    dicts with keys `intended_bias`, `effective_bias`, `repeat`, `seed`.

    Effective bias here is the fraction of test molecules whose maximum
    symmetrised-Tversky similarity to any train molecule falls below
    `similarity_threshold`, i.e. test molecules with no shared substructure
    above threshold to anything in train.

    Attributes:
        similarityThreshold (float): Threshold on symmetrised Tversky.
        intendedBiases (list[float]): Sequence of intended bias values to sweep.
        nRepeats (int): Repeats per intended bias.
        testFraction (float): Fraction of dataset assigned to test per fold.
        effectiveBiases (list[dict]): Populated during split(). One entry per fold.
    """

    def __init__(
        self,
        similarity_threshold,
        intended_biases=(1.0,),
        n_repeats=1,
        test_fraction=0.1,
        seed=None,
        data_set: QSPRDataSet | None = None,
    ):
        super().__init__(data_set)
        self.similarityThreshold = similarity_threshold
        self.intendedBiases = list(intended_biases)
        self.nRepeats = n_repeats
        self.testFraction = test_fraction
        self.randomState = seed
        self.effectiveBiases: list[dict] = []

    @property
    def randomState(self) -> int | None:
        return self._seed

    @randomState.setter
    def randomState(self, seed: int | None):
        self._seed = seed

    def split(self, X, y) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        assert (
            self.hasDataSet
        ), "No dataset attached to this splitter, set dataset with setDataSet()"

        y_index = y.index.copy()
        smiless = self.dataSet.getDF()[self.dataSet.smilesProp][y_index].values

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(
                    "SubstructureDistanceSplit only supports single-task datasets; "
                    f"got {y.shape[1]} target columns"
                )
            activity_values = np.asarray(y.iloc[:, 0].values, dtype=float)
        else:
            activity_values = np.asarray(y.values, dtype=float)

        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        tversky_matrix = compute_similarity_matrix(fps_bitvect, method="tversky")

        underlying = SubstructureDistanceSplitter(
            similarity_threshold=self.similarityThreshold,
            test_fraction=self.testFraction,
        )

        self.effectiveBiases = []
        base_seed = self._seed if self._seed is not None else 0
        fold_index = 0
        for intended_bias in self.intendedBiases:
            for repeat_index in range(self.nRepeats):
                seed = base_seed + fold_index
                train_idx, test_idx, effective_bias = (
                    underlying.split_for_intended_bias(
                        smiless=smiless,
                        tversky_matrix=tversky_matrix,
                        activity_values=activity_values,
                        intended_bias=intended_bias,
                        random_seed=seed,
                    )
                )
                self.effectiveBiases.append(
                    {
                        "intended_bias": float(intended_bias),
                        "effective_bias": float(effective_bias),
                        "repeat": int(repeat_index),
                        "seed": int(seed),
                    }
                )
                fold_index += 1
                yield (
                    np.asarray(train_idx, dtype=int),
                    np.asarray(test_idx, dtype=int),
                )

    def evaluateQuestionOnTest(self, train_indices, test_indices):
        """Recompute the substructure-isolation question on a given partition."""
        assert (
            self.hasDataSet
        ), "No dataset attached to this splitter, set dataset with setDataSet()"
        smiless = self.dataSet.getDF()[self.dataSet.smilesProp].values

        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        tversky_matrix = compute_similarity_matrix(fps_bitvect, method="tversky")

        question_results = SubstructureDistanceSplitter.evaluate_substructure_question(
            test_indices=np.asarray(test_indices, dtype=int),
            train_indices=np.asarray(train_indices, dtype=int),
            tversky_matrix=tversky_matrix,
            similarity_threshold=self.similarityThreshold,
        )
        return SubstructureDistanceSplitter.effective_bias_from_question_results(
            question_results
        )
