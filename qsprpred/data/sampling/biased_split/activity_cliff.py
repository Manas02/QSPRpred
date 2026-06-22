import os
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from .molecularnetwork import (
    smiles_to_ecfp4_bitvect,
    compute_similarity_matrix,
    molecular_network_from_list,
    visualise_molnet_split,
)
from ...tables.interfaces.data_set_dependent import DataSetDependent
from ...tables.interfaces.qspr_data_set import QSPRDataSet
from ....utils.interfaces.randomized import Randomized
from ..splits import DataSplit

UNASSIGNED_NODE = 0
TRAIN_NODE = 1
TEST_NODE = 2


class ActivityCliffSplitter:
    def __init__(
        self,
        similarity_threshold,
        activity_threshold,
        test_fraction=0.2,
    ):
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.test_fraction = test_fraction

    def split_for_intended_bias(
        self,
        smiless,
        activity_values,
        similarity_matrix,
        intended_bias,
        random_seed,
    ):
        if not (0.0 <= intended_bias <= 1.0):
            raise ValueError(f"intended_bias must be in [0, 1], got {intended_bias}")

        rng = np.random.default_rng(random_seed)
        n_molecules = len(smiless)
        target_test_size = int(self.test_fraction * n_molecules)
        n_cliff_test_molecules = int(intended_bias * target_test_size)

        cliff_edges = self.find_cliff_edges(
            similarity_matrix=similarity_matrix,
            activity_values=activity_values,
            similarity_threshold=self.similarity_threshold,
            activity_threshold=self.activity_threshold,
        )

        rng.shuffle(cliff_edges)

        cliff_degrees = self.compute_cliff_degrees(cliff_edges, n_molecules)

        assignment = self.walk_cliff_edges(
            cliff_edges=cliff_edges,
            cliff_degrees=cliff_degrees,
            n_molecules=n_molecules,
            n_cliff_test_target=n_cliff_test_molecules,
            rng=rng,
        )

        unassigned_indices = np.where(assignment == UNASSIGNED_NODE)[0]
        unassigned_non_cliff_indices = unassigned_indices[
            cliff_degrees[unassigned_indices] == 0
        ]
        unassigned_cliff_indices = unassigned_indices[
            cliff_degrees[unassigned_indices] > 0
        ]

        n_random_fill = target_test_size - int((assignment == TEST_NODE).sum())

        if n_random_fill > 0:
            if len(unassigned_non_cliff_indices) >= n_random_fill:
                random_test_indices = rng.choice(
                    unassigned_non_cliff_indices, size=n_random_fill, replace=False
                )
            else:
                shortfall = n_random_fill - len(unassigned_non_cliff_indices)
                cliff_topup_indices = rng.choice(
                    unassigned_cliff_indices,
                    size=min(shortfall, len(unassigned_cliff_indices)),
                    replace=False,
                )
                random_test_indices = np.concatenate(
                    [unassigned_non_cliff_indices, cliff_topup_indices]
                )
            assignment[random_test_indices] = TEST_NODE

        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE

        train_indices = np.where(assignment == TRAIN_NODE)[0]
        test_indices = np.where(assignment == TEST_NODE)[0]

        question_results = self.evaluate_cliff_question(
            test_indices=test_indices,
            train_indices=train_indices,
            similarity_matrix=similarity_matrix,
            activity_values=activity_values,
            similarity_threshold=self.similarity_threshold,
            activity_threshold=self.activity_threshold,
        )

        effective_bias = self.effective_bias_from_question_results(question_results)
        return train_indices, test_indices, effective_bias

    def split(self, smiless, activity_values, intended_biases, n_repeats):
        fps_bitvect = [smiles_to_ecfp4_bitvect(smiles) for smiles in smiless]
        similarity_matrix = compute_similarity_matrix(fps_bitvect)

        for intended_bias in intended_biases:
            for repeat_index in range(n_repeats):
                train_indices, test_indices, effective_bias = (
                    self.split_for_intended_bias(
                        smiless=smiless,
                        activity_values=activity_values,
                        similarity_matrix=similarity_matrix,
                        intended_bias=intended_bias,
                        random_seed=repeat_index,
                    )
                )
                yield train_indices, test_indices, effective_bias, intended_bias, repeat_index

    @staticmethod
    def effective_bias_from_question_results(question_results):
        if question_results.size == 0:
            return 0.0
        return float(question_results.mean())

    @staticmethod
    def evaluate_cliff_question(
        test_indices,
        train_indices,
        similarity_matrix,
        activity_values,
        activity_threshold,
        similarity_threshold,
    ):
        if len(test_indices) == 0:
            return np.array([])

        similarity_test_vs_train = similarity_matrix[
            test_indices[:, None], train_indices
        ]

        activity_diff_test_vs_train = np.abs(
            activity_values[test_indices][:, None] - activity_values[train_indices]
        )

        is_cliff_edge = (similarity_test_vs_train >= similarity_threshold) & (
            activity_diff_test_vs_train >= activity_threshold
        )

        test_molecule_has_cliff_partner = is_cliff_edge.any(axis=1)
        return test_molecule_has_cliff_partner.astype(float)

    @staticmethod
    def find_cliff_edges(
        similarity_matrix,
        activity_values,
        similarity_threshold,
        activity_threshold,
    ):
        n = len(activity_values)
        cliff_edges = []

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] < similarity_threshold:
                    continue
                activity_difference = abs(
                    float(activity_values[i]) - float(activity_values[j])
                )
                if activity_difference >= activity_threshold:
                    cliff_edges.append((i, j, activity_difference))

        return cliff_edges

    @staticmethod
    def compute_cliff_degrees(cliff_edges, n_molecules):
        degrees = np.zeros(n_molecules, dtype=int)
        for mol_a, mol_b, _ in cliff_edges:
            degrees[mol_a] += 1
            degrees[mol_b] += 1
        return degrees

    @staticmethod
    def walk_cliff_edges(
        cliff_edges, cliff_degrees, n_molecules, n_cliff_test_target, rng
    ):
        assignment = np.full(n_molecules, UNASSIGNED_NODE, dtype=np.int8)
        n_cliff_test_placed = 0

        for mol_a, mol_b, _ in cliff_edges:
            if n_cliff_test_placed >= n_cliff_test_target:
                break

            status_a = assignment[mol_a]
            status_b = assignment[mol_b]

            if status_a == UNASSIGNED_NODE and status_b == UNASSIGNED_NODE:
                if cliff_degrees[mol_a] > cliff_degrees[mol_b]:
                    train_molecule, test_molecule = mol_a, mol_b
                elif cliff_degrees[mol_b] > cliff_degrees[mol_a]:
                    train_molecule, test_molecule = mol_b, mol_a
                else:
                    if rng.random() < 0.5:
                        train_molecule, test_molecule = mol_a, mol_b
                    else:
                        train_molecule, test_molecule = mol_b, mol_a

                assignment[train_molecule] = TRAIN_NODE
                assignment[test_molecule] = TEST_NODE
                n_cliff_test_placed += 1

            elif status_a == TRAIN_NODE and status_b == UNASSIGNED_NODE:
                assignment[mol_b] = TEST_NODE
                n_cliff_test_placed += 1

            elif status_b == TRAIN_NODE and status_a == UNASSIGNED_NODE:
                assignment[mol_a] = TEST_NODE
                n_cliff_test_placed += 1

            elif status_a == TEST_NODE and status_b == UNASSIGNED_NODE:
                assignment[mol_b] = TRAIN_NODE

            elif status_b == TEST_NODE and status_a == UNASSIGNED_NODE:
                assignment[mol_a] = TRAIN_NODE

        return assignment

    def visualise_splits(
        self,
        smiless,
        activity_values,
        intended_biases,
        n_repeats,
        output_path,
        duration=500,
    ):
        G = molecular_network_from_list(
            smiless, activity_values, self.similarity_threshold, self.activity_threshold
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for frame_index, (
                train_idx,
                test_idx,
                effective_bias,
                intended_bias,
                _,
            ) in enumerate(
                self.split(smiless, activity_values, intended_biases, n_repeats)
            ):
                p = os.path.join(tmpdir, f"frame_{frame_index:04d}.png")
                visualise_molnet_split(
                    G, train_idx, test_idx, effective_bias, intended_bias, filepath=p
                )
                paths.append(p)
            frames = [Image.open(p) for p in paths]
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )


class ActivityCliffSplit(DataSplit, Randomized, DataSetDependent):
    """QSPRPred-compatible wrapper around ActivityCliffSplitter.

    Yields one (train_indices, test_indices) fold per (intended_bias, repeat) pair.
    Effective bias per fold is stored on `self.effectiveBiases` as a list of
    dicts with keys `intended_bias`, `effective_bias`, `repeat`, `seed`.

    Attributes:
        similarityThreshold (float): Minimum Tanimoto similarity for a cliff edge.
        activityThreshold (float): Minimum activity difference for a cliff edge.
        intendedBiases (list[float]): Sequence of intended bias values to sweep.
        nRepeats (int): Repeats per intended bias.
        testFraction (float): Fraction of dataset assigned to test per fold.
        effectiveBiases (list[dict]): Populated during split(). One entry per fold.
    """

    def __init__(
        self,
        similarity_threshold,
        activity_threshold,
        intended_biases=(1.0,),
        n_repeats=1,
        test_fraction=0.1,
        seed=None,
        data_set: QSPRDataSet | None = None,
    ):
        super().__init__(data_set)
        self.similarityThreshold = similarity_threshold
        self.activityThreshold = activity_threshold
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
                    "ActivityCliffSplit only supports single-task datasets; "
                    f"got {y.shape[1]} target columns"
                )
            activity_values = np.asarray(y.iloc[:, 0].values, dtype=float)
        else:
            activity_values = np.asarray(y.values, dtype=float)

        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        similarity_matrix = compute_similarity_matrix(fps_bitvect)

        underlying = ActivityCliffSplitter(
            similarity_threshold=self.similarityThreshold,
            activity_threshold=self.activityThreshold,
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
                        activity_values=activity_values,
                        similarity_matrix=similarity_matrix,
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
        """Recompute the cliff-question result on a given partition.

        Useful for post-hoc verification that a stored fold's effective_bias
        matches the realised split.
        """
        assert (
            self.hasDataSet
        ), "No dataset attached to this splitter, set dataset with setDataSet()"
        smiless = self.dataSet.getDF()[self.dataSet.smilesProp].values
        activity_values = np.asarray(
            self.dataSet.getTargets().iloc[:, 0].values, dtype=float
        )

        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        similarity_matrix = compute_similarity_matrix(fps_bitvect)

        question_results = ActivityCliffSplitter.evaluate_cliff_question(
            test_indices=np.asarray(test_indices, dtype=int),
            train_indices=np.asarray(train_indices, dtype=int),
            similarity_matrix=similarity_matrix,
            activity_values=activity_values,
            similarity_threshold=self.similarityThreshold,
            activity_threshold=self.activityThreshold,
        )
        return ActivityCliffSplitter.effective_bias_from_question_results(
            question_results
        )
