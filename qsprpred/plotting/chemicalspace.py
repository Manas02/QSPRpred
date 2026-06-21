import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class PCAPlot:
    """Class to generate PCA plots for a dataset's descriptors."""

    def __init__(self, dataset):
        self.dataset = dataset

    def make(self, out_path="fingerprint_pca.jpg"):
        df = self.dataset.getDescriptors()
        numeric_df = df.select_dtypes(include=[np.number, bool]).astype(int)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(numeric_df)

        plot_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            plot_df["PC1"],
            plot_df["PC2"],
            alpha=0.7,
            edgecolors="DarkSlateGrey",
            s=50,
            c="royalblue",
        )

        ax.set_title("Fingerprint PCA Visualization")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return X_pca


class UMAPPlot:
    """Class to generate UMAP plots for a dataset's descriptors."""

    def __init__(self, dataset):
        self.dataset = dataset

    def make(self, out_path="fingerprint_umap.jpg"):
        df = self.dataset.getDescriptors()
        numeric_df = df.select_dtypes(include=[np.number, bool]).astype(int)

        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(numeric_df)

        plot_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            plot_df["UMAP1"],
            plot_df["UMAP2"],
            alpha=0.7,
            edgecolors="DarkSlateGrey",
            s=50,
            c="royalblue",
        )

        ax.set_title("Fingerprint UMAP Visualization")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return X_umap