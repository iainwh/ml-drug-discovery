"""
Chapter 03: hERG Blocker Classification
- Load and standardize hERG blocker dataset
- Morgan fingerprint featurization
- SGD classifier with grid/random search
- Model evaluation: F1, MCC
"""

import os
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score
)
from scipy.stats import uniform as sp_rand

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.MolStandardize.rdMolStandardize import (
    Cleanup, LargestFragmentChooser, TautomerEnumerator, Uncharger
)
from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput, GetMorganGenerator

warnings.filterwarnings("ignore")

CHAPTER = "ch03"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

np.random.seed(RANDOM_SEED)


def setup_visualization_style():
    """Configure consistent visualization style."""
    colors = ["#A20025", "#6C8EBF"]
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16


def setup_rdkit_drawing():
    """Configure RDKit drawing settings."""
    d2d = Draw.MolDraw2DSVG(-1, -1)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
    dopts.setHighlightColour((.635, .0, .145, .4))
    dopts.baseFontSize = 1.0
    dopts.additionalAtomLabelPadding = 0.15
    return dopts


def load_herg_blockers_data():
    """Load the hERG blockers dataset from local file."""
    herg_blockers_path = Path("data/ch03/hERG_blockers.xlsx")

    try:
        df = pd.read_excel(
            herg_blockers_path,
            usecols="A:F",
            header=None,
            skiprows=[0, 1],
            names=["SMILES", "Name", "pIC50", "Class", "Scaffold Split", "Random Split"],
        ).head(-68)

        print(f"Successfully loaded {len(df)} compounds.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def simulate_annotation_error(df, col="pIC50", error_value=3.0):
    """Simulate annotation errors in a dataset by adding an error value."""
    return df[col] + error_value


def process_smiles(smi):
    """Process SMILES strings to create standardized molecule objects."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    mol = Cleanup(mol)
    mol = LargestFragmentChooser().choose(mol)
    mol = Uncharger().uncharge(mol)
    mol = TautomerEnumerator().Canonicalize(mol)

    return mol


def compute_fingerprint(mol, radius=2, nBits=2048):
    """Generate Morgan fingerprint for a molecule."""
    morgan_generator = GetMorganGenerator(radius=radius, fpSize=nBits)
    if mol is None:
        return None
    return morgan_generator.GetFingerprintAsNumPy(mol)


def explore_fingerprint_features(fingerprints):
    """Explore fingerprint features through visualizations."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    sns.heatmap(
        fingerprints[:50, :100],
        cbar=False,
        cmap="Blues",
        ax=ax[0]
    )
    ax[0].set_xlabel("Fingerprint Bits (first 100)", fontsize=14)
    ax[0].set_ylabel("Compounds (first 50)", fontsize=14)
    ax[0].set_title("Fingerprint Heatmap", fontsize=16)

    bit_counts = fingerprints.sum(axis=1)
    sns.histplot(
        bit_counts,
        kde=True,
        stat="density",
        kde_kws=dict(cut=3),
        edgecolor=(1, 1, 1, .4),
        color="#6C8EBF",
        ax=ax[1]
    )
    ax[1].set_xlabel("Number of bits set per molecule", fontsize=14)
    ax[1].set_ylabel("Density", fontsize=14)
    ax[1].set_title("Distribution of Fingerprint Density", fontsize=16)

    ax[1].annotate(
        f"Mean bits per molecule: {bit_counts.mean():.1f}\n"
        f"Min bits: {bit_counts.min()}\n"
        f"Max bits: {bit_counts.max()}",
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig("figures/ch03/fingerprint_eda.svg", bbox_inches='tight', dpi=300)
    plt.savefig("figures/ch03/fingerprint_eda.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    return fig


def split_data(df, split_col="Random Split", train_pattern="Train", test_pattern="Test"):
    """Split data into training and testing sets based on a split column."""
    train_mask = df[split_col].str.contains(train_pattern)
    test_mask = df[split_col].str.contains(test_pattern)

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Split data into {len(train_df)} training and {len(test_df)} testing examples")
    return train_df, test_df


def train_sgd_classifier(X_train, y_train, **kwargs):
    """Train a Stochastic Gradient Descent classifier."""
    params = {
        'random_state': RANDOM_SEED,
        'max_iter': 1000,
        'tol': 1e-3
    }
    params.update(kwargs)
    clf = SGDClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def visualize_cv_results(cv_df):
    """Visualize cross-validation results."""
    metrics = [col.split('_', 1)[1] for col in cv_df.columns if col.startswith('test_')]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    fig.suptitle("Discrepancy between Train & Test Performance", fontsize=16)

    ylim = (0.5, 1.0)

    for i, metric in enumerate(metrics):
        ax = axes[i]

        train_scores = cv_df[f'train_{metric}']
        test_scores = cv_df[f'test_{metric}']

        x = range(1, len(train_scores) + 1)
        ax.plot(x, train_scores, 'o-', label='Training', color='#6C8EBF', linewidth=3, markersize=8)
        ax.plot(x, test_scores, 'o-', label='Validation', color='#A20025', linewidth=3, markersize=8)

        ax.set_title(metric.replace('_', ' ').title(), fontsize=18)
        ax.set_xlabel('CV Trials', fontsize=18)
        if i == 0:
            ax.set_ylabel('Score', fontsize=18)

        ax.set_xticks(list(x))
        ax.set_ylim(ylim)

        if i == 0:
            ax.legend(fontsize=18)

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig('figures/ch03/cv_metrics_discrepancy.svg', bbox_inches='tight', dpi=300)
    plt.savefig('figures/ch03/cv_metrics_discrepancy.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    return fig


def visualize_model_weights(model):
    """Visualize the distribution of weights in a linear model."""
    if hasattr(model, 'named_steps') and 'sgdclassifier' in model.named_steps:
        weights = model.named_steps['sgdclassifier'].coef_.squeeze()
    else:
        weights = model.coef_.squeeze()

    plt.figure(figsize=(10, 6))
    sns.histplot(
        weights,
        kde=True,
        stat="density",
        kde_kws=dict(cut=3),
        edgecolor=(1, 1, 1, .4),
        color="#A20025",
    )

    plt.annotate(
        f"Mean weight: {weights.mean():.4f}\n"
        f"Std. dev: {weights.std():.4f}\n"
        f"Min weight: {weights.min():.4f}\n"
        f"Max weight: {weights.max():.4f}\n"
        f"Non-zero weights: {np.count_nonzero(weights)}/{len(weights)}",
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    plt.axvline(x=0, color='#6C8EBF', linestyle='--')
    plt.xlabel("Model Weights")
    plt.ylabel("Density")
    plt.title("Distribution of Linear Model Weights")

    plt.tight_layout()
    plt.savefig('figures/ch03/model_weights_displot.svg', bbox_inches='tight', dpi=300)
    plt.savefig('figures/ch03/model_weights_displot.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    return plt.gcf()


class SmilesToMols(BaseEstimator, TransformerMixin):
    """Transformer that converts SMILES strings to RDKit molecules."""

    def fit(self, X, y=None):
        return self

    def _process_smiles(self, smi):
        """Process a SMILES string to create a standardized molecule."""
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            mol = Cleanup(mol)
            mol = LargestFragmentChooser().choose(mol)
            mol = Uncharger().uncharge(mol)
            mol = TautomerEnumerator().Canonicalize(mol)
            return mol
        except Exception as e:
            print(f"Error processing SMILES {smi}: {str(e)}")
            return None

    def transform(self, X, y=None):
        """Transform SMILES strings to RDKit molecules."""
        print("Converting SMILES to molecules...")
        mols = [self._process_smiles(smi) for smi in X]
        return np.asarray(mols, dtype=object)


class FingerprintFeaturizer(BaseEstimator, TransformerMixin):
    """Transformer that converts RDKit molecules to fingerprints."""

    def __init__(self, radius=2, nBits=2048):
        self.radius = radius
        self.nBits = nBits

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Transform molecules to fingerprints."""
        print(f"Generating fingerprints (radius={self.radius}, nBits={self.nBits})...")

        def compute_fp(mol):
            if mol is None:
                return np.zeros(self.nBits, dtype=np.int8)
            morgan_generator = GetMorganGenerator(radius=self.radius, fpSize=self.nBits)
            return morgan_generator.GetFingerprintAsNumPy(mol)

        fingerprints = [compute_fp(mol) for mol in X]
        return np.vstack(fingerprints)


def tune_model_hyperparameters(X, y, cv=5):
    """Tune model hyperparameters using grid search."""
    cachedir = "gs_linear_sgd"

    pipe = make_pipeline(
        SmilesToMols(),
        FingerprintFeaturizer(),
        SGDClassifier(random_state=RANDOM_SEED),
        memory=cachedir
    )

    param_grid = [{
        'sgdclassifier__penalty': ["l2", "l1"],
        'sgdclassifier__alpha': [1e-3, 1e-2, 1e-1],
        'fingerprintfeaturizer__radius': [2, 4],
        'fingerprintfeaturizer__nBits': [1024, 2048],
    }]

    print("Performing grid search for hyperparameter tuning...")
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        verbose=1,
    )

    search.fit(X, y)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best score: {search.best_score_:.4f}")

    cv_results_df = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, cv_results_df


def tune_model_hyperparameters_randomized_search(X, y, n_iter=20, cv=5):
    """Tune model hyperparameters using randomized search."""
    cachedir = "rs_nonlinear_sgd"

    pipe = make_pipeline(
        FingerprintFeaturizer(),
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
        SGDClassifier(max_iter=5000, random_state=RANDOM_SEED, early_stopping=True),
        memory=cachedir
    )

    param_dist = [{
        'sgdclassifier__penalty': ["l2", "l1"],
        'sgdclassifier__alpha': sp_rand(0.01),
        'fingerprintfeaturizer__radius': [2, 4],
        'fingerprintfeaturizer__nBits': [64, 128, 256],
    }]

    print("Performing randomized search for hyperparameter tuning...")
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        scoring="f1",
        cv=cv,
        n_iter=n_iter,
    )

    search.fit(X, y)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best score: {search.best_score_:.4f}")

    cv_results_df = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, cv_results_df


def draw_fragment_from_bit(mol, bit_number):
    """Given an rdkit mol, draw the local fragment for the set bit."""
    ao = AdditionalOutput()
    ao.AllocateBitInfoMap()
    morgan_generator = GetMorganGenerator(radius=2, fpSize=2048)
    fp = morgan_generator.GetFingerprint(mol, additionalOutput=ao)
    try:
        svg = Draw.DrawMorganBit(mol, bit_number, ao.GetBitInfoMap(), useSVG=True)
    except Exception:
        raise ValueError(f"Featurization of mol doesn't have bit {bit_number} set")
    return svg


def get_examples_for_bit(bit_number, mols, fingerprints):
    """For a given bit number, get a visual representation of the substructure."""
    res = np.argwhere(fingerprints[:, bit_number] == 1)
    mols = np.array(mols)[res.flatten()]
    return [draw_fragment_from_bit(mols[i], bit_number) for i in range(len(mols))]


def evaluate_final_model(model, X_test, y_test, output_file=None):
    """Evaluate the final model on the test set."""
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    matthews_cc = matthews_corrcoef(y_test, y_pred)

    report = (
        f"Final Model Evaluation on Test Set\n"
        f"==================================\n"
        f"F1 Score (macro): {f1:.4f}\n"
        f"\nMatthews Correlation Coefficient:\n{matthews_cc:.4f}\n"
    )

    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Evaluation report saved to {output_file}")

    return {
        'f1_score': f1,
        'matthews_corrcoef': matthews_cc,
        'predictions': y_pred
    }


if __name__ == "__main__":
    setup_visualization_style()
    rdkit_drawing_options = setup_rdkit_drawing()

    # Load data
    herg_blockers = load_herg_blockers_data()
    if herg_blockers is None:
        raise SystemExit("Failed to load hERG blocker data. Ensure data/ch03/hERG_blockers.xlsx exists.")

    print("Preview of the hERG blockers dataset:")
    print(herg_blockers.head())

    # Plot pIC50 distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(
        herg_blockers["pIC50"], kde=True,
        stat="density", kde_kws=dict(cut=3),
        edgecolor=(1, 1, 1, .4),
    )
    plt.title("Distribution of pIC50 Values", fontsize=16)
    plt.xlabel("pIC50", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/ch03/distribution_pic50.svg", bbox_inches='tight', dpi=300)
    plt.savefig("figures/ch03/distribution_pic50.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # Simulate annotation error
    simulated_error = simulate_annotation_error(herg_blockers, "pIC50", 3.0)
    herg_blockers_with_error = pd.concat(
        [herg_blockers["pIC50"], simulated_error], ignore_index=True
    )
    plt.figure(figsize=(10, 6))
    sns.histplot(
        herg_blockers_with_error, kde=True,
        stat="density", kde_kws=dict(cut=3),
        edgecolor=(1, 1, 1, .4),
    )
    plt.title("Distribution of pIC50 with Simulated Annotation Error", fontsize=16)
    plt.xlabel("pIC50", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/ch03/distribution_pic50_error.svg", bbox_inches='tight', dpi=300)
    plt.savefig("figures/ch03/distribution_pic50_error.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # Standardize molecules
    herg_blockers["mol"] = herg_blockers["SMILES"].apply(process_smiles)

    invalid_mols = herg_blockers[herg_blockers["mol"].isna()]
    if len(invalid_mols) > 0:
        print(f"Warning: {len(invalid_mols)} molecules could not be standardized.")
        herg_blockers = herg_blockers.dropna(subset=["mol"])

    # Visualize before/after standardization
    before_and_after_mols = [
        Chem.MolFromSmiles(herg_blockers.iloc[1].SMILES),
        herg_blockers.iloc[1].mol,
        Chem.MolFromSmiles(herg_blockers.iloc[3].SMILES),
        herg_blockers.iloc[3].mol
    ]
    legend_text = ["Before Standardization", "After Standardization"] * 2

    img = Draw.MolsToGridImage(
        before_and_after_mols, molsPerRow=4, subImgSize=(150, 150),
        legends=legend_text, useSVG=True, drawOptions=rdkit_drawing_options,
    )
    with open("figures/ch03/rdkit_before_and_after.svg", "w") as f:
        f.write(img.data)

    # Visualize extreme molecules (highest/lowest pIC50)
    def visualize_extreme_molecules(df, activity_col="pIC50", name_col="Name",
                                     smiles_col="SMILES", n=4):
        """Visualize molecules with extreme activity values."""
        df_sorted = df.sort_values(activity_col, ascending=False)
        extremes = pd.concat([df_sorted.head(n), df_sorted.dropna().tail(n)])

        mols = [Chem.MolFromSmiles(smi) for smi in extremes[smiles_col]]
        legends = [
            f"{name}: pIC50 = {activity:.2f}"
            for name, activity in zip(extremes[name_col], extremes[activity_col])
        ]

        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=n,
            subImgSize=(250, 250),
            legends=legends,
            useSVG=True,
            drawOptions=rdkit_drawing_options
        )

        with open("figures/ch03/rdkit_extremes.svg", "w") as f:
            f.write(img.data)

        return img

    visualize_extreme_molecules(herg_blockers)

    # Compute fingerprints
    fingerprints = np.stack([compute_fingerprint(mol, 2, 2048) for mol in herg_blockers.mol])
    print(f"Fingerprint array shape: {fingerprints.shape}")

    explore_fingerprint_features(fingerprints)

    # Split data
    train_set, test_set = split_data(herg_blockers)

    train_fingerprints = np.stack([compute_fingerprint(mol, 2, 2048) for mol in train_set.mol])
    train_labels = train_set.Class

    # Train linear classifier
    lin_cls = train_sgd_classifier(train_fingerprints, train_labels)
    print("Linear model training completed")

    herg_blockers_pred = lin_cls.predict(train_fingerprints)
    print(f"Sample predictions: {herg_blockers_pred[:10]}")
    print(f"Training accuracy: {accuracy_score(train_labels, herg_blockers_pred):.4f}")

    # Cross-validation
    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision',
        'rec_macro': 'recall',
        'f1_macro': 'f1',
    }
    lin_cls_scores = cross_validate(
        lin_cls, train_fingerprints, train_labels,
        scoring=scoring, cv=5, return_train_score=True
    )
    lin_cls_scores_df = pd.DataFrame.from_dict(lin_cls_scores)
    print("Cross-validation results:")
    print(lin_cls_scores_df.describe().round(3))

    visualize_cv_results(lin_cls_scores_df)
    visualize_model_weights(lin_cls)

    # Grid search hyperparameter tuning
    X_smiles = herg_blockers['SMILES'].values
    y_labels = herg_blockers['Class'].values

    best_model_gs, cv_results = tune_model_hyperparameters(X_smiles, y_labels)
    joblib.dump(best_model_gs, "artifacts/ch03/herg_blockers_cls_model.pkl")
    print("Best GS model saved to 'artifacts/ch03/herg_blockers_cls_model.pkl'")

    top_results = cv_results.sort_values('mean_test_score', ascending=False).head(5)
    print("Top 5 GS parameter combinations:")
    print(top_results[['params', 'mean_test_score', 'std_test_score']].to_string())

    # Randomized search
    best_model_rs, cv_results_rs = tune_model_hyperparameters_randomized_search(
        train_set.mol, train_set.Class
    )
    joblib.dump(best_model_rs, "artifacts/ch03/herg_blockers_cls_model_rs.pkl")
    print("Best RS model saved to 'artifacts/ch03/herg_blockers_cls_model_rs.pkl'")

    # Weight comparison plot
    best_model_gs_weights = best_model_gs.named_steps["sgdclassifier"].coef_.squeeze()
    best_model_rs_weights = best_model_rs.named_steps["sgdclassifier"].coef_.squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, weights, title in zip(
        axes,
        [best_model_gs_weights, best_model_rs_weights],
        ["Grid Search Model Weights", "Randomized Search Model Weights"]
    ):
        sns.histplot(
            weights, kde=True, stat="density", kde_kws=dict(cut=3),
            edgecolor=(1, 1, 1, .4), ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("Regularized Model Weights")
        ax.annotate(
            f"Mean weight: {weights.mean():.4f}\n"
            f"Std. dev: {weights.std():.4f}\n"
            f"Non-zero: {np.count_nonzero(weights)}/{len(weights)}",
            xy=(0.95, 0.95), xycoords='axes fraction',
            ha='right', va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    plt.tight_layout()
    plt.savefig("figures/ch03/regularized_model_weights_comparison.svg", bbox_inches='tight', dpi=300)
    plt.savefig("figures/ch03/regularized_model_weights_comparison.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # Important bits visualization
    important_bits = np.argwhere(best_model_gs_weights).flatten()
    print(f"Number of non-zero weights: {len(important_bits)}")

    N = 3
    top_ind_unsorted = np.argpartition(best_model_gs_weights, -N)[-N:]
    top_ind_sorted = top_ind_unsorted[np.argsort(best_model_gs_weights[top_ind_unsorted])[::-1]]
    bot_ind_unsorted = np.argpartition(best_model_gs_weights, N)[:N]
    bot_ind_sorted = bot_ind_unsorted[np.argsort(best_model_gs_weights[bot_ind_unsorted])]

    for i, bit in enumerate(top_ind_sorted):
        try:
            examples = get_examples_for_bit(bit, train_set.mol, train_fingerprints)
            if examples:
                with open(f"figures/ch03/top_example_bit{i}.svg", "w") as f:
                    f.write(examples[0].data)
        except Exception as e:
            print(f"Could not visualize top bit {bit}: {e}")

    for i, bit in enumerate(bot_ind_sorted):
        try:
            examples = get_examples_for_bit(bit, train_set.mol, train_fingerprints)
            if examples:
                with open(f"figures/ch03/bot_example_bit{i}.svg", "w") as f:
                    f.write(examples[0].data)
        except Exception as e:
            print(f"Could not visualize bot bit {bit}: {e}")

    # Final evaluation on test set
    X_test_smiles = test_set['SMILES'].values
    y_test = test_set['Class'].values

    final_metrics = evaluate_final_model(
        best_model_gs,
        X_test_smiles,
        y_test,
        output_file="artifacts/ch03/final_model_evaluation.txt"
    )

    print(f"\nChapter 03 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
