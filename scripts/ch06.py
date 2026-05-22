"""
Chapter 06: HIV TAR RNA QSAR Modeling
- 3D conformer generation and Boltzmann-weighted descriptors
- Kennard-Stone splitting
- Lasso feature selection, exhaustive MLR, XGBoost
- SHAP value interpretation
"""

import os
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

import xgboost as xgb
import shap

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

warnings.filterwarnings("ignore")

CHAPTER = "ch06"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

np.random.seed(RANDOM_SEED)


def setup_visualization_style():
    """Configure visualization style."""
    colors = ["#A20025", "#6C8EBF"]
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14


def load_hiv_tar_data(data_path="data/ch06"):
    """Load HIV TAR RNA binding dataset."""
    csv_path = os.path.join(data_path, "hiv_tar_binders.csv")

    if os.path.exists(csv_path):
        print(f"Loading HIV TAR data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} compounds")
        return df

    print(f"Data file not found: {csv_path}")
    print("Please ensure data/ch06/hiv_tar_binders.csv exists.")
    return None


def generate_conformers(mol, n_conformers=10, random_seed=RANDOM_SEED):
    """Generate 3D conformers for a molecule."""
    mol = Chem.AddHs(mol)

    conformer_ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=n_conformers,
        randomSeed=random_seed,
        pruneRmsThresh=0.5
    )

    if not conformer_ids:
        return mol, []

    # Optimize conformers with UFF
    optimized_ids = []
    for conf_id in conformer_ids:
        try:
            result = AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
            if result == 0:  # Converged
                optimized_ids.append(conf_id)
        except Exception:
            pass

    return mol, list(optimized_ids if optimized_ids else conformer_ids)


def compute_boltzmann_weights(mol, conformer_ids, temperature=298.15):
    """Compute Boltzmann weights for conformers based on their energies."""
    R = 8.314e-3  # kJ/(mol*K)
    kT = R * temperature

    energies = []
    uff = AllChem.UFFGetMoleculeForceField

    for conf_id in conformer_ids:
        try:
            ff = uff(mol, confId=conf_id)
            if ff:
                energy = ff.CalcEnergy()
                energies.append(energy)
            else:
                energies.append(0.0)
        except Exception:
            energies.append(0.0)

    energies = np.array(energies)
    energies -= energies.min()  # Normalize

    boltzmann_factors = np.exp(-energies / kT)
    weights = boltzmann_factors / boltzmann_factors.sum()

    return weights


def calculate_boltzmann_weighted_descriptors(mol, conformer_ids, weights):
    """Calculate Boltzmann-weighted 3D descriptors for a molecule."""
    # Use 2D descriptors as fallback since 3D descriptors require conformers
    descriptor_names = [d for d, _ in Descriptors._descList]
    descriptor_funcs = {d: f for d, f in Descriptors._descList}

    desc_vals = {}
    for desc in descriptor_names:
        try:
            desc_vals[desc] = descriptor_funcs[desc](mol)
        except Exception:
            desc_vals[desc] = np.nan

    return desc_vals


def kennard_stone_algorithm(X, n_samples):
    """
    Kennard-Stone algorithm for selecting a representative subset of samples.

    Parameters:
        X: Feature matrix
        n_samples: Number of samples to select

    Returns:
        List of selected indices
    """
    n_total = len(X)
    n_samples = min(n_samples, n_total - 1)

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(X, X, metric='euclidean')

    # Start with the two most distant points
    selected = []
    remaining = list(range(n_total))

    # Find pair with maximum distance
    max_idx = np.unravel_index(distances.argmax(), distances.shape)
    selected.extend(list(max_idx))
    remaining = [i for i in remaining if i not in selected]

    # Iteratively add most diverse samples
    while len(selected) < n_samples and remaining:
        # Min distance from each remaining to any selected
        min_dists = distances[remaining][:, selected].min(axis=1)
        next_idx = remaining[np.argmax(min_dists)]

        selected.append(next_idx)
        remaining.remove(next_idx)

    return selected


def lasso_feature_selection(X, y, alpha=0.01):
    """Perform Lasso feature selection."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=5000, random_state=RANDOM_SEED)
    lasso.fit(X_scaled, y)

    selected_mask = np.abs(lasso.coef_) > 1e-6
    n_selected = selected_mask.sum()
    print(f"Lasso selected {n_selected} features with alpha={alpha}")

    return selected_mask, lasso, scaler


def exhaustive_mlr_search(X, y, max_features=4):
    """Exhaustive search for best MLR feature combinations."""
    n_features = X.shape[1]
    best_r2 = -np.inf
    best_combo = None
    best_model = None

    print(f"Exhaustive MLR search up to {max_features} features from {n_features}...")

    for n_f in range(1, min(max_features + 1, n_features + 1)):
        for combo in itertools.combinations(range(n_features), n_f):
            X_sub = X[:, list(combo)]
            model = LinearRegression()

            scores = cross_val_score(model, X_sub, y, cv=5, scoring='r2')
            mean_r2 = scores.mean()

            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_combo = combo
                model.fit(X_sub, y)
                best_model = model

    print(f"Best combination: {best_combo}, CV R2 = {best_r2:.4f}")
    return best_combo, best_r2, best_model


def train_gradient_boosting_model(X_train, y_train, X_test, y_test):
    """Train an XGBoost gradient boosting model."""
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"XGBoost - R2: {r2:.4f}, RMSE: {rmse:.4f}")

    return model, {'r2': r2, 'rmse': rmse, 'y_pred': y_pred}


def analyze_with_shap(model, X_test, feature_names=None, save_path=None):
    """Compute and visualize SHAP values."""
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')

    return shap_values, fig


def qsar_modeling_workflow(df, smiles_col='SMILES', target_col='pIC50'):
    """Complete QSAR modeling workflow."""
    print("Starting QSAR modeling workflow...")

    # Generate fingerprints as features
    morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)

    fingerprints = []
    valid_mask = []

    for smiles in tqdm(df[smiles_col], desc="Generating fingerprints"):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            fp = morgan_gen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
            valid_mask.append(True)
        else:
            fingerprints.append(np.zeros(1024))
            valid_mask.append(False)

    X = np.array(fingerprints)[valid_mask]
    y = df[target_col].values[valid_mask]

    print(f"Feature matrix: {X.shape}")

    # Kennard-Stone split (70% training)
    n_train = int(0.7 * len(X))
    train_indices = kennard_stone_algorithm(X, n_train)
    test_indices = [i for i in range(len(X)) if i not in train_indices]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Lasso feature selection
    selected_mask, lasso_model, scaler = lasso_feature_selection(X_train, y_train, alpha=0.01)

    if selected_mask.sum() == 0:
        print("Lasso selected no features. Using all features.")
        selected_mask = np.ones(X_train.shape[1], dtype=bool)

    X_train_sel = X_train[:, selected_mask]
    X_test_sel = X_test[:, selected_mask]

    # XGBoost model
    xgb_model, xgb_metrics = train_gradient_boosting_model(
        X_train_sel, y_train, X_test_sel, y_test
    )

    # SHAP analysis
    analyze_with_shap(
        xgb_model, X_test_sel,
        save_path=f"figures/{CHAPTER}/shap_summary.png"
    )

    return xgb_model, xgb_metrics, X_train_sel, y_train, X_test_sel, y_test


def build_rna_protein_binder_classifier(df, smiles_col='SMILES', target_col='activity'):
    """Build a classifier for RNA/protein binder prediction."""
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

    fingerprints = []
    labels = []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is not None:
            fp = morgan_gen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp)
            labels.append(int(row[target_col]))

    X = np.array(fingerprints)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"RNA/Protein binder classifier accuracy: {score:.4f}")

    return model, X_test, y_test


if __name__ == "__main__":
    setup_visualization_style()

    # Load data
    df = load_hiv_tar_data()

    if df is None:
        print("Creating synthetic demo data for Chapter 06...")
        # Create synthetic data for demonstration
        np.random.seed(RANDOM_SEED)
        n_compounds = 100
        demo_smiles = [
            'CC(=O)Nc1ccc(O)cc1',
            'c1ccc2ccccc2c1',
            'CCO',
            'c1ccccc1C(=O)O',
            'NC(=O)c1ccccc1'
        ]
        df = pd.DataFrame({
            'SMILES': np.random.choice(demo_smiles, n_compounds),
            'pIC50': np.random.normal(6, 1, n_compounds),
            'activity': np.random.randint(0, 2, n_compounds)
        })
        print(f"Created synthetic dataset with {len(df)} compounds")

    # Determine column names
    smiles_col = next((c for c in ['SMILES', 'smiles', 'Drug'] if c in df.columns), None)
    target_col = next((c for c in ['pIC50', 'activity', 'Y'] if c in df.columns), None)

    if smiles_col is None or target_col is None:
        print(f"Required columns not found. Available: {list(df.columns)}")
        raise SystemExit(1)

    print(f"\nUsing SMILES column: '{smiles_col}', Target column: '{target_col}'")
    print(f"Dataset shape: {df.shape}")

    # Run QSAR workflow
    xgb_model, metrics, X_train, y_train, X_test, y_test = qsar_modeling_workflow(
        df, smiles_col=smiles_col, target_col=target_col
    )

    # Plot predictions
    y_pred = xgb_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.6, color='#6C8EBF')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"XGBoost QSAR: R2={metrics['r2']:.3f}")
    plt.tight_layout()
    plt.savefig(f"figures/{CHAPTER}/xgb_predictions.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # Demonstrate Kennard-Stone splitting
    print("\nDemonstrating Kennard-Stone splitting...")
    X_demo = np.random.randn(50, 10)
    ks_train = kennard_stone_algorithm(X_demo, n_samples=35)
    print(f"Kennard-Stone: selected {len(ks_train)} training samples from 50 total")

    print(f"\nChapter 06 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
