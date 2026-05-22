"""
Chapter 04: Aqueous Solubility Regression
- TDC solubility dataset loading
- Molecular descriptors (RDKit)
- Linear regression, Ridge, Lasso, RANSAC, SGD regressor
- Bias-variance tradeoff, sequential feature selection
"""

import os
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso,
    SGDRegressor, RANSACRegressor
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_validate,
    learning_curve, validation_curve, StratifiedKFold,
    LearningCurveDisplay, ValidationCurveDisplay,
    RandomizedSearchCV
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Draw, Crippen, GraphDescriptors, Lipinski,
    rdMolDescriptors, MolFromSmiles, MolFromSmarts
)

warnings.filterwarnings("ignore")

CHAPTER = "ch04"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

np.random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_visualization_style():
    """Configure consistent visualization style."""
    colors = ["#A20025", "#6C8EBF"]
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


def load_solubility_data(dataset_name="Solubility_AqSolDB", split_method="scaffold"):
    """Load the solubility dataset from TDC or cached files."""
    cache_dir = f'artifacts/{CHAPTER}'
    train_path = f'{cache_dir}/{dataset_name.lower()}_train.csv'
    valid_path = f'{cache_dir}/{dataset_name.lower()}_valid.csv'
    test_path = f'{cache_dir}/{dataset_name.lower()}_test.csv'

    if all(os.path.exists(p) for p in [train_path, valid_path, test_path]):
        print(f"Loading {dataset_name} dataset from cached CSV files...")
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
    else:
        print(f"Loading {dataset_name} dataset from TDC...")
        from tdc.single_pred import ADME

        data = ADME(name=dataset_name)
        splits = data.get_split(method=split_method, seed=RANDOM_SEED)

        train_df = splits['train']
        valid_df = splits['valid']
        test_df = splits['test']

        # Rename columns to standard names
        for df in [train_df, valid_df, test_df]:
            if 'Drug' in df.columns:
                df.rename(columns={'Drug': 'smiles'}, inplace=True)
            if 'Y' in df.columns:
                df.rename(columns={'Y': 'solubility'}, inplace=True)

        train_df.to_csv(train_path, index=False)
        valid_df.to_csv(valid_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Saved datasets to {cache_dir}/")

    print(f"Training set: {len(train_df)} compounds")
    print(f"Validation set: {len(valid_df)} compounds")
    print(f"Test set: {len(test_df)} compounds")

    return train_df, valid_df, test_df


def add_molecule_objects(df, smiles_col='Drug'):
    """Add RDKit molecule objects to a dataframe."""
    if smiles_col not in df.columns:
        # Try alternative column names
        for col in ['smiles', 'SMILES', 'Drug']:
            if col in df.columns:
                smiles_col = col
                break

    print(f"Adding molecule objects from column '{smiles_col}'...")
    df['mol'] = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None)

    valid_count = df['mol'].notnull().sum()
    print(f"Valid molecules: {valid_count} / {len(df)}")

    return df


def calculate_molecular_descriptors(df, mol_col='mol'):
    """Calculate a comprehensive set of molecular descriptors."""
    descriptor_names = [desc for desc, _ in Descriptors._descList]
    descriptor_funcs = {desc: func for desc, func in Descriptors._descList}

    print(f"Calculating {len(descriptor_names)} molecular descriptors...")
    rows = []

    for mol in tqdm(df[mol_col], desc="Computing descriptors"):
        if mol is None:
            rows.append({d: np.nan for d in descriptor_names})
            continue

        desc_vals = {}
        for desc_name in descriptor_names:
            try:
                desc_vals[desc_name] = descriptor_funcs[desc_name](mol)
            except Exception:
                desc_vals[desc_name] = np.nan
        rows.append(desc_vals)

    desc_df = pd.DataFrame(rows, index=df.index)

    # Remove columns with too many missing values
    missing_thresh = 0.1
    desc_df = desc_df.loc[:, desc_df.isnull().mean() < missing_thresh]

    print(f"Descriptor matrix shape: {desc_df.shape}")
    return desc_df


def build_simple_linear_model(X_train, y_train, X_test, y_test):
    """Build and evaluate a simple linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"Simple Linear Regression:")
    print(f"  Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}, R2: {test_r2:.4f}")

    return model, {
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_r2': train_r2, 'test_r2': test_r2
    }


def tune_ridge_regression(X_train, y_train, X_val, y_val):
    """Tune Ridge regression using cross-validation."""
    alphas = np.logspace(-3, 3, 50)

    train_scores = []
    val_scores = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        train_scores.append(r2_score(y_train, model.predict(X_train)))
        val_scores.append(r2_score(y_val, model.predict(X_val)))

    best_alpha_idx = np.argmax(val_scores)
    best_alpha = alphas[best_alpha_idx]

    print(f"Best alpha: {best_alpha:.4f} (val R2: {val_scores[best_alpha_idx]:.4f})")

    best_model = Ridge(alpha=best_alpha)
    best_model.fit(X_train, y_train)

    return best_model, best_alpha, alphas, train_scores, val_scores


def train_ransac_regressor(X_train, y_train):
    """Train a RANSAC robust regressor."""
    ransac = RANSACRegressor(random_state=RANDOM_SEED, max_trials=200)
    ransac.fit(X_train, y_train)

    inlier_mask = ransac.inlier_mask_
    n_inliers = inlier_mask.sum()
    print(f"RANSAC: {n_inliers}/{len(y_train)} inliers ({100*n_inliers/len(y_train):.1f}%)")

    return ransac


def sequential_feature_selection(X_train, y_train, n_features=10):
    """Perform sequential forward feature selection."""
    base_model = Ridge(alpha=1.0)
    sfs = SequentialFeatureSelector(
        base_model,
        n_features_to_select=n_features,
        direction='forward',
        cv=5
    )

    print(f"Running sequential feature selection (selecting {n_features} features)...")
    sfs.fit(X_train, y_train)

    selected_features = sfs.get_support()
    print(f"Selected {selected_features.sum()} features")

    return sfs, selected_features


def evaluate_final_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a regression model on the test set."""
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} - Final Test Evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred': y_pred}


def compare_with_baselines(X_train, y_train, X_test, y_test):
    """Compare Ridge regression with baseline models."""
    from sklearn.dummy import DummyRegressor

    results = {}

    # Mean baseline
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    dummy_r2 = r2_score(y_test, dummy.predict(X_test))
    results['baseline_mean'] = dummy_r2

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_r2 = r2_score(y_test, ridge.predict(X_test))
    results['ridge'] = ridge_r2

    print("\nModel comparison (test R2):")
    for name, score in results.items():
        print(f"  {name}: {score:.4f}")

    return results


def plot_predictions(y_true, y_pred, title="Predicted vs Actual", save_path=None):
    """Plot predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='#6C8EBF')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel("Actual Solubility", fontsize=14)
    ax.set_ylabel("Predicted Solubility", fontsize=14)
    ax.set_title(title, fontsize=16)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f"RMSE={rmse:.3f}\nR2={r2:.3f}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.close('all')
    return fig


if __name__ == "__main__":
    setup_visualization_style()

    # Load data
    train_df, valid_df, test_df = load_solubility_data()

    # Determine SMILES column
    smiles_col = 'Drug' if 'Drug' in train_df.columns else 'smiles'
    target_col = 'Y' if 'Y' in train_df.columns else 'solubility'

    # Add molecule objects
    for df in [train_df, valid_df, test_df]:
        add_molecule_objects(df, smiles_col=smiles_col)

    # Calculate descriptors
    print("\nCalculating training descriptors...")
    train_desc = calculate_molecular_descriptors(train_df)
    print("Calculating validation descriptors...")
    valid_desc = calculate_molecular_descriptors(valid_df)
    print("Calculating test descriptors...")
    test_desc = calculate_molecular_descriptors(test_df)

    # Align columns
    common_cols = list(set(train_desc.columns) & set(valid_desc.columns) & set(test_desc.columns))
    train_desc = train_desc[common_cols]
    valid_desc = valid_desc[common_cols]
    test_desc = test_desc[common_cols]

    # Extract targets
    y_train = train_df[target_col].values
    y_valid = valid_df[target_col].values
    y_test = test_df[target_col].values

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(train_desc)
    X_valid = imputer.transform(valid_desc)
    X_test = imputer.transform(test_desc)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    # Build and evaluate models
    linear_model, linear_metrics = build_simple_linear_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    plot_predictions(
        y_test, linear_model.predict(X_test_scaled),
        title="Linear Regression: Predicted vs Actual Solubility",
        save_path=f"figures/{CHAPTER}/linear_regression_predictions.png"
    )

    # Ridge regression tuning
    ridge_model, best_alpha, alphas, train_r2s, val_r2s = tune_ridge_regression(
        X_train_scaled, y_train, X_valid_scaled, y_valid
    )

    # Plot regularization path
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(alphas, train_r2s, label='Training', color='#6C8EBF', linewidth=2)
    ax.semilogx(alphas, val_r2s, label='Validation', color='#A20025', linewidth=2)
    ax.axvline(best_alpha, color='gray', linestyle='--', label=f'Best alpha={best_alpha:.3f}')
    ax.set_xlabel('Regularization (alpha)')
    ax.set_ylabel('R2 Score')
    ax.set_title('Ridge Regression: Regularization Path')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{CHAPTER}/ridge_regularization_path.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # RANSAC
    ransac_model = train_ransac_regressor(X_train_scaled, y_train)
    evaluate_final_model(ransac_model, X_test_scaled, y_test, "RANSAC")

    # Sequential feature selection
    sfs, selected_mask = sequential_feature_selection(X_train_scaled, y_train, n_features=15)

    X_train_sfs = X_train_scaled[:, selected_mask]
    X_test_sfs = X_test_scaled[:, selected_mask]

    sfs_model = Ridge(alpha=best_alpha)
    sfs_model.fit(X_train_sfs, y_train)
    sfs_metrics = evaluate_final_model(sfs_model, X_test_sfs, y_test, "Ridge + SFS")

    plot_predictions(
        y_test, sfs_model.predict(X_test_sfs),
        title="Ridge + Feature Selection: Predicted vs Actual",
        save_path=f"figures/{CHAPTER}/ridge_sfs_predictions.png"
    )

    # Baseline comparison
    compare_with_baselines(X_train_scaled, y_train, X_test_scaled, y_test)

    print(f"\nChapter 04 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
