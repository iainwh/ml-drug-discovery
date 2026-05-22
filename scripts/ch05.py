"""
Chapter 05: CYP Inhibition Multi-label Classification
- TDC CYP inhibition dataset for CYP2C19/2D6/3A4/1A2/2C9
- Morgan fingerprint featurization
- Multi-label classification with RandomForest and LogisticRegression
- Calibration, ROC/PR curves, class imbalance handling
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

warnings.filterwarnings("ignore")

CHAPTER = "ch05"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

np.random.seed(RANDOM_SEED)

CYP_TARGETS = ['CYP2C19', 'CYP2D6', 'CYP3A4', 'CYP1A2', 'CYP2C9']


def setup_visualization_style():
    """Configure visualization style."""
    colors = ["#A20025", "#6C8EBF"]
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14


def load_cyp_data(target_names=None):
    """Load CYP inhibition dataset from TDC."""
    if target_names is None:
        target_names = CYP_TARGETS

    cache_path = f"artifacts/{CHAPTER}/cyp_data.csv"

    if os.path.exists(cache_path):
        print(f"Loading cached CYP data from {cache_path}...")
        df = pd.read_csv(cache_path)
        print(f"Loaded {len(df)} compounds")
        return df

    print("Loading CYP inhibition data from TDC...")
    from tdc.multi_pred import CYP_P450

    dfs = []
    for target in target_names:
        try:
            data = CYP_P450(name=f'{target}_Substrate_Carbon-Mangels')
        except Exception:
            try:
                data = CYP_P450(name=f'{target}_Inhibition_Veith')
            except Exception as e:
                print(f"Could not load {target}: {e}")
                continue

        df = data.get_data()
        df = df.rename(columns={'Drug': 'smiles', 'Y': target})
        dfs.append(df[['smiles', target]])
        print(f"Loaded {len(df)} compounds for {target}")

    if dfs:
        from functools import reduce
        combined = reduce(lambda l, r: pd.merge(l, r, on='smiles', how='outer'), dfs)
        combined.to_csv(cache_path, index=False)
        print(f"Saved combined dataset to {cache_path}")
        return combined
    else:
        print("Could not load any CYP data.")
        return pd.DataFrame()


def process_df(df, target_cols=None):
    """Process dataframe: add mol objects and fingerprints."""
    if target_cols is None:
        target_cols = [c for c in CYP_TARGETS if c in df.columns]

    smiles_col = 'smiles' if 'smiles' in df.columns else 'Drug'

    print("Converting SMILES to molecules...")
    df['mol'] = df[smiles_col].apply(
        lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
    )

    df = df[df['mol'].notna()].copy()
    print(f"Valid molecules: {len(df)}")

    print("Generating Morgan fingerprints...")
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    df['fingerprint'] = df['mol'].apply(lambda m: morgan_gen.GetFingerprintAsNumPy(m))

    return df, target_cols


def train_and_evaluate_model(X_train, y_train, X_test, y_test, target_name,
                              model_type='rf'):
    """Train and evaluate a single-target classifier."""
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        )
    else:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)

    print(f"\n{target_name} ({model_type}):")
    print(f"  ROC-AUC: {auc_roc:.4f}")
    print(f"  PR-AUC:  {auc_pr:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Non-inhibitor', 'Inhibitor']))

    return model, {'roc_auc': auc_roc, 'pr_auc': auc_pr, 'y_pred': y_pred, 'y_prob': y_prob}


def plot_confusion_matrix(y_true, y_pred, target_name, save_path=None):
    """Plot confusion matrix for a classifier."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Neg', 'Predicted Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'])
    ax.set_title(f"Confusion Matrix: {target_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')
    return fig


def reliability_diagram(y_true, y_prob, target_name, n_bins=10, save_path=None):
    """Plot reliability diagram (calibration curve)."""
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_pred, fraction_pos, 's-', color='#6C8EBF', label='Model')
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Diagram: {target_name}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')
    return fig


def apply_calibration(model, X_cal, y_cal):
    """Apply Platt scaling calibration to a model."""
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated.fit(X_cal, y_cal)
    return calibrated


def plot_roc_pr_curves(y_true, y_prob, target_name, save_path=None):
    """Plot ROC and PR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color='#6C8EBF', linewidth=2, label=f'AUC-ROC = {auc_roc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve: {target_name}')
    ax1.legend()

    ax2.plot(recall, precision, color='#A20025', linewidth=2, label=f'AUC-PR = {auc_pr:.3f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'PR Curve: {target_name}')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')
    return fig


def handle_class_imbalance(X, y, method='smote'):
    """Handle class imbalance using oversampling or undersampling."""
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    print(f"Class distribution: Positive={pos_count}, Negative={neg_count}")

    if method == 'smote':
        sampler = SMOTE(random_state=RANDOM_SEED)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=RANDOM_SEED)
    else:
        return X, y

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    new_pos = y_resampled.sum()
    new_neg = len(y_resampled) - new_pos
    print(f"After {method}: Positive={new_pos}, Negative={new_neg}")

    return X_resampled, y_resampled


if __name__ == "__main__":
    setup_visualization_style()

    # Load data
    cyp_df = load_cyp_data()

    if cyp_df.empty:
        print("No CYP data loaded. Exiting.")
        raise SystemExit(1)

    # Process data
    cyp_df, target_cols = process_df(cyp_df)

    if not target_cols:
        print("No target columns found. Exiting.")
        raise SystemExit(1)

    # Get fingerprint matrix
    X = np.stack(cyp_df['fingerprint'].values)

    # Train/test split
    train_idx, test_idx = train_test_split(
        range(len(cyp_df)), test_size=0.2, random_state=RANDOM_SEED
    )
    X_train, X_test = X[train_idx], X[test_idx]

    all_results = {}

    for target in target_cols:
        if target not in cyp_df.columns:
            continue

        y = cyp_df[target].fillna(0).astype(int).values
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Skip if only one class in test
        if len(np.unique(y_test)) < 2:
            print(f"Skipping {target}: only one class in test set")
            continue

        print(f"\n{'='*60}")
        print(f"Training models for {target}")

        # Handle class imbalance
        X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train, method='smote')

        # Train Random Forest
        model, metrics = train_and_evaluate_model(
            X_train_bal, y_train_bal, X_test, y_test, target, model_type='rf'
        )
        all_results[target] = metrics

        # Confusion matrix
        plot_confusion_matrix(
            y_test, metrics['y_pred'], target,
            save_path=f"figures/{CHAPTER}/confusion_matrix_{target}.png"
        )

        # ROC and PR curves
        plot_roc_pr_curves(
            y_test, metrics['y_prob'], target,
            save_path=f"figures/{CHAPTER}/roc_pr_{target}.png"
        )

        # Reliability diagram
        reliability_diagram(
            y_test, metrics['y_prob'], target,
            save_path=f"figures/{CHAPTER}/reliability_{target}.png"
        )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = pd.DataFrame({
        target: {'ROC-AUC': r['roc_auc'], 'PR-AUC': r['pr_auc']}
        for target, r in all_results.items()
    }).T
    print(summary.round(4))
    summary.to_csv(f"artifacts/{CHAPTER}/cyp_model_summary.csv")

    print(f"\nChapter 05 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
