"""
Chapter 08: Deep Learning for Kinase Binding Prediction
- ChEMBL EGFR kinase inhibitor data
- Scaffold splitting for train/test
- PyTorch neural network (KinaseBinderNN)
- Enrichment factor evaluation
- TensorBoard logging
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")

CHAPTER = "ch08"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)
os.makedirs(f"artifacts/{CHAPTER}/tensorboard", exist_ok=True)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def setup_visualization_style():
    """Configure visualization style."""
    colors = ["#A20025", "#6C8EBF"]
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14


def load_kinase_data(data_path=None):
    """Load EGFR kinase inhibitor data."""
    csv_path = data_path or "data/ch08/egfr_chembl_data.csv"

    if os.path.exists(csv_path):
        print(f"Loading kinase data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} compounds")
        return df

    print(f"Data not found at {csv_path}. Trying TDC...")
    try:
        from tdc.single_pred import HTS
        data = HTS(name='EGFR_ChEMBL')
        df = data.get_data()
        df = df.rename(columns={'Drug': 'smiles', 'Y': 'activity'})
        df.to_csv(csv_path, index=False)
        print(f"Loaded {len(df)} compounds from TDC, saved to {csv_path}")
        return df
    except Exception as e:
        print(f"TDC loading failed: {e}")
        print("Creating synthetic demo data...")

        np.random.seed(RANDOM_SEED)
        n = 500
        demo_smiles = ['CC(=O)Nc1ccc(O)cc1', 'c1ccc2ccccc2c1', 'c1ccccc1C(=O)O',
                       'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CC1=CC=C(C=C1)S(=O)(=O)N'] * 100

        df = pd.DataFrame({
            'smiles': demo_smiles[:n],
            'activity': np.random.randint(0, 2, n)
        })
        return df


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def scaffold_split(df, smiles_col='smiles', test_frac=0.2, val_frac=0.1):
    """Split dataset by Murcko scaffold to prevent data leakage."""
    print("Performing scaffold split...")

    scaffolds = defaultdict = {}
    scaffold_to_indices = {}

    for idx, smiles in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            scaffold = 'invalid'
        else:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
            except Exception:
                scaffold = 'invalid'

        if scaffold not in scaffold_to_indices:
            scaffold_to_indices[scaffold] = []
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size (largest first) for better splitting
    scaffold_sizes = sorted(
        scaffold_to_indices.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    n_total = len(df)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)

    train_indices, val_indices, test_indices = [], [], []

    for scaffold, indices in scaffold_sizes:
        if len(test_indices) < n_test:
            test_indices.extend(indices)
        elif len(val_indices) < n_val:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    print(f"Scaffold split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    return (
        df.iloc[train_indices].reset_index(drop=True),
        df.iloc[val_indices].reset_index(drop=True),
        df.iloc[test_indices].reset_index(drop=True)
    )


class MoleculeDataset(Dataset):
    """PyTorch Dataset for molecular fingerprints."""

    def __init__(self, df, smiles_col='smiles', label_col='activity',
                 radius=2, n_bits=2048):
        self.fingerprints = []
        self.labels = []

        for _, row in df.iterrows():
            fp = smiles_to_fingerprint(row[smiles_col], radius=radius, n_bits=n_bits)
            if fp is not None:
                self.fingerprints.append(fp)
                self.labels.append(float(row[label_col]))

        print(f"Dataset: {len(self.fingerprints)} valid molecules")

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.fingerprints[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class KinaseBinderNN(nn.Module):
    """Neural network for kinase binder prediction."""

    def __init__(self, input_dim=2048, hidden_dims=(1024, 512, 256), dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def create_data_loaders(train_df, val_df, test_df, batch_size=128, n_bits=2048):
    """Create PyTorch DataLoaders."""
    train_dataset = MoleculeDataset(train_df, n_bits=n_bits)
    val_dataset = MoleculeDataset(val_df, n_bits=n_bits)
    test_dataset = MoleculeDataset(test_df, n_bits=n_bits)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def visualize_activation_functions():
    """Plot common activation functions."""
    x = torch.linspace(-3, 3, 200)

    activations = {
        'ReLU': F.relu,
        'Sigmoid': torch.sigmoid,
        'Tanh': torch.tanh,
        'LeakyReLU': lambda x: F.leaky_relu(x, 0.1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (name, func) in zip(axes, activations.items()):
        y = func(x).numpy()
        ax.plot(x.numpy(), y, color='#6C8EBF', linewidth=2)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"figures/{CHAPTER}/activation_functions.png", bbox_inches='tight', dpi=300)
    plt.close('all')


def train_model(model, train_loader, val_loader, n_epochs=50, lr=0.001,
                log_dir=None):
    """Train the neural network model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)

    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    best_val_auc = 0.0
    best_model_state = None

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for fps, labels in train_loader:
            fps, labels = fps.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(fps)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for fps, labels in val_loader:
                fps, labels = fps.to(device), labels.to(device)
                outputs = model(fps)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            val_auc = 0.5

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()

        scheduler.step(val_loss)

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('AUC/val', val_auc, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")

    if writer:
        writer.close()

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    return history


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for fps, labels in test_loader:
            fps = fps.to(device)
            outputs = model(fps)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc_roc = roc_auc_score(all_labels, all_probs)
    auc_pr = average_precision_score(all_labels, all_probs)
    y_pred = (all_probs >= 0.5).astype(int)

    print(f"Test ROC-AUC: {auc_roc:.4f}")
    print(f"Test PR-AUC:  {auc_pr:.4f}")

    return {
        'roc_auc': auc_roc, 'pr_auc': auc_pr,
        'y_prob': all_probs, 'y_pred': y_pred, 'y_true': all_labels
    }


def calculate_enrichment_factors(y_true, y_prob, fractions=(0.01, 0.05, 0.1)):
    """Calculate enrichment factors at various fractions."""
    n_total = len(y_true)
    n_actives = y_true.sum()

    if n_actives == 0:
        return {}

    baseline_rate = n_actives / n_total

    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]

    ef_results = {}
    for frac in fractions:
        n_fraction = int(n_total * frac)
        if n_fraction == 0:
            continue
        n_actives_in_fraction = y_true_sorted[:n_fraction].sum()
        ef = (n_actives_in_fraction / n_fraction) / baseline_rate
        ef_results[f'EF{int(frac*100)}'] = ef
        print(f"EF at {int(frac*100)}%: {ef:.2f}x")

    return ef_results


def evaluate_model_enrichment(model, test_loader, top_frac=0.1):
    """Evaluate model enrichment on test set."""
    test_results = evaluate_model(model, test_loader)
    ef_results = calculate_enrichment_factors(
        test_results['y_true'], test_results['y_prob']
    )
    return test_results, ef_results


def create_enrichment_plot(y_true, y_prob, save_path=None):
    """Create an enrichment plot."""
    n_total = len(y_true)
    sorted_indices = np.argsort(y_prob)[::-1]
    y_sorted = y_true[sorted_indices]

    cumulative_actives = np.cumsum(y_sorted)
    fractions = np.arange(1, n_total + 1) / n_total

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fractions, cumulative_actives / y_true.sum(),
            color='#6C8EBF', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel("Fraction of dataset screened")
    ax.set_ylabel("Fraction of actives found")
    ax.set_title("Enrichment Plot")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')

    return fig


if __name__ == "__main__":
    setup_visualization_style()
    visualize_activation_functions()

    # Load data
    df = load_kinase_data()
    print(f"\nDataset shape: {df.shape}")

    # Determine column names
    smiles_col = next((c for c in ['smiles', 'Drug', 'SMILES'] if c in df.columns), None)
    label_col = next((c for c in ['activity', 'Y', 'label'] if c in df.columns), None)

    if smiles_col is None or label_col is None:
        print(f"Required columns not found. Available: {list(df.columns)}")
        raise SystemExit(1)

    # Rename for consistency
    df = df.rename(columns={smiles_col: 'smiles', label_col: 'activity'})

    # Scaffold split
    train_df, val_df, test_df = scaffold_split(df, smiles_col='smiles', test_frac=0.15, val_frac=0.1)

    # Create data loaders
    n_bits = 2048
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, batch_size=64, n_bits=n_bits
    )

    # Build model
    model = KinaseBinderNN(input_dim=n_bits, hidden_dims=(512, 256, 128), dropout=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Train
    history = train_model(
        model, train_loader, val_loader,
        n_epochs=50, lr=0.001,
        log_dir=f"artifacts/{CHAPTER}/tensorboard"
    )

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history['train_loss'], label='Train', color='#6C8EBF')
    ax1.plot(history['val_loss'], label='Val', color='#A20025')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(history['val_auc'], color='#A20025')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Validation AUC')

    plt.tight_layout()
    plt.savefig(f"figures/{CHAPTER}/training_curves.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    # Save model
    model_path = f"artifacts/{CHAPTER}/kinase_binder_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': n_bits,
            'hidden_dims': (512, 256, 128),
            'dropout': 0.3
        }
    }, model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    test_results, ef_results = evaluate_model_enrichment(model, test_loader)

    # Enrichment plot
    create_enrichment_plot(
        test_results['y_true'], test_results['y_prob'],
        save_path=f"figures/{CHAPTER}/enrichment_plot.png"
    )

    # ROC curve
    fpr, tpr, _ = roc_curve(test_results['y_true'], test_results['y_prob'])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#6C8EBF', linewidth=2,
            label=f"AUC={test_results['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve - Kinase Binder NN")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{CHAPTER}/roc_curve.png", bbox_inches='tight', dpi=300)
    plt.close('all')

    print(f"\nChapter 08 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
