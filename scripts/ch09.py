"""
Chapter 09: Molecular Docking and Active Learning
- AutoDock Vina-based docking oracle
- Deep docking surrogate model
- Active learning loop: greedy, uncertainty, PI, EI acquisition functions
- Experiment management and comparison
"""

import os
import json
import glob
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

warnings.filterwarnings("ignore")

CHAPTER = "ch09"
RANDOM_SEED = 42

BASE_DIR = f"artifacts/{CHAPTER}/al_experiments"
os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

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


@dataclass
class MoleculePreparationConfig:
    """Configuration for molecule preparation."""
    ph: float = 7.4
    add_hydrogens: bool = True
    generate_3d: bool = True
    minimize_energy: bool = True
    n_conformers: int = 1
    random_seed: int = RANDOM_SEED


@dataclass
class Point:
    """3D coordinate point."""
    x: float
    y: float
    z: float


@dataclass
class Box:
    """Docking box definition."""
    center: Point
    size: Point


def load_molecules(smiles_list):
    """Load and prepare molecules for docking."""
    mols = []
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(smi)

    print(f"Loaded {len(mols)}/{len(smiles_list)} valid molecules")
    return mols, valid_smiles


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)


class DeepDockingModel(nn.Module):
    """Neural network surrogate for docking score prediction."""

    def __init__(self, input_dim=2048, hidden_dims=(512, 256, 128), dropout=0.3):
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


class GPSurrogate:
    """Gaussian Process surrogate for Bayesian optimization."""

    def __init__(self):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.gp.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X, return_std=False):
        if not self.is_fitted:
            raise ValueError("GP not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.gp.predict(X_scaled, return_std=return_std)


def train_deep_model(X_train, y_train, X_val=None, y_val=None,
                     n_epochs=50, batch_size=128, lr=0.001):
    """Train the deep docking surrogate model."""
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepDockingModel(input_dim=X_train.shape[1])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")

    return model


def initialize_samples(X_pool, y_pool, n_initial, method='random'):
    """Initialize the labeled set for active learning."""
    if method == 'random':
        indices = np.random.choice(len(X_pool), size=n_initial, replace=False)
    elif method == 'diverse':
        # Select diverse initial set using greedy max-min distance
        indices = [np.random.randint(len(X_pool))]

        for _ in range(n_initial - 1):
            dists = np.min(
                np.array([np.linalg.norm(X_pool - X_pool[idx], axis=1)
                          for idx in indices]),
                axis=0
            )
            dists[indices] = -1
            indices.append(np.argmax(dists))

        indices = np.array(indices)
    else:
        indices = np.random.choice(len(X_pool), size=n_initial, replace=False)

    mask = np.zeros(len(X_pool), dtype=bool)
    mask[indices] = True

    return mask


def greedy_acquisition(model, X_unlabeled, top_n=1, negate=True):
    """Greedy acquisition: select molecules with best predicted score."""
    with torch.no_grad():
        X_tensor = torch.tensor(X_unlabeled, dtype=torch.float32).to(device)
        if isinstance(model, DeepDockingModel):
            preds = model(X_tensor).cpu().numpy()
        else:
            preds = model.predict(X_unlabeled)

    if negate:
        preds = -preds

    indices = np.argsort(preds)[-top_n:]
    return indices


def uncertainty_sampling(model, X_unlabeled, top_n=1):
    """Uncertainty-based acquisition using GP posterior standard deviation."""
    if isinstance(model, GPSurrogate):
        _, std = model.predict(X_unlabeled, return_std=True)
    else:
        # Use Monte Carlo dropout for neural networks
        model.train()
        preds = []
        with torch.no_grad():
            X_tensor = torch.tensor(X_unlabeled, dtype=torch.float32).to(device)
            for _ in range(20):
                preds.append(model(X_tensor).cpu().numpy())
        std = np.std(preds, axis=0)
        model.eval()

    indices = np.argsort(std)[-top_n:]
    return indices


def probability_of_improvement(model, X_unlabeled, y_best, xi=0.01, top_n=1):
    """Probability of Improvement (PI) acquisition function."""
    from scipy.stats import norm

    if isinstance(model, GPSurrogate):
        mu, std = model.predict(X_unlabeled, return_std=True)
    else:
        with torch.no_grad():
            X_tensor = torch.tensor(X_unlabeled, dtype=torch.float32).to(device)
            mu = model(X_tensor).cpu().numpy()
        std = np.ones_like(mu) * 0.1  # Fallback

    Z = (mu - y_best - xi) / (std + 1e-9)
    pi = norm.cdf(Z)

    indices = np.argsort(pi)[-top_n:]
    return indices


def expected_improvement(model, X_unlabeled, y_best, xi=0.01, top_n=1):
    """Expected Improvement (EI) acquisition function."""
    from scipy.stats import norm

    if isinstance(model, GPSurrogate):
        mu, std = model.predict(X_unlabeled, return_std=True)
    else:
        with torch.no_grad():
            X_tensor = torch.tensor(X_unlabeled, dtype=torch.float32).to(device)
            mu = model(X_tensor).cpu().numpy()
        std = np.ones_like(mu) * 0.1

    improvement = mu - y_best - xi
    Z = improvement / (std + 1e-9)
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std == 0.0] = 0.0

    indices = np.argsort(ei)[-top_n:]
    return indices


def select_next_molecule(model, X_unlabeled, acquisition_fn='greedy', y_best=None,
                          budget=1):
    """Select the next molecule(s) to evaluate."""
    if acquisition_fn == 'greedy':
        return greedy_acquisition(model, X_unlabeled, top_n=budget)
    elif acquisition_fn == 'uncertainty':
        return uncertainty_sampling(model, X_unlabeled, top_n=budget)
    elif acquisition_fn == 'pi':
        return probability_of_improvement(model, X_unlabeled, y_best, top_n=budget)
    elif acquisition_fn == 'ei':
        return expected_improvement(model, X_unlabeled, y_best, top_n=budget)
    else:
        return np.random.choice(len(X_unlabeled), size=budget, replace=False)


def vina_oracle(smiles, protein_pdb=None, box=None):
    """Mock oracle for AutoDock Vina docking scores."""
    # Simplified oracle that uses molecular properties as proxy
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    from rdkit.Chem import Descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # Simulate docking score (negative values = better binding)
    score = -(5 + logp * 0.5 - hbd * 0.3 - hba * 0.2 + np.random.normal(0, 0.5))
    return score


def deepdock_oracle(smiles, model=None, scaler=None):
    """Oracle using trained deep docking model."""
    if model is None:
        return vina_oracle(smiles)

    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return 0.0

    if scaler:
        fp = scaler.transform(fp.reshape(1, -1))[0]

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
        score = model(x_tensor).item()

    return score


def active_learning_loop(X_pool, y_pool, smiles_pool, config):
    """Run active learning loop."""
    n_initial = config.get('n_initial_samples', 100)
    n_iterations = config.get('n_iterations', 10)
    budget = config.get('budget_per_iteration', 20)
    acquisition_fn = config.get('acquisition_function', 'greedy')
    init_method = config.get('initial_selection_method', 'random')

    top_reference = config.get('top_reference', set())

    # Initialize labeled set
    labeled_mask = initialize_samples(X_pool, y_pool, n_initial, method=init_method)
    unlabeled_mask = ~labeled_mask

    history = []

    for iteration in range(n_iterations):
        X_labeled = X_pool[labeled_mask]
        y_labeled = y_pool[labeled_mask]
        X_unlabeled = X_pool[unlabeled_mask]

        # Train surrogate
        surrogate = GPSurrogate() if len(X_labeled) < 500 else None
        if surrogate:
            surrogate.fit(X_labeled, y_labeled)
            model = surrogate
        else:
            model = train_deep_model(X_labeled, y_labeled, n_epochs=20)
            model.eval()

        # Compute y_best
        y_best = y_labeled.max()

        # Select next molecules
        unlabeled_indices = np.where(unlabeled_mask)[0]
        selected_local = select_next_molecule(
            model, X_unlabeled, acquisition_fn=acquisition_fn,
            y_best=y_best, budget=budget
        )
        selected_global = unlabeled_indices[selected_local]

        # Update masks
        labeled_mask[selected_global] = True
        unlabeled_mask[selected_global] = False

        # Track progress
        labeled_smiles = set(smiles_pool[labeled_mask])
        n_found = len(labeled_smiles & top_reference) if top_reference else 0
        total = labeled_mask.sum()

        history.append({
            'iteration': iteration + 1,
            'total_molecules': total,
            'top_molecules_found': n_found,
            'best_score': y_labeled.max(),
            'mean_score': y_labeled.mean()
        })

        print(f"Iteration {iteration+1}: {total} labeled, best={y_labeled.max():.3f}, "
              f"top found={n_found}")

    return pd.DataFrame(history)


class ExperimentManager:
    """Manager for running and tracking active learning experiments."""

    def __init__(self, base_dir=BASE_DIR, X_pool=None, smiles_pool=None,
                 top_reference_smiles=None, device=None, oracle=None,
                 reference_df=None):
        self.base_dir = base_dir
        self.X_pool = X_pool
        self.smiles_pool = np.array(smiles_pool) if smiles_pool is not None else None
        self.top_reference = set(top_reference_smiles) if top_reference_smiles else set()
        self.device = device or torch.device('cpu')
        self.oracle = oracle
        self.reference_df = reference_df

        os.makedirs(base_dir, exist_ok=True)

    def run_experiment(self, config, exp_id=None):
        """Run a single active learning experiment."""
        if exp_id is None:
            import hashlib
            exp_id = hashlib.md5(str(config).encode()).hexdigest()[:8]

        exp_dir = os.path.join(self.base_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        config_with_pool = config.copy()
        config_with_pool['top_reference'] = self.top_reference

        results = active_learning_loop(
            self.X_pool,
            self.oracle if self.oracle else np.random.randn(len(self.X_pool)),
            self.smiles_pool,
            config_with_pool
        )

        results_path = os.path.join(exp_dir, 'results.csv')
        results.to_csv(results_path, index=False)

        return results

    def run_comparison(self, configs, parallel=False):
        """Run multiple experiments and compare."""
        results = {}
        for i, config in enumerate(configs):
            exp_id = f"exp_{i:03d}"
            print(f"\nRunning experiment {i+1}/{len(configs)}: {exp_id}")
            try:
                result = self.run_experiment(config, exp_id=exp_id)
                results[exp_id] = result
            except Exception as e:
                print(f"Experiment {exp_id} failed: {e}")

        return results

    def load_experiments(self):
        """Load all saved experiment results."""
        results = {}
        for exp_dir in glob.glob(os.path.join(self.base_dir, "exp_*")):
            exp_id = os.path.basename(exp_dir)
            results_path = os.path.join(exp_dir, 'results.csv')
            if os.path.exists(results_path):
                results[exp_id] = pd.read_csv(results_path)

        print(f"Loaded {len(results)} experiments")
        return results

    def create_summary_table(self, results):
        """Create a summary table of experiment results."""
        summary_rows = []

        for exp_id, df in results.items():
            config_path = os.path.join(self.base_dir, exp_id, 'config.json')
            config = {}
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)

            row = config.copy()
            row['exp_id'] = exp_id

            if 'top_molecules_found' in df.columns:
                row['final_top_molecules_found'] = df['top_molecules_found'].iloc[-1]
            if 'best_score' in df.columns:
                row['final_best_score'] = df['best_score'].iloc[-1]

            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def plot_results(self, results):
        """Plot learning curves for all experiments."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for exp_id, df in results.items():
            if 'top_molecules_found' in df.columns:
                ax.plot(df['iteration'], df['top_molecules_found'],
                        marker='o', label=exp_id, alpha=0.7)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Top Molecules Found')
        ax.set_title('Active Learning Progress')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"figures/{CHAPTER}/al_comparison.png", bbox_inches='tight', dpi=300)
        plt.close('all')

        return {'learning_curves': fig}


def create_experiment_configs(param_grid):
    """Create experiment configurations from a parameter grid."""
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


def aggregate_and_sample_csvs(root_folder, output_file, n_top=5000,
                               n_bottom=5000, n_middle=40000):
    """Aggregate CSV files from iteration subfolders and sample by score."""
    csv_pattern = os.path.join(root_folder, "iteration_*", "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"No CSV files found in {root_folder}")
        return

    print(f"Found {len(csv_files)} CSV files")

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            required_columns = ["r_i_docking_score", "ZINC_ID", "SMILES"]
            if all(col in df.columns for col in required_columns):
                dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        print("No valid CSV files found")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    sorted_df = combined_df.sort_values('r_i_docking_score')

    top_entries = sorted_df.head(n_top)
    bottom_entries = sorted_df.tail(n_bottom)
    middle_df = sorted_df.iloc[n_top:-n_bottom]

    if len(middle_df) < n_middle:
        middle_sample = middle_df
    else:
        middle_sample = middle_df.sample(n=n_middle, random_state=42)

    final_df = pd.concat([top_entries, middle_sample, bottom_entries])
    final_df.to_csv(output_file, index=False)
    print(f"Saved {len(final_df)} entries to {output_file}")


if __name__ == "__main__":
    setup_visualization_style()

    # Load or generate pool of molecules
    pool_path = f"data/{CHAPTER}/molecule_pool.csv"

    if os.path.exists(pool_path):
        print(f"Loading molecule pool from {pool_path}...")
        pool_df = pd.read_csv(pool_path)
    else:
        print("Creating synthetic molecule pool...")
        from rdkit.Chem import Descriptors

        # Generate synthetic molecule data
        demo_smiles = [
            'CC(=O)Nc1ccc(O)cc1', 'c1ccc2ccccc2c1', 'CCO',
            'c1ccccc1C(=O)O', 'NC(=O)c1ccccc1',
            'CC1=CC=C(C=C1)S(=O)(=O)N', 'c1ccc(cc1)Cl',
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'OC(=O)c1ccccc1O'
        ] * 50

        pool_df = pd.DataFrame({'smiles': demo_smiles[:200]})
        pool_df['smiles'] = pool_df['smiles'].drop_duplicates()
        pool_df = pool_df.dropna().reset_index(drop=True)

        # Compute docking scores using oracle
        print("Computing oracle scores...")
        pool_df['docking_score'] = pool_df['smiles'].apply(
            lambda s: vina_oracle(s) + np.random.normal(0, 0.2)
        )

        pool_df.to_csv(pool_path, index=False)
        print(f"Saved molecule pool to {pool_path}")

    print(f"Pool size: {len(pool_df)}")

    # Generate fingerprints
    smiles_list = pool_df['smiles'].tolist()
    fingerprints = []
    valid_smiles = []

    for smi in tqdm(smiles_list, desc="Computing fingerprints"):
        fp = smiles_to_fingerprint(smi, n_bits=512)
        if fp is not None:
            fingerprints.append(fp)
            valid_smiles.append(smi)

    X_pool = np.array(fingerprints)
    smiles_pool = np.array(valid_smiles)

    # Get oracle scores for pool
    y_pool = np.array([
        pool_df.loc[pool_df['smiles'] == smi, 'docking_score'].values[0]
        if smi in pool_df['smiles'].values else vina_oracle(smi)
        for smi in valid_smiles
    ])

    # Identify top molecules as reference
    top_n_ref = max(10, int(len(y_pool) * 0.05))
    top_indices = np.argsort(y_pool)[-top_n_ref:]
    top_reference_smiles = set(smiles_pool[top_indices])
    print(f"Reference top molecules: {len(top_reference_smiles)}")

    # Define experiment parameter grid
    param_grid = {
        'n_initial_samples': [20, 50],
        'n_iterations': [5],
        'budget_per_iteration': [5, 10],
        'initial_selection_method': ['random'],
        'acquisition_function': ['greedy', 'ei']
    }

    configs = create_experiment_configs(param_grid)
    print(f"Created {len(configs)} experiment configurations")

    # Initialize experiment manager
    manager = ExperimentManager(
        base_dir=BASE_DIR,
        X_pool=X_pool,
        smiles_pool=smiles_pool,
        top_reference_smiles=top_reference_smiles,
        device=device,
        oracle=y_pool
    )

    # Run experiments
    print("Starting experiments...")
    results = manager.run_comparison(configs, parallel=False)

    # Visualize
    plots = manager.plot_results(results)

    # Summary table
    summary = manager.create_summary_table(results)
    summary.to_csv(f"artifacts/{CHAPTER}/al_experiments/summary_table.csv", index=False)
    print("\nSummary:")
    print(summary.to_string())

    print(f"\nChapter 09 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
