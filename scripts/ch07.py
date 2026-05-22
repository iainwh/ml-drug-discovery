"""
Chapter 07: Drug Repurposing with SOMs, UMAP, Clustering & Library Generation
- Self-Organizing Maps (MiniSom) for molecular fingerprints
- UMAP/PCA dimensionality reduction
- Taylor-Butina clustering, hierarchical clustering
- BRICS fragmentation and combinatorial library generation
- Pharmacophore scoring with KDE
"""

import os
import warnings
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import umap

from minisom import MiniSom

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, BRICS, rdMolDescriptors
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
try:
    from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatures
    from rdkit.Chem import ChemicalFeatures
    CHEM_FEATURES = ChemicalFeatures.BuildFeatureFactory(
        os.path.join(os.environ.get('RDBASE', ''), 'Data/BaseFeatures.fdef')
    ) if os.path.exists(os.path.join(os.environ.get('RDBASE', ''), 'Data/BaseFeatures.fdef')) else None
except Exception:
    CHEM_FEATURES = None

warnings.filterwarnings("ignore")

CHAPTER = "ch07"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def setup_visualization_style():
    """Configure visualization style."""
    colors = ["#A20025", "#6C8EBF"]
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14


def save_figure(fig, filename, dpi=300):
    """Save a figure to the figures directory."""
    path = f"figures/{CHAPTER}/{filename}.png"
    if hasattr(fig, 'savefig'):
        fig.savefig(path, bbox_inches='tight', dpi=dpi)
    else:
        plt.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close('all')
    print(f"Saved: {path}")


def load_and_process_activity_data(data_path, smiles_col='SMILES', activity_col='activity'):
    """Load and process activity data for drug repurposing."""
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} compounds from {data_path}")
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return None

    df['mol'] = df[smiles_col].apply(
        lambda s: Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
    )
    df = df[df['mol'].notna()].copy()
    print(f"Valid molecules: {len(df)}")

    return df


def generate_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """Generate Morgan fingerprint for a molecule."""
    if mol is None:
        return None
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return morgan_gen.GetFingerprint(mol)


def hill_equation(x, ec50, hill_coeff, baseline=0, top=100):
    """Hill equation for dose-response curve fitting."""
    return baseline + (top - baseline) * (x**hill_coeff) / (ec50**hill_coeff + x**hill_coeff)


def fit_and_plot_dose_response(concentrations, responses, compound_name="compound",
                                save_path=None):
    """Fit and plot a dose-response curve."""
    from scipy.optimize import curve_fit

    try:
        popt, _ = curve_fit(
            hill_equation, concentrations, responses,
            p0=[1.0, 1.0, 0, 100],
            maxfev=5000,
            bounds=([0, 0.1, -50, 50], [100, 5, 50, 150])
        )
        ec50, hill, baseline, top = popt
        print(f"{compound_name}: EC50={ec50:.3f}, Hill={hill:.2f}")
    except Exception as e:
        print(f"Curve fitting failed for {compound_name}: {e}")
        popt = [1.0, 1.0, 0, 100]
        ec50 = None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(concentrations, responses, color='#A20025', s=60, label='Data', zorder=5)

    if ec50 is not None:
        x_fit = np.logspace(
            np.log10(min(concentrations) * 0.1),
            np.log10(max(concentrations) * 10),
            100
        )
        y_fit = hill_equation(x_fit, *popt)
        ax.semilogx(x_fit, y_fit, color='#6C8EBF', linewidth=2, label=f'Fit (EC50={ec50:.3f})')

    ax.set_xlabel('Concentration')
    ax.set_ylabel('Response (%)')
    ax.set_title(f'Dose-Response: {compound_name}')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')

    return ec50, fig


def train_and_evaluate_som(fingerprints, som_size=(10, 10), sigma=1.0, lr=0.5, n_iter=1000):
    """Train a Self-Organizing Map on molecular fingerprints."""
    n_features = fingerprints[0].GetNumBits() if hasattr(fingerprints[0], 'GetNumBits') else fingerprints.shape[1]

    # Convert to numpy if needed
    if not isinstance(fingerprints, np.ndarray):
        fps_array = np.array([list(fp) for fp in fingerprints], dtype=float)
    else:
        fps_array = fingerprints

    print(f"Training SOM ({som_size[0]}x{som_size[1]}) on {len(fps_array)} fingerprints...")

    som = MiniSom(
        som_size[0], som_size[1], fps_array.shape[1],
        sigma=sigma, learning_rate=lr,
        random_seed=RANDOM_SEED
    )
    som.random_weights_init(fps_array)
    som.train(fps_array, n_iter, verbose=False)

    print("SOM training complete")
    return som, fps_array


def visualize_som_color_grid(som, fps_array, labels=None, title="SOM", save_path=None):
    """Visualize the SOM with activation frequency."""
    freq = np.zeros(som.get_weights().shape[:2])
    for fp in fps_array:
        winner = som.winner(fp)
        freq[winner] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.pcolor(freq.T, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')

    return fig


def identify_repurposing_candidates(som, fps_array, smiles_list, query_smiles,
                                     top_n=10):
    """Identify drug repurposing candidates using SOM similarity."""
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        print(f"Invalid query SMILES: {query_smiles}")
        return []

    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    query_fp = morgan_gen.GetFingerprintAsNumPy(query_mol).astype(float)

    query_winner = som.winner(query_fp)

    # Find molecules mapping to similar SOM cells
    winners = [som.winner(fp) for fp in fps_array]
    distances = [np.linalg.norm(np.array(w) - np.array(query_winner)) for w in winners]

    sorted_indices = np.argsort(distances)
    candidates = [(smiles_list[i], distances[i]) for i in sorted_indices[:top_n]]

    print(f"Top {top_n} repurposing candidates for query:")
    for smi, dist in candidates[:5]:
        print(f"  {smi[:50]}... (SOM distance: {dist:.2f})")

    return candidates


def compare_dimensionality_reduction_techniques(fps_array, labels=None):
    """Compare PCA and UMAP for dimensionality reduction of fingerprints."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(fps_array)

    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=10, c=labels if labels is not None else '#6C8EBF')
    axes[0].set_title(f"PCA (Var: {pca.explained_variance_ratio_.sum():.2%})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # UMAP
    reducer = umap.UMAP(random_state=RANDOM_SEED)
    X_umap = reducer.fit_transform(fps_array)

    axes[1].scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.5, s=10, c=labels if labels is not None else '#A20025')
    axes[1].set_title("UMAP")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")

    plt.tight_layout()
    save_figure(fig, "pca_vs_umap_comparison")

    return X_pca, X_umap


def butina_clustering(fps, cutoff=0.4):
    """Cluster molecules using the Butina (Taylor-Butina) algorithm."""
    from rdkit.ML.Cluster import Butina
    from rdkit import DataStructs

    # Compute pairwise Tanimoto distances
    n = len(fps)
    dists = []

    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    print(f"Clustering {n} molecules with cutoff={cutoff}...")
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    print(f"Found {len(clusters)} clusters")

    return clusters


def optimize_butina_cutoff(fps, cutoff_range=None):
    """Find optimal Butina clustering cutoff."""
    if cutoff_range is None:
        cutoff_range = np.arange(0.2, 0.8, 0.1)

    results = []
    for cutoff in cutoff_range:
        clusters = butina_clustering(fps, cutoff=cutoff)
        n_clusters = len(clusters)
        largest = max(len(c) for c in clusters) if clusters else 0
        results.append({'cutoff': cutoff, 'n_clusters': n_clusters, 'largest_cluster': largest})
        print(f"Cutoff={cutoff:.1f}: {n_clusters} clusters, largest={largest}")

    return pd.DataFrame(results)


def extract_cores_and_fragments(df, mol_col='mol', smiles_col='smiles'):
    """Extract Murcko scaffolds and BRICS fragments from molecules."""
    scaffolds = []
    for mol in tqdm(df[mol_col], desc="Extracting scaffolds"):
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaffold))
        except Exception:
            scaffolds.append(None)

    df['scaffold'] = scaffolds

    # BRICS fragmentation
    all_fragments = set()
    for mol in tqdm(df[mol_col], desc="BRICS fragmentation"):
        try:
            frags = BRICS.BRICSDecompose(mol)
            all_fragments.update(frags)
        except Exception:
            pass

    print(f"Extracted {len(set(scaffolds))} unique scaffolds")
    print(f"Extracted {len(all_fragments)} unique BRICS fragments")

    return df, list(all_fragments)


def generate_brics_library(fragments, n_molecules=100):
    """Generate a combinatorial library using BRICS."""
    fragment_mols = []
    for frag_smi in fragments:
        mol = Chem.MolFromSmiles(frag_smi)
        if mol is not None:
            fragment_mols.append(mol)

    if len(fragment_mols) < 2:
        print("Not enough valid fragments for BRICS library generation")
        return []

    print(f"Generating BRICS library from {len(fragment_mols)} fragments...")

    library_mols = []
    try:
        library_generator = BRICS.BRICSBuild(fragment_mols[:50])  # Limit for speed
        for i, mol in enumerate(library_generator):
            if i >= n_molecules:
                break
            if mol is not None:
                library_mols.append(mol)
    except Exception as e:
        print(f"BRICS build error: {e}")

    print(f"Generated {len(library_mols)} library molecules")
    return library_mols


def generate_combinatorial_library(scaffolds, substituents, n_molecules=100):
    """Generate a combinatorial library by combining scaffolds and substituents."""
    library_mols = []

    for scaffold_smi in random.sample(scaffolds, min(10, len(scaffolds))):
        scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
        if scaffold_mol is None:
            continue

        for sub_smi in random.sample(substituents, min(10, len(substituents))):
            if len(library_mols) >= n_molecules:
                break

            sub_mol = Chem.MolFromSmiles(sub_smi)
            if sub_mol is not None:
                # Simple combination: just record the SMILES pair
                library_mols.append((scaffold_smi, sub_smi))

    print(f"Generated {len(library_mols)} combinatorial library entries")
    return library_mols


def compute_pharmacophore_data(mol, mol_id=0):
    """Extract pharmacophore features from a molecule."""
    if mol is None:
        return pd.DataFrame()

    rows = []

    # Define pharmacophore types based on atom properties
    pcore_definitions = {
        'Donor': lambda atom: atom.GetAtomicNum() in [7, 8] and atom.GetTotalNumHs() > 0,
        'Acceptor': lambda atom: atom.GetAtomicNum() in [7, 8] and atom.GetTotalNumHs() == 0,
        'Aromatic': lambda atom: atom.GetIsAromatic(),
        'Hydrophobic': lambda atom: atom.GetAtomicNum() == 6 and not atom.GetIsAromatic(),
    }

    # Generate 3D conformer if needed
    mol_3d = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_3d, randomSeed=RANDOM_SEED)
        AllChem.UFFOptimizeMolecule(mol_3d)
        conf = mol_3d.GetConformer()
        has_3d = True
    except Exception:
        has_3d = False

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()

        for pcore_name, condition in pcore_definitions.items():
            if condition(atom):
                if has_3d and atom_idx < mol_3d.GetNumAtoms():
                    pos = conf.GetAtomPosition(atom_idx)
                    xyz = (pos.x, pos.y, pos.z)
                else:
                    xyz = (0.0, 0.0, 0.0)

                rows.append({
                    'pcore': pcore_name,
                    'smiles': Chem.MolToSmiles(mol),
                    'mol_id': mol_id,
                    'coord_x': xyz[0],
                    'coord_y': xyz[1],
                    'coord_z': xyz[2],
                })

    return pd.DataFrame(rows)


def fit_kernel_density(distances, cv_folds=5):
    """Fit a KDE model to distance data with cross-validated bandwidth selection."""
    if len(distances) == 0:
        return None

    distances = distances.reshape(-1, 1)
    params = {'bandwidth': np.logspace(-2, 1, 20)}

    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        params,
        cv=min(cv_folds, len(distances))
    )

    try:
        grid.fit(distances)
        kde = grid.best_estimator_
        print(f"Optimal bandwidth: {kde.bandwidth:.4f}")
        return kde
    except Exception as e:
        print(f"Error fitting KDE: {e}")
        return None


def score_molecule_with_kdes(kdes, distances_dict, pairs):
    """Score a molecule based on pharmacophore distance KDE models."""
    scores = []

    for pair in pairs:
        kde = kdes.get(pair)
        distances = distances_dict.get(pair, np.array([]))

        if kde is not None and len(distances) > 0:
            distances_2d = distances.reshape(-1, 1)
            log_densities = kde.score_samples(distances_2d)
            probabilities = np.exp(log_densities)
            scores.append(np.max(probabilities))

    if scores:
        return np.mean(np.log(np.array(scores) + 1e-10))
    return 0.0


if __name__ == "__main__":
    setup_visualization_style()

    # Try loading activity data
    data_files = [
        "data/ch07/activity_data.csv",
        "data/ch07/repurposing_data.csv",
    ]

    df = None
    for data_file in data_files:
        if os.path.exists(data_file):
            df = load_and_process_activity_data(data_file)
            if df is not None:
                break

    if df is None:
        print("No activity data found. Creating synthetic demo dataset...")
        from rdkit.Chem import RDConfig

        demo_smiles = [
            'CC(=O)Nc1ccc(O)cc1', 'c1ccc2ccccc2c1', 'CCO',
            'c1ccccc1C(=O)O', 'NC(=O)c1ccccc1',
            'CC1=CC=C(C=C1)S(=O)(=O)N', 'c1ccc(cc1)Cl',
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'OC(=O)c1ccccc1O'
        ] * 20

        df = pd.DataFrame({
            'smiles': demo_smiles[:100],
            'mol': [Chem.MolFromSmiles(s) for s in demo_smiles[:100]],
            'activity': np.random.randint(0, 2, 100)
        })
        df = df[df['mol'].notna()].copy()
        print(f"Demo dataset: {len(df)} compounds")

    # Generate fingerprints
    morgan_gen = GetMorganGenerator(radius=2, fpSize=512)
    fps = [morgan_gen.GetFingerprint(mol) for mol in df['mol']]
    fps_array = np.array([list(fp) for fp in fps], dtype=float)

    print(f"\nFingerprint matrix: {fps_array.shape}")

    # Train SOM
    som, _ = train_and_evaluate_som(fps_array, som_size=(8, 8), n_iter=500)

    visualize_som_color_grid(
        som, fps_array, title="SOM Activation Frequency",
        save_path=f"figures/{CHAPTER}/som_activation.png"
    )

    # Dimensionality reduction
    X_pca, X_umap = compare_dimensionality_reduction_techniques(fps_array)

    # BRICS fragmentation and library generation
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    mol_col = 'mol'

    df, brics_frags = extract_cores_and_fragments(df, mol_col=mol_col, smiles_col=smiles_col)

    if brics_frags:
        library = generate_brics_library(brics_frags[:50], n_molecules=50)
        print(f"Generated BRICS library with {len(library)} molecules")

    # Butina clustering
    try:
        fps_rdkit = [morgan_gen.GetFingerprint(mol) for mol in df['mol']]
        clusters = butina_clustering(fps_rdkit, cutoff=0.4)

        # Cluster size distribution
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(cluster_sizes[:30])), cluster_sizes[:30], color='#6C8EBF')
        ax.set_xlabel("Cluster Rank")
        ax.set_ylabel("Cluster Size")
        ax.set_title("Butina Cluster Size Distribution (Top 30)")
        plt.tight_layout()
        save_figure(fig, "butina_cluster_sizes")
    except Exception as e:
        print(f"Butina clustering error: {e}")

    # Pharmacophore scoring demo
    print("\nPharmacophore feature extraction demo...")
    sample_mol = df['mol'].iloc[0]
    pcore_df = compute_pharmacophore_data(sample_mol, mol_id=0)
    if not pcore_df.empty:
        print(f"Extracted {len(pcore_df)} pharmacophore features")
        print(pcore_df['pcore'].value_counts())

    # Repurposing candidate identification
    query_smi = 'CC(=O)Nc1ccc(O)cc1'  # Acetaminophen
    smiles_list = df[smiles_col].tolist()
    candidates = identify_repurposing_candidates(
        som, fps_array, smiles_list, query_smi, top_n=5
    )

    print(f"\nChapter 07 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
