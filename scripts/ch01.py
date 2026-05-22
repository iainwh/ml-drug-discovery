"""
Chapter 01: Introduction to ML for Drug Discovery
- FDA-approved drugs by USAN stem
- PCA of molecular fingerprints
- Logistic regression classification
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, PandasTools
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

CHAPTER = "ch01"
RANDOM_SEED = 41

os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}/fda_approved_drugs", exist_ok=True)

np.random.seed(RANDOM_SEED)


def setup_visualization_style():
    """Configure consistent visualization style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(sns.color_palette('PuBu'))
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16


def setup_rdkit_drawing():
    """Configure RDKit drawing settings."""
    d2d = Draw.MolDraw2DSVG(-1, -1)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
    dopts.setHighlightColour((.635, .0, .145, .4))
    dopts.baseFontSize = 1.0
    dopts.additionalAtomLabelPadding = 0.15
    return dopts


def create_molecule_from_smiles(smiles):
    """Convert a SMILES string to an RDKit molecule object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Warning: Could not parse SMILES string: {smiles}")
    return mol


def display_molecule_properties(mol):
    """Display basic properties of an RDKit molecule object."""
    if mol is None:
        print("No valid molecule to display")
        return

    print(f"Chemical Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"Number of atoms: {mol.GetNumAtoms()}")
    print(f"Number of bonds: {mol.GetNumBonds()}")
    print(f"Number of rings: {Chem.rdMolDescriptors.CalcNumRings(mol)}")

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    print(f"Molecular Weight: {mw:.2f}")
    print(f"LogP: {logp:.2f}")
    print(f"Topological Polar Surface Area: {tpsa:.2f}")


def save_molecule_image(img, filename):
    """Save a molecule SVG image to a file."""
    with open(filename, "w") as f:
        f.write(img.data)
    print(f"Image saved to {filename}")


def load_usan_stem_data(stems, data_path=f"data/{CHAPTER}/fda_approved_drugs"):
    """Load and combine data for multiple USAN stems."""
    os.makedirs(data_path, exist_ok=True)

    dfs = []
    for stem in stems:
        try:
            df = pd.read_csv(
                f'{data_path}/{stem}.csv',
                sep=';',
                usecols=['Name', 'Smiles', '#RO5 Violations (Lipinski)'],
            )
            df['USAN Stem'] = stem
            dfs.append(df)
            print(f"Loaded data for {stem}")
        except Exception as e:
            print(f"Failed to load data for {stem}: {str(e)}")

    if not dfs:
        raise ValueError("No data was loaded. Check the file paths and formats.")

    return pd.concat(dfs)


def add_molecular_data(df, smiles_col='Smiles'):
    """Add RDKit molecule objects and fingerprints to a dataframe."""
    print("Adding RDKit molecule objects...")
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol=smiles_col, molCol='ROMol')

    valid_mol_count = df['ROMol'].notnull().sum()
    invalid_mol_count = df['ROMol'].isnull().sum()
    print(f"Valid molecules: {valid_mol_count}")
    print(f"Invalid molecules: {invalid_mol_count}")

    if invalid_mol_count > 0:
        df = df[~df['ROMol'].isnull()]
        print(f"Removed {invalid_mol_count} records with invalid molecules")

    print("Generating molecular fingerprints (ECFP6)...")
    morgan_gen = GetMorganGenerator(radius=3, fpSize=1024)
    df['ECFP6'] = df['ROMol'].apply(lambda mol: morgan_gen.GetFingerprint(mol))
    print("Fingerprint generation complete")

    return df


def perform_pca_analysis(df, fingerprint_col='ECFP6', n_components=4):
    """Perform PCA on molecular fingerprints."""
    print(f"Performing PCA with {n_components} components...")

    X = np.array([x for x in df[fingerprint_col]])

    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(X)

    for i in range(n_components):
        df[f'PC{i+1}'] = pca_result.T[i]

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("Explained variance ratio by component:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

    return df, pca


def plot_pca_results(df, group_col='USAN Stem', pc_cols=None):
    """Create visualizations of PCA results."""
    if pc_cols is None:
        pc_cols = [f'PC{i+1}' for i in range(4)]

    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(
        df,
        hue=group_col,
        vars=pc_cols,
        height=2.5,
        palette='viridis',
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'w', 'linewidth': 0.5},
        diag_kws={'alpha': 0.5, 'fill': True}
    )

    pairplot.figure.suptitle('PCA Analysis of FDA-Approved Drugs by USAN Stem',
                             y=1.02, fontsize=16)

    handles = pairplot._legend_data.values()
    labels = pairplot._legend_data.keys()

    pairplot._legend.remove()
    pairplot.figure.legend(handles=handles, labels=labels,
                           loc='center right', bbox_to_anchor=(1.15, 0.5),
                           frameon=True, fontsize=12, title='USAN Stem')

    pairplot.figure.tight_layout()
    return pairplot.fig


def prepare_classification_targets(df, stems_to_classify):
    """Create binary target variables for specific USAN stems."""
    for stem in stems_to_classify:
        stem_col = stem.replace('-', '')
        df[stem_col] = (df['USAN Stem'] == stem).astype(int)
        print(f"Created target variable '{stem_col}' with {df[stem_col].sum()} positive examples")
    return df


def train_and_visualize_classifier(df, target_col, feature_cols, title):
    """Train a logistic regression classifier and visualize its decision boundary."""
    if len(feature_cols) != 2:
        raise ValueError("Exactly 2 feature columns required for visualization")

    X = df[feature_cols].values
    y = df[target_col].values

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    contour = ax.contourf(xx, yy, Z, 25, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
    ax_c = fig.colorbar(contour)
    ax_c.set_label(f"Probability of {target_col}")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(
        X[:, 0], X[:, 1],
        c=y, s=100, cmap='Blues',
        vmin=-0.2, vmax=1.2,
        edgecolor='black', linewidth=1.5
    )

    ax.set(
        aspect='equal',
        xlabel=feature_cols[0],
        ylabel=feature_cols[1],
        title=title
    )

    fig.tight_layout()
    return model, fig


if __name__ == "__main__":
    setup_visualization_style()
    rdkit_drawing_options = setup_rdkit_drawing()

    # Demo caffeine molecule
    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    caffeine_mol = create_molecule_from_smiles(caffeine_smiles)
    display_molecule_properties(caffeine_mol)

    caffeine_img = Draw.MolsToGridImage(
        mols=[caffeine_mol],
        molsPerRow=1,
        subImgSize=(300, 300),
        legends=["Caffeine"],
        useSVG=True,
        drawOptions=rdkit_drawing_options
    )
    save_molecule_image(caffeine_img, f"figures/{CHAPTER}/caffeine.svg")

    # Load USAN stem data
    usan_stems = [
        '-caine', '-cillin', '-conazole', '-olol',
        '-oxacin', '-pine', '-sulfa', '-terol',
        '-tinib', '-vir'
    ]

    try:
        fda_approved_df = load_usan_stem_data(usan_stems)
        print(f"\nLoaded data for {len(fda_approved_df)} drugs across {len(usan_stems)} USAN stems")

        stem_counts = fda_approved_df.groupby('USAN Stem').size().sort_values(ascending=False)
        print("\nNumber of drugs per USAN stem:")
        print(stem_counts)

        fda_approved_df.to_csv(
            f'data/{CHAPTER}/fda_approved_drugs/fda_approved_drugs.csv',
            sep=',', index=False
        )
        print("Saved combined data")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise SystemExit(1)

    # Add molecular data (mol objects + fingerprints)
    fda_approved_df = add_molecular_data(fda_approved_df)

    # Visualize sample molecules per USAN stem
    sample_drugs = pd.concat(
        [fda_approved_df[fda_approved_df['USAN Stem'] == stem].head(1) for stem in usan_stems]
    )
    n_cols = min(4, len(sample_drugs))
    usan_example_img = PandasTools.FrameToGridImage(
        sample_drugs,
        legendsCol='USAN Stem',
        molsPerRow=n_cols,
        subImgSize=(250, 250),
        useSVG=True,
        drawOptions=rdkit_drawing_options
    )
    save_molecule_image(usan_example_img, f"figures/{CHAPTER}/sampled_drugs.svg")

    # PCA analysis
    fda_approved_df, pca_model = perform_pca_analysis(fda_approved_df)
    pca_fig = plot_pca_results(fda_approved_df)
    pca_fig.savefig(f'figures/{CHAPTER}/pca_pairplot.png', bbox_inches='tight', dpi=300)
    pca_fig.savefig(f'figures/{CHAPTER}/pca_pairplot.svg', bbox_inches='tight', dpi=300)
    plt.close('all')
    print(f"Saved PCA pairplot to figures/{CHAPTER}/")

    # Classification
    stems_to_classify = ['-cillin', '-olol']
    fda_approved_df = prepare_classification_targets(fda_approved_df, stems_to_classify)

    cillin_model, cillin_fig = train_and_visualize_classifier(
        df=fda_approved_df,
        target_col='cillin',
        feature_cols=['PC1', 'PC3'],
        title='Logistic Regression: -cillin Antibiotic Classification'
    )
    cillin_acc = cillin_model.score(fda_approved_df[['PC1', 'PC3']], fda_approved_df['cillin'])
    print(f"Accuracy for -cillin classification: {cillin_acc:.4f}")
    cillin_fig.savefig(f'figures/{CHAPTER}/cillin_decision_boundary.png', bbox_inches='tight', dpi=300)
    cillin_fig.savefig(f'figures/{CHAPTER}/cillin_decision_boundary.svg', bbox_inches='tight', dpi=300)
    plt.close('all')

    olol_model, olol_fig = train_and_visualize_classifier(
        df=fda_approved_df,
        target_col='olol',
        feature_cols=['PC1', 'PC3'],
        title='Logistic Regression: -olol Beta Blocker Classification'
    )
    olol_acc = olol_model.score(fda_approved_df[['PC1', 'PC3']], fda_approved_df['olol'])
    print(f"Accuracy for -olol classification: {olol_acc:.4f}")
    olol_fig.savefig(f'figures/{CHAPTER}/olol_decision_boundary.png', bbox_inches='tight', dpi=300)
    olol_fig.savefig(f'figures/{CHAPTER}/olol_decision_boundary.svg', bbox_inches='tight', dpi=300)
    plt.close('all')

    print(f"\nChapter 01 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
