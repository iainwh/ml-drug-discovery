"""
Chapter 02: Chemical Library Screening
- Load Specs SDF library
- Lipinski RO5 filter, PAINS/BRENK filters, Glaxo structural alerts
- Morgan fingerprints and similarity search vs Malaria Box
"""

import os
import gzip
import shutil
import subprocess
import warnings
from collections import defaultdict
import heapq

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem, Draw, PandasTools, FilterCatalog,
    Descriptors, MolFromSmiles, MolFromSmarts,
)
from rdkit.Chem.rdFingerprintGenerator import AdditionalOutput, GetMorganGenerator

warnings.filterwarnings("ignore")

CHAPTER = "ch02"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

np.random.seed(RANDOM_SEED)

# Build descriptor lookup from RDKit
RDKIT_DESCRIPTORS = {desc: func for desc, func in Descriptors._descList}


def setup_visualization_style():
    """Configure consistent visualization style."""
    colors = ["#A20025", "#6C8EBF"]
    sns.set_palette(sns.color_palette(colors))
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


def prepare_data_files():
    """Prepare data files by decompressing if needed."""
    specs_sdf_path = "data/ch02/Specs.sdf"
    specs_sdf_gz_path = "data/ch02/Specs.sdf.gz"

    if os.path.exists(specs_sdf_path):
        print(f"File already exists: {specs_sdf_path}")
        return True

    if not os.path.exists(specs_sdf_gz_path):
        print(f"Compressed file not found: {specs_sdf_gz_path}")
        return False

    print(f"Decompressing {specs_sdf_gz_path}...")
    try:
        subprocess.run(['gzip', '-d', '-k', specs_sdf_gz_path], check=True)
        print(f"Successfully decompressed to {specs_sdf_path}")
        return True
    except subprocess.CalledProcessError:
        try:
            with gzip.open(specs_sdf_gz_path, 'rb') as f_in:
                with open(specs_sdf_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Successfully decompressed to {specs_sdf_path} using Python")
            return True
        except Exception as e:
            print(f"Failed to decompress file: {str(e)}")
            return False


def load_sdf_file(file_path, smiles_name='smiles', mol_col_name='mol'):
    """Load an SDF file into a pandas DataFrame using RDKit."""
    try:
        print(f"Loading SDF file: {file_path}")
        df = PandasTools.LoadSDF(
            file_path,
            smilesName=smiles_name,
            molColName=None
        )

        essential_cols = ["PUBCHEM_SUBSTANCE_ID", smiles_name]
        available_cols = [c for c in essential_cols if c in df.columns]
        df = df[available_cols]

        print("Adding RDKit molecule objects...")
        PandasTools.AddMoleculeColumnToFrame(df, smiles_name, mol_col_name)

        print(f"Loaded {len(df)} compounds")
        print(f"Valid molecules: {df[mol_col_name].notnull().sum()}")
        print(f"Invalid molecules: {df[mol_col_name].isnull().sum()}")

        return df

    except Exception as e:
        print(f"Error loading SDF file: {str(e)}")
        return None


def calculate_ro5_descriptors(df, mol_col='mol'):
    """Calculate Lipinski's Rule of Five descriptors."""
    RO5_PROPS = ['ExactMolWt', 'NumHAcceptors', 'NumHDonors', 'MolLogP']

    def compute_descriptor(mol, func_name, missing_val=None):
        if mol is None:
            return missing_val
        try:
            func = RDKIT_DESCRIPTORS[func_name]
            return func(mol)
        except Exception:
            return missing_val

    print("Calculating molecular descriptors...")
    for desc in RO5_PROPS:
        df[desc] = df[mol_col].apply(lambda x: compute_descriptor(x, desc))

    row_count_before = len(df)
    df = df.dropna(subset=RO5_PROPS)
    row_count_after = len(df)

    print(f"Removed {row_count_before - row_count_after} rows with missing descriptor values")
    print(f"Remaining molecules: {row_count_after}")

    return df


def apply_lipinski_filter(df):
    """Apply Lipinski's Rule of Five filter."""
    def count_ro5_violations(row):
        violations = 0
        if row['ExactMolWt'] > 500:
            violations += 1
        if row['MolLogP'] > 5:
            violations += 1
        if row['NumHDonors'] > 5:
            violations += 1
        if row['NumHAcceptors'] > 10:
            violations += 1
        return violations

    df['ro5_violations'] = df.apply(count_ro5_violations, axis=1)
    df['ro5_compliant'] = df['ro5_violations'] <= 1

    compliant_df = df[df['ro5_compliant']]
    violated_df = df[~df['ro5_compliant']]

    print(f"Compound library size pre-RO5 filter: {len(df)}")
    print(f"Compound library size post-RO5 filter: {len(compliant_df)}")
    print(f"Removed compounds: {len(violated_df)}")
    print(f"Percentage passing: {len(compliant_df) / len(df) * 100:.1f}%")

    return df, compliant_df, violated_df


def visualize_property_distribution(df, property_col, hue_col='ro5_compliant',
                                     title=None, x_label=None, file_prefix=None):
    """Visualize the distribution of a molecular property."""
    plt.figure(figsize=(10, 6))

    ax = sns.histplot(
        data=df,
        x=property_col,
        hue=hue_col,
        multiple="layer",
        palette=["#A20025", "#6C8EBF"],
        alpha=0.7,
        kde=True
    )

    if title:
        ax.set_title(title, fontsize=14)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Non-compliant", "RO5 Compliant"], title="Lipinski Status")

    plt.tight_layout()

    if file_prefix:
        file_path = f"figures/ch02/{file_prefix}_{property_col.lower()}.png"
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {file_path}")

    fig = plt.gcf()
    plt.close('all')
    return fig


def compare_properties_between_groups(compliant_df, violated_df, properties):
    """Compare molecular properties between two groups."""
    data = []
    for prop in properties:
        compliant_mean = compliant_df[prop].mean()
        violated_mean = violated_df[prop].mean()
        rel_change = (compliant_mean - violated_mean) / violated_mean * 100
        data.append({
            'property': prop,
            'RO5 Compliant Mols': compliant_mean,
            'RO5 Violated Mols': violated_mean,
            'rel_change': rel_change
        })
    return pd.DataFrame(data)


def visualize_property_differences(comparison_df):
    """Visualize differences in properties between two groups."""
    ordered_df = comparison_df.sort_values(by='RO5 Compliant Mols')
    my_range = range(1, len(ordered_df.index) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hlines(
        y=my_range,
        xmin=ordered_df['RO5 Violated Mols'],
        xmax=ordered_df['RO5 Compliant Mols'],
        color='#B0ABAC',
        linewidth=2
    )

    for x, y, ann in zip(ordered_df['RO5 Compliant Mols'], my_range, ordered_df['rel_change']):
        if ann < 0:
            ax.text(x - 0.1, y, f'{ann:.1f}%', ha='right', va='center', fontsize=12)
        else:
            ax.text(x + 0.1, y, f'+{ann:.1f}%', ha='left', va='center', fontsize=12)

    ax.scatter(
        ordered_df['RO5 Violated Mols'], my_range,
        s=150, color='#A20025', label="RO5 Violated Mols",
        zorder=3, edgecolor='white', linewidth=1
    )
    ax.scatter(
        ordered_df['RO5 Compliant Mols'], my_range,
        s=150, color='#6C8EBF', label="RO5 Compliant Mols",
        zorder=3, edgecolor='white', linewidth=1
    )

    ax.set_yticks(list(my_range))
    ax.set_yticklabels(ordered_df['property'], fontsize=12)
    ax.set_xlabel('Property Value', fontsize=16)
    ax.set_ylabel('Property', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(ncol=2, bbox_to_anchor=(1., 1.01), loc="lower right", frameon=False, fontsize=16)

    plt.tight_layout()
    plt.savefig('figures/ch02/relchange_ro5.png', bbox_inches='tight', dpi=300)
    plt.savefig('figures/ch02/relchange_ro5.pdf', bbox_inches='tight', dpi=300)
    plt.close('all')

    return fig


def apply_pains_brenk_filters(df, mol_col='mol'):
    """Apply PAINS and BRENK filters to a set of molecules."""
    df_copy = df.copy()

    print("Initializing PAINS and BRENK filters...")
    filter_params = FilterCatalog.FilterCatalogParams()
    filter_params.AddCatalog(filter_params.FilterCatalogs.PAINS)
    filter_params.AddCatalog(filter_params.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(filter_params)

    print("Applying filters to molecules...")
    df_copy.loc[:, 'PAINS_BRENK_match'] = df_copy[mol_col].apply(catalog.HasMatch)
    df_copy.loc[:, 'PAINS_BRENK_compliant'] = ~df_copy['PAINS_BRENK_match']

    filtered_df = df_copy[df_copy['PAINS_BRENK_compliant']]

    print(f"Compounds before PAINS/BRENK filter: {len(df_copy)}")
    print(f"Compounds failing PAINS/BRENK filter: {len(df_copy) - len(filtered_df)}")
    print(f"Compounds after PAINS/BRENK filter: {len(filtered_df)}")
    print(f"Percentage passing: {len(filtered_df) / len(df_copy) * 100:.1f}%")

    return df_copy, filtered_df


def load_glaxo_alerts():
    """Load Glaxo Wellcome structural alerts from a CSV file."""
    try:
        alerts_df = pd.read_csv("data/ch02/glaxo_structural_alerts.csv")
        alerts_df["ROMol"] = alerts_df.smarts.apply(MolFromSmarts)

        invalid_count = alerts_df.ROMol.isnull().sum()
        if invalid_count > 0:
            print(f"Warning: {invalid_count} invalid SMARTS patterns detected")
            alerts_df = alerts_df.dropna(subset=['ROMol'])

        print(f"Loaded {len(alerts_df)} Glaxo Wellcome structural alerts")
        return alerts_df

    except Exception as e:
        print(f"Error loading Glaxo alerts: {str(e)}")
        return None


def apply_glaxo_filters(df, alerts, mol_col='mol'):
    """Apply Glaxo Wellcome structural filters to a set of molecules."""
    df = df.copy()
    glaxo_sa_matches = []

    def check_glaxo_match(mol, alerts_df):
        match_ = False
        for _, alert in alerts_df.iterrows():
            if mol.HasSubstructMatch(alert.ROMol):
                glaxo_sa_matches.append({
                    "mol": mol,
                    "alert": alert.ROMol,
                    "description": alert.description,
                })
                match_ = True
        return match_

    print("Applying Glaxo Wellcome filters...")
    df.loc[:, 'GLAXO_match'] = df[mol_col].apply(check_glaxo_match, alerts_df=alerts)
    df.loc[:, 'GLAXO_compliant'] = ~df['GLAXO_match']

    filtered_df = df[df['GLAXO_compliant']].copy()

    print(f"Compounds before Glaxo filter: {len(df)}")
    print(f"Compounds failing Glaxo filter: {len(df) - len(filtered_df)}")
    print(f"Compounds after Glaxo filter: {len(filtered_df)}")
    print(f"Percentage passing: {len(filtered_df) / len(df) * 100:.1f}%")

    match_df = pd.DataFrame(glaxo_sa_matches)

    if not match_df.empty:
        print("\nTop 5 most common Glaxo alert matches:")
        print(match_df["description"].value_counts().head(5))

    return df, filtered_df, match_df


def generate_morgan_fingerprints(df, mol_col='mol', radius=2, n_bits=2048):
    """Generate Morgan fingerprints for a set of molecules."""
    print(f"Generating Morgan fingerprints (radius={radius}, n_bits={n_bits})...")

    df_copy = df.copy()
    fp_col_name = f"morgan_fp_r{radius}_b{n_bits}"

    morgan_generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    def compute_fingerprint(mol):
        if mol is None:
            return None
        return morgan_generator.GetFingerprint(mol)

    df_copy.loc[:, fp_col_name] = df_copy[mol_col].apply(compute_fingerprint)

    print(f"Fingerprints stored in column '{fp_col_name}'")
    return df_copy, fp_col_name


def compute_similarity_to_query(query_fp, fp_list):
    """Compute Tanimoto and Dice similarity between a query fp and a list."""
    tanimoto_sim = DataStructs.BulkTanimotoSimilarity(query_fp, fp_list)
    dice_sim = DataStructs.BulkDiceSimilarity(query_fp, fp_list)
    return tanimoto_sim, dice_sim


def analyze_similarity_scores(df, tanimoto_scores, dice_scores):
    """Analyze similarity scores distribution."""
    df = df.copy()
    df['tanimoto_sim'] = tanimoto_scores
    df['dice_sim'] = dice_scores

    print("Similarity Score Statistics:")
    print(f"Tanimoto - Min: {min(tanimoto_scores):.4f}, Max: {max(tanimoto_scores):.4f}, "
          f"Mean: {np.mean(tanimoto_scores):.4f}")
    print(f"Dice     - Min: {min(dice_scores):.4f}, Max: {max(dice_scores):.4f}, "
          f"Mean: {np.mean(dice_scores):.4f}")

    return df


def search_similar_compounds(query_fps, library_fps, similarity_threshold=0.65):
    """Find compounds in a library similar to any query compound (Dice similarity)."""
    matches = defaultdict(int)

    print(f"Searching for compounds with dice similarity >= {similarity_threshold}...")
    print(f"Comparing {len(query_fps)} query compounds against {len(library_fps)} library compounds")

    update_frequency = max(1, len(query_fps) // 10)

    for counter, query_fp in enumerate(query_fps, 1):
        dice_sim = DataStructs.BulkDiceSimilarity(query_fp, library_fps)

        for idx, sim in enumerate(dice_sim):
            if sim >= similarity_threshold:
                matches[idx] = max(sim, matches[idx])

        if counter % update_frequency == 0 or counter == len(query_fps):
            print(f"Progress: {counter}/{len(query_fps)} queries processed "
                  f"({counter/len(query_fps)*100:.1f}%)")

    print(f"Found {len(matches)} compounds with similarity >= {similarity_threshold}")
    return matches


def select_top_matches(matches, budget=1000):
    """Select the top matches up to a budget constraint."""
    heap = []
    for idx, sim in matches.items():
        heapq.heappush(heap, (-sim, idx))

    top_matches = []
    for _ in range(min(budget, len(heap))):
        neg_sim, idx = heapq.heappop(heap)
        top_matches.append((idx, -neg_sim))

    print(f"Selected top {len(top_matches)} compounds (budget: {budget})")
    return top_matches


def load_malaria_box():
    """Load the Malaria Box dataset."""
    try:
        malaria_df = pd.read_excel(
            "data/ch02/MalariaBox400compoundsDec2014.xls",
            usecols=["HEOS_COMPOUND_ID", "Smiles"]
        )

        PandasTools.AddMoleculeColumnToFrame(malaria_df, 'Smiles', 'mol')

        invalid_mol_count = malaria_df['mol'].isnull().sum()
        if invalid_mol_count > 0:
            print(f"Removed {invalid_mol_count} invalid molecules from Malaria Box")
            malaria_df = malaria_df[malaria_df['mol'].notnull()]

        print(f"Loaded {len(malaria_df)} valid compounds from Malaria Box")
        return malaria_df

    except Exception as e:
        print(f"Error loading Malaria Box: {str(e)}")
        return None


def visualize_similarity_extremes(df, sim_col, sim_col_2=None, n=3,
                                   rdkit_drawing_options=None):
    """Visualize molecules with highest and lowest similarity scores."""
    df_ordered = df.sort_values([sim_col], ascending=False).reset_index(drop=True)
    extremes = pd.concat([df_ordered[:n], df_ordered.dropna()[-n:]])

    if sim_col_2:
        legend_text = [f"Tani: {x:.2f}  Dice: {y:.2f}"
                       for x, y in zip(extremes[sim_col], extremes[sim_col_2])]
    else:
        legend_text = [f"{x:.2f}" for x in extremes[sim_col]]

    kwargs = {}
    if rdkit_drawing_options:
        kwargs['drawOptions'] = rdkit_drawing_options

    img = Draw.MolsToGridImage(
        extremes.mol,
        molsPerRow=n,
        legends=legend_text,
        useSVG=True,
        **kwargs
    )

    return img


if __name__ == "__main__":
    setup_visualization_style()
    rdkit_drawing_options = setup_rdkit_drawing()

    # Prepare data
    data_ready = prepare_data_files()
    if not data_ready:
        print("Warning: Could not prepare Specs.sdf. Some steps may fail.")

    # Load Specs library
    specs = load_sdf_file("data/ch02/Specs.sdf")

    if specs is None:
        print("Failed to load Specs library.")
        raise SystemExit(1)

    print(f"\nSample data preview:")
    print(specs.head(3))

    # Examine sample molecule
    sample_mol = specs['mol'].iloc[0]
    print(f"\nSample molecule - Atoms: {sample_mol.GetNumAtoms()}, Bonds: {sample_mol.GetNumBonds()}")

    sample_img = Draw.MolsToGridImage(
        mols=[sample_mol],
        molsPerRow=1,
        subImgSize=(400, 300),
        useSVG=True,
        drawOptions=rdkit_drawing_options
    )
    with open("figures/ch02/example_mol.svg", "w") as f:
        f.write(sample_img.data)

    # Calculate RO5 descriptors
    specs = calculate_ro5_descriptors(specs)

    print("\nSummary statistics for RO5 properties:")
    print(specs[['ExactMolWt', 'NumHAcceptors', 'NumHDonors', 'MolLogP']].describe())

    # Apply Lipinski filter
    specs, specs_ro5_compliant, specs_ro5_violated = apply_lipinski_filter(specs)

    # Visualize MW distribution
    visualize_property_distribution(
        specs,
        'ExactMolWt',
        title='Distribution of Molecular Weight',
        x_label='Molecular Weight (Da)',
        file_prefix='exactmolwt_dist'
    )

    # Compare properties between groups
    properties_to_compare = ['NumHAcceptors', 'NumHDonors', 'MolLogP']
    comparison_df = compare_properties_between_groups(
        specs_ro5_compliant, specs_ro5_violated, properties_to_compare
    )
    print("\nProperty comparison:")
    print(comparison_df)

    visualize_property_differences(comparison_df)

    # Apply PAINS/BRENK filters
    specs_ro5_compliant, specs_ro5_pains_brenk_compliant = apply_pains_brenk_filters(
        specs_ro5_compliant
    )

    # Load and apply Glaxo alerts
    glaxo_alerts = load_glaxo_alerts()
    if glaxo_alerts is not None:
        print(f"\nExample structural alerts:")
        print(glaxo_alerts[['description', 'smarts']].head(5))

        alerts_img = Draw.MolsToGridImage(
            mols=glaxo_alerts["ROMol"].iloc[2:8].tolist(),
            molsPerRow=3,
            legends=glaxo_alerts["description"].iloc[2:8].tolist(),
            useSVG=True,
            drawOptions=rdkit_drawing_options
        )
        with open("figures/ch02/glaxo_alerts_examples.svg", "w") as f:
            f.write(alerts_img.data)

        specs_ro5_pains_brenk_compliant, specs_filtered, glaxo_match_df = apply_glaxo_filters(
            specs_ro5_pains_brenk_compliant, glaxo_alerts
        )

        # Visualize alert matches
        if not glaxo_match_df.empty:
            match_img = Draw.MolsToGridImage(
                glaxo_match_df.mol.iloc[2:8].tolist(),
                highlightAtomLists=[
                    mol.GetSubstructMatch(alert)
                    for mol, alert in zip(
                        glaxo_match_df.mol.iloc[2:8],
                        glaxo_match_df.alert.iloc[2:8]
                    )
                ],
                molsPerRow=3,
                legends=glaxo_match_df.description.iloc[2:8].tolist(),
                useSVG=True,
                drawOptions=rdkit_drawing_options
            )
            with open("figures/ch02/glaxo_alerts_matches.svg", "w") as f:
                f.write(match_img.data)
    else:
        print("No Glaxo alerts loaded; skipping Glaxo filter step.")
        specs_filtered = specs_ro5_pains_brenk_compliant.copy()
        glaxo_match_df = pd.DataFrame()

    # Filtering summary
    original_count = len(specs)
    pains_brenk_count = len(specs_ro5_pains_brenk_compliant)
    final_count = len(specs_filtered)
    print("\nFiltering Summary:")
    print(f"Original: {original_count}")
    print(f"After RO5 + PAINS/BRENK: {pains_brenk_count} ({pains_brenk_count/original_count*100:.1f}%)")
    print(f"After Glaxo: {final_count} ({final_count/original_count*100:.1f}%)")

    # Generate fingerprints
    specs_filtered, fp_col_name = generate_morgan_fingerprints(specs_filtered)

    # Visualize fingerprint decomposition for sample molecule
    example_mol = specs_filtered['mol'].iloc[0]
    ao = AdditionalOutput()
    ao.AllocateBitInfoMap()
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp = morgan_gen.GetFingerprint(example_mol, additionalOutput=ao)
    set_bits = [idx for idx, bit in enumerate(fp) if bit]

    for i, bit_idx in enumerate(set_bits[:5]):
        try:
            bit_img = Draw.DrawMorganBit(example_mol, bit_idx, ao.GetBitInfoMap(), useSVG=True)
            with open(f"figures/ch02/fp_decomposition_bit{i}.svg", "w") as f:
                f.write(bit_img.data)
        except Exception as e:
            print(f"Could not draw bit {bit_idx}: {e}")

    # Load Malaria Box and search for similar compounds
    malaria_box = load_malaria_box()

    if malaria_box is not None:
        malaria_box, malaria_fp_col = generate_morgan_fingerprints(malaria_box)

        # Query molecule visualization
        query_idx = min(236, len(malaria_box) - 1)
        query_fp = malaria_box[malaria_fp_col].iloc[query_idx]
        query_mol = malaria_box['mol'].iloc[query_idx]

        query_img = Draw.MolsToGridImage(
            mols=[query_mol],
            molsPerRow=1,
            subImgSize=(400, 300),
            useSVG=True,
            drawOptions=rdkit_drawing_options
        )
        with open("figures/ch02/example_malaria_box_mol.svg", "w") as f:
            f.write(query_img.data)

        # Single-query similarity
        library_fps = specs_filtered[fp_col_name].tolist()
        tanimoto_scores, dice_scores = compute_similarity_to_query(query_fp, library_fps)
        specs_filtered = analyze_similarity_scores(specs_filtered, tanimoto_scores, dice_scores)

        sim_img = visualize_similarity_extremes(
            specs_filtered,
            sim_col="tanimoto_sim",
            sim_col_2="dice_sim",
            n=3,
            rdkit_drawing_options=rdkit_drawing_options
        )
        with open("figures/ch02/example_tani_dice_sim.svg", "w") as f:
            f.write(sim_img.data)

        plt.figure()
        sns.pairplot(specs_filtered[["tanimoto_sim", "dice_sim"]])
        plt.tight_layout()
        plt.savefig('figures/ch02/sim_pairplot.png', bbox_inches='tight', dpi=300)
        plt.close('all')

        # Full Malaria Box similarity search
        query_fps = malaria_box[malaria_fp_col].tolist()
        matches = search_similar_compounds(query_fps, library_fps, similarity_threshold=0.65)
        top_matches = select_top_matches(matches, budget=1000)

        if top_matches:
            hit_indices = [idx for idx, sim in top_matches]
            hit_similarities = [sim for idx, sim in top_matches]

            specs_hits = specs_filtered.iloc[hit_indices].copy()
            specs_hits['max_dice_sim'] = hit_similarities

            print(f"\nHit Statistics:")
            print(f"Mean similarity: {np.mean(hit_similarities):.4f}")
            print(f"Min/Max: {min(hit_similarities):.4f} / {max(hit_similarities):.4f}")

            output_file = "artifacts/ch02/specs_hits_to_malaria_box.csv"
            specs_hits.to_csv(
                output_file,
                columns=["PUBCHEM_SUBSTANCE_ID", "smiles", "max_dice_sim"],
                index=False
            )
            print(f"Saved {len(specs_hits)} hits to {output_file}")

            # Visualize top hits
            top_n = 6
            if len(specs_hits) >= top_n:
                top_hits = specs_hits.sort_values('max_dice_sim', ascending=False).head(top_n)
                hits_img = Draw.MolsToGridImage(
                    top_hits['mol'],
                    molsPerRow=3,
                    subImgSize=(300, 250),
                    legends=[f"Dice: {sim:.2f}" for sim in top_hits.max_dice_sim],
                    useSVG=True,
                    drawOptions=rdkit_drawing_options
                )
                with open("figures/ch02/top_hits_to_malaria_box.svg", "w") as f:
                    f.write(hits_img.data)
        else:
            print("No hits found meeting the similarity threshold")
    else:
        print("Could not load Malaria Box. Similarity searching skipped.")

    print(f"\nChapter 02 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
