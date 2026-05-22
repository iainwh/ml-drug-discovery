"""
Chapter 11: Drug-Target Affinity Prediction with Graph Neural Networks
- Molecular graph construction (78-dim atom features)
- Protein graph construction (PSSM + biochemical properties)
- GCN, GAT, and GIN dual-branch architectures (DGraphDTA)
- Davis and KIBA dataset loading
- Architecture comparison and evaluation
"""

import os
import json
import pickle
import random
import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from scipy import stats
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data, InMemoryDataset, DataLoader, Batch
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import networkx as nx

warnings.filterwarnings('ignore')

CHAPTER = "ch11"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}/models", exist_ok=True)
os.makedirs(f"artifacts/{CHAPTER}/results", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)


def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)

CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_features_mol': 78,
    'num_features_pro': 54,
    'num_features_pro_simplified': 33,
    'output_dim': 128,
    'dropout': 0.2,
    'batch_size': 512,
    'learning_rate': 0.001,
    'epochs': 2000,
    'log_interval': 10,
    'demo_samples': 100,
    'demo_train_ratio': 0.8,
    'demo_epochs': 30,
    'demo_batch_size': 16,
    'use_full_dataset': False,
    'max_samples': 1000,
    'contact_map_threshold': 0.5,
    'pssm_pseudocount': 0.8,
    'data_dir': f'data/{CHAPTER}',
    'model_dir': f'artifacts/{CHAPTER}/models',
    'results_dir': f'artifacts/{CHAPTER}/results',
    'gat_heads': 2,
}

device = torch.device(CONFIG['device'])

# Atom symbols for one-hot encoding (44 types)
ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'X'
]

# Amino acid symbols (21 types including 'X' for unknown)
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
]

ALIPHATIC = ['A', 'I', 'L', 'M', 'V']
AROMATIC = ['F', 'W', 'Y']
POLAR_NEUTRAL = ['C', 'N', 'Q', 'S', 'T']
ACIDIC_CHARGED = ['D', 'E']
BASIC_CHARGED = ['H', 'K', 'R']

RES_WEIGHT_RAW = {
    'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18,
    'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.18, 'L': 113.16,
    'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19,
    'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18
}

RES_PKA_RAW = {
    'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83,
    'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18, 'L': 2.36,
    'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17,
    'S': 2.21, 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32
}

RES_PKB_RAW = {
    'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13,
    'G': 9.60, 'H': 9.17, 'I': 9.60, 'K': 8.95, 'L': 9.60,
    'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13, 'R': 9.04,
    'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62
}

RES_PKX_RAW = {
    'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00,
    'G': 0.00, 'H': 6.00, 'I': 0.00, 'K': 10.53, 'L': 0.00,
    'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00, 'R': 12.48,
    'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00
}

RES_PI_RAW = {
    'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48,
    'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98,
    'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76,
    'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96
}

RES_HYDROPHOBIC_PH2_RAW = {
    'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0,
    'H': -42, 'I': 100, 'K': -37, 'L': 100, 'M': 74, 'N': -41,
    'P': -46, 'Q': -18, 'R': -26, 'S': -7, 'T': 13, 'V': 79,
    'W': 84, 'Y': 49
}

RES_HYDROPHOBIC_PH7_RAW = {
    'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0,
    'H': 8, 'I': 99, 'K': -23, 'L': 97, 'M': 74, 'N': -28,
    'P': -46, 'Q': -10, 'R': -14, 'S': -5, 'T': 13, 'V': 76,
    'W': 97, 'Y': 63
}


def one_of_k_encoding(x, allowable_set):
    """One-hot encoding; raises if x not in allowable_set."""
    if x not in allowable_set:
        raise Exception(f'input {x} not in allowable set{allowable_set}')
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """One-hot encoding with unknown handling (maps unknown to last element)."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom) -> np.ndarray:
    """
    Generate 78-dimensional feature vector for a single atom.
    - 44 features: atom type (one-hot)
    - 11 features: degree (0-10)
    - 11 features: total hydrogens (0-10)
    - 11 features: implicit valence (0-10)
    - 1 feature: aromaticity
    """
    features = []
    features += one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
    features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features += [int(atom.GetIsAromatic())]
    return np.array(features, dtype=np.float32)


def smile_to_graph(smiles: str) -> Tuple[int, List, List]:
    """
    Convert SMILES string to graph representation.
    Returns: (num_atoms, features_list, edge_index_list)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    num_atoms = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / (sum(feature) + 1e-10))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()

    mol_adj = np.zeros((num_atoms, num_atoms))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))

    edge_index = []
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    return num_atoms, features, edge_index


def dic_normalize(dic):
    """Normalize dictionary values to [0, 1]; adds 'X' as midpoint."""
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


# Normalize all property tables at module load time
RES_WEIGHT = dic_normalize(RES_WEIGHT_RAW.copy())
RES_PKA = dic_normalize(RES_PKA_RAW.copy())
RES_PKB = dic_normalize(RES_PKB_RAW.copy())
RES_PKX = dic_normalize(RES_PKX_RAW.copy())
RES_PI = dic_normalize(RES_PI_RAW.copy())
RES_HYDROPHOBIC_PH2 = dic_normalize(RES_HYDROPHOBIC_PH2_RAW.copy())
RES_HYDROPHOBIC_PH7 = dic_normalize(RES_HYDROPHOBIC_PH7_RAW.copy())


def residue_features(residue: str) -> np.ndarray:
    """
    Generate 12-dimensional biochemical property vector for a residue.
    - 5 binary features: chemical classification
    - 7 continuous features: normalized biochemical properties
    """
    res_property1 = [
        1 if residue in ALIPHATIC else 0,
        1 if residue in AROMATIC else 0,
        1 if residue in POLAR_NEUTRAL else 0,
        1 if residue in ACIDIC_CHARGED else 0,
        1 if residue in BASIC_CHARGED else 0
    ]
    res_property2 = [
        RES_WEIGHT[residue],
        RES_PKA[residue],
        RES_PKB[residue],
        RES_PKX[residue],
        RES_PI[residue],
        RES_HYDROPHOBIC_PH2[residue],
        RES_HYDROPHOBIC_PH7[residue]
    ]
    return np.array(res_property1 + res_property2, dtype=np.float32)


def PSSM_calculation(aln_file: str, pro_seq: str) -> np.ndarray:
    """
    Calculate Position-Specific Scoring Matrix from MSA alignment file.
    Returns PSSM matrix of shape (21, seq_length).
    """
    pfm_mat = np.zeros((len(AMINO_ACIDS), len(pro_seq)))

    with open(aln_file, 'r') as f:
        lines = f.readlines()
        line_count = len(lines)

        for line in lines:
            if len(line.strip()) != len(pro_seq):
                continue
            count = 0
            for res in line.strip():
                if res not in AMINO_ACIDS:
                    count += 1
                    continue
                pfm_mat[AMINO_ACIDS.index(res), count] += 1
                count += 1

    pseudocount = CONFIG['pssm_pseudocount']
    ppm_mat = (pfm_mat + pseudocount / 20) / (float(line_count) + pseudocount)
    return ppm_mat


def seq_feature(pro_seq: str) -> np.ndarray:
    """
    Generate one-hot + biochemical properties for protein sequence.
    Returns feature matrix of shape (seq_length, 33).
    """
    pro_hot = np.zeros((len(pro_seq), len(AMINO_ACIDS)))
    pro_property = np.zeros((len(pro_seq), 12))

    for i in range(len(pro_seq)):
        pro_hot[i, :] = one_of_k_encoding(pro_seq[i], AMINO_ACIDS)
        pro_property[i, :] = residue_features(pro_seq[i])

    return np.concatenate((pro_hot, pro_property), axis=1)


def compute_target_feature(aln_file: str, pro_seq: str) -> np.ndarray:
    """
    Generate complete 54-dim feature matrix for protein.
    Combines 21-dim PSSM + 33-dim sequence features.
    """
    pssm = PSSM_calculation(aln_file, pro_seq)
    pssm_transposed = np.transpose(pssm, (1, 0))
    other_feature = seq_feature(pro_seq)
    return np.concatenate((pssm_transposed, other_feature), axis=1)


def target_to_graph(target_key: str, target_sequence: str,
                    contact_dir: str, aln_dir: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Construct protein graph from contact map and sequence features.
    Returns: (target_size, target_feature, target_edge_index)
    """
    target_size = len(target_sequence)

    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))

    threshold = CONFIG['contact_map_threshold']
    index_row, index_col = np.where(contact_map >= threshold)
    target_edge_index = [[i, j] for i, j in zip(index_row, index_col)]

    aln_file = os.path.join(aln_dir, target_key + '.aln')
    target_feature = compute_target_feature(aln_file, target_sequence)
    target_edge_index = np.array(target_edge_index)

    return target_size, target_feature, target_edge_index


def valid_target(key: str, dataset: str) -> bool:
    """Check if required preprocessing files exist for a protein."""
    contact_dir = f"{CONFIG['data_dir']}/{dataset}/pconsc4"
    aln_dir = f"{CONFIG['data_dir']}/{dataset}/aln"
    contact_file = os.path.join(contact_dir, key + '.npy')
    aln_file = os.path.join(aln_dir, key + '.aln')
    return os.path.exists(contact_file) and os.path.exists(aln_file)


# ============================================================
# Model Architecture
# ============================================================

class BaseDualGNN(nn.Module):
    """
    Base class for Dual GNN architectures (molecule + protein branches).
    Subclasses must implement forward_molecule() and forward_protein().
    """

    def __init__(self, n_output=1, output_dim=128, dropout=0.2):
        super(BaseDualGNN, self).__init__()
        self.n_output = n_output
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

    def forward_molecule(self, data_mol):
        raise NotImplementedError("Subclass must implement forward_molecule()")

    def forward_protein(self, data_pro):
        raise NotImplementedError("Subclass must implement forward_protein()")

    def forward(self, data_mol, data_pro):
        mol_embedding = self.forward_molecule(data_mol)
        protein_embedding = self.forward_protein(data_pro)
        combined = torch.cat((mol_embedding, protein_embedding), 1)
        combined = self.relu(self.fc1(combined))
        combined = self.dropout(combined)
        combined = self.relu(self.fc2(combined))
        combined = self.dropout(combined)
        return self.out(combined)


class GNNNet(BaseDualGNN):
    """Dual Graph Convolutional Network (GCN) for DTA prediction."""

    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78,
                 output_dim=128, dropout=0.2):
        super(GNNNet, self).__init__(n_output, output_dim, dropout)
        print('GNNNet (GCN) Loaded')
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

    def forward_molecule(self, data_mol):
        x, edge_index, batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x = self.relu(self.mol_conv1(x, edge_index))
        x = self.relu(self.mol_conv2(x, edge_index))
        x = self.relu(self.mol_conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.mol_fc_g1(x)))
        return self.dropout(self.mol_fc_g2(x))

    def forward_protein(self, data_pro):
        x, edge_index, batch = data_pro.x, data_pro.edge_index, data_pro.batch
        x = self.relu(self.pro_conv1(x, edge_index))
        x = self.relu(self.pro_conv2(x, edge_index))
        x = self.relu(self.pro_conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.pro_fc_g1(x)))
        return self.dropout(self.pro_fc_g2(x))


class GNNNet_GAT(BaseDualGNN):
    """Dual Graph Attention Network (GAT) for DTA prediction."""

    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78,
                 output_dim=128, dropout=0.2, heads=None):
        super(GNNNet_GAT, self).__init__(n_output, output_dim, dropout)
        if heads is None:
            heads = CONFIG['gat_heads']
        print('GNNNet_GAT Loaded')
        self.heads = heads
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=heads, concat=False)
        self.mol_conv2 = GATConv(num_features_mol, num_features_mol * 2, heads=heads, concat=False)
        self.mol_conv3 = GATConv(num_features_mol * 2, num_features_mol * 4, heads=heads, concat=False)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)
        self.pro_conv1 = GATConv(num_features_pro, num_features_pro, heads=heads, concat=False)
        self.pro_conv2 = GATConv(num_features_pro, num_features_pro * 2, heads=heads, concat=False)
        self.pro_conv3 = GATConv(num_features_pro * 2, num_features_pro * 4, heads=heads, concat=False)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

    def forward_molecule(self, data_mol):
        x, edge_index, batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x = self.relu(self.mol_conv1(x, edge_index))
        x = self.relu(self.mol_conv2(x, edge_index))
        x = self.relu(self.mol_conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.mol_fc_g1(x)))
        return self.dropout(self.mol_fc_g2(x))

    def forward_protein(self, data_pro):
        x, edge_index, batch = data_pro.x, data_pro.edge_index, data_pro.batch
        x = self.relu(self.pro_conv1(x, edge_index))
        x = self.relu(self.pro_conv2(x, edge_index))
        x = self.relu(self.pro_conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.pro_fc_g1(x)))
        return self.dropout(self.pro_fc_g2(x))


class GNNNet_GIN(BaseDualGNN):
    """Dual Graph Isomorphism Network (GIN) for DTA prediction."""

    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78,
                 output_dim=128, dropout=0.2):
        super(GNNNet_GIN, self).__init__(n_output, output_dim, dropout)
        print('GNNNet_GIN Loaded')

        mol_nn1 = nn.Sequential(nn.Linear(num_features_mol, num_features_mol), nn.ReLU(),
                                nn.Linear(num_features_mol, num_features_mol))
        self.mol_conv1 = GINConv(mol_nn1)
        mol_nn2 = nn.Sequential(nn.Linear(num_features_mol, num_features_mol * 2), nn.ReLU(),
                                nn.Linear(num_features_mol * 2, num_features_mol * 2))
        self.mol_conv2 = GINConv(mol_nn2)
        mol_nn3 = nn.Sequential(nn.Linear(num_features_mol * 2, num_features_mol * 4), nn.ReLU(),
                                nn.Linear(num_features_mol * 4, num_features_mol * 4))
        self.mol_conv3 = GINConv(mol_nn3)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)

        pro_nn1 = nn.Sequential(nn.Linear(num_features_pro, num_features_pro), nn.ReLU(),
                                nn.Linear(num_features_pro, num_features_pro))
        self.pro_conv1 = GINConv(pro_nn1)
        pro_nn2 = nn.Sequential(nn.Linear(num_features_pro, num_features_pro * 2), nn.ReLU(),
                                nn.Linear(num_features_pro * 2, num_features_pro * 2))
        self.pro_conv2 = GINConv(pro_nn2)
        pro_nn3 = nn.Sequential(nn.Linear(num_features_pro * 2, num_features_pro * 4), nn.ReLU(),
                                nn.Linear(num_features_pro * 4, num_features_pro * 4))
        self.pro_conv3 = GINConv(pro_nn3)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

    def forward_molecule(self, data_mol):
        x, edge_index, batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x = self.relu(self.mol_conv1(x, edge_index))
        x = self.relu(self.mol_conv2(x, edge_index))
        x = self.relu(self.mol_conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.mol_fc_g1(x)))
        return self.dropout(self.mol_fc_g2(x))

    def forward_protein(self, data_pro):
        x, edge_index, batch = data_pro.x, data_pro.edge_index, data_pro.batch
        x = self.relu(self.pro_conv1(x, edge_index))
        x = self.relu(self.pro_conv2(x, edge_index))
        x = self.relu(self.pro_conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.pro_fc_g1(x)))
        return self.dropout(self.pro_fc_g2(x))


class GNNNetLegacy(nn.Module):
    """
    Original DGraphDTA architecture (GCN-based) for loading pre-trained checkpoints.
    Compatible with PyG 1.3.2 state dicts via convert_legacy_state_dict().
    """

    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78,
                 output_dim=128, dropout=0.2):
        super(GNNNetLegacy, self).__init__()
        print('GNNNetLegacy Loaded')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol, bias=False)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2, bias=False)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4, bias=False)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro, bias=False)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2, bias=False)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4, bias=False)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        x = self.relu(self.mol_conv1(mol_x, mol_edge_index))
        x = self.relu(self.mol_conv2(x, mol_edge_index))
        x = self.relu(self.mol_conv3(x, mol_edge_index))
        x = global_mean_pool(x, mol_batch)
        x = self.dropout(self.relu(self.mol_fc_g1(x)))
        x = self.dropout(self.mol_fc_g2(x))

        xt = self.relu(self.pro_conv1(target_x, target_edge_index))
        xt = self.relu(self.pro_conv2(xt, target_edge_index))
        xt = self.relu(self.pro_conv3(xt, target_edge_index))
        xt = global_mean_pool(xt, target_batch)
        xt = self.dropout(self.relu(self.pro_fc_g1(xt)))
        xt = self.dropout(self.pro_fc_g2(xt))

        xc = torch.cat((x, xt), 1)
        xc = self.dropout(self.relu(self.fc1(xc)))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return self.out(xc)


# ============================================================
# Dataset
# ============================================================

class DTADataset(InMemoryDataset):
    """Drug-Target Affinity Dataset using PyTorch Geometric InMemoryDataset."""

    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_graph)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), \
            'The three lists must be the same length!'

        data_list_mol = []
        data_list_pro = []

        for i in range(len(xd)):
            smiles = xd[i]
            protein_key = target_key[i]
            affinity_value = y[i]

            num_atoms, features, edge_index = smile_graph[smiles]
            target_size, target_features, target_edge_index = target_graph[protein_key]

            GCNData_mol = Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([affinity_value])
            )
            GCNData_mol.__setitem__('c_size', torch.LongTensor([num_atoms]))

            GCNData_pro = Data(
                x=torch.Tensor(target_features),
                edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                y=torch.FloatTensor([affinity_value])
            )
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))

            data_list_mol.append(GCNData_mol)
            data_list_pro.append(GCNData_pro)

        if self.pre_filter is not None:
            data_list_mol = [d for d in data_list_mol if self.pre_filter(d)]
            data_list_pro = [d for d in data_list_pro if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(d) for d in data_list_mol]
            data_list_pro = [self.pre_transform(d) for d in data_list_pro]

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]


def collate(data_list):
    """Custom collate function for batching dual graphs."""
    batchA = Batch.from_data_list([d[0] for d in data_list])
    batchB = Batch.from_data_list([d[1] for d in data_list])
    return batchA, batchB


# ============================================================
# Data Loading
# ============================================================

def load_davis_data(data_dir=None, use_full=None):
    """
    Load Davis kinase inhibitor dataset.
    Returns: (smiles_list, protein_sequences, protein_keys, affinities)
    """
    if data_dir is None:
        data_dir = f"{CONFIG['data_dir']}/davis"
    if use_full is None:
        use_full = CONFIG['use_full_dataset']
    try:
        with open(os.path.join(data_dir, 'ligands_can.txt'), 'r') as f:
            ligands = json.load(f, object_pairs_hook=OrderedDict)
        with open(os.path.join(data_dir, 'proteins.txt'), 'r') as f:
            proteins = json.load(f, object_pairs_hook=OrderedDict)
        with open(os.path.join(data_dir, 'Y'), 'rb') as f:
            affinity = pickle.load(f, encoding='latin1')

        affinity = [-np.log10(y / 1e9) for y in affinity]
        affinity = np.asarray(affinity)

        drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
                 for d in ligands.keys()]
        prots = list(proteins.values())
        prot_keys = list(proteins.keys())

        rows, cols = np.where(~np.isnan(affinity))
        smiles_list = [drugs[i] for i in rows]
        protein_sequences = [prots[i] for i in cols]
        protein_keys_list = [prot_keys[i] for i in cols]
        affinity_list = [affinity[i, j] for i, j in zip(rows, cols)]

        if not use_full:
            max_samples = CONFIG['max_samples']
            if len(smiles_list) > max_samples:
                indices = np.random.choice(len(smiles_list), max_samples, replace=False)
                smiles_list = [smiles_list[i] for i in indices]
                protein_sequences = [protein_sequences[i] for i in indices]
                protein_keys_list = [protein_keys_list[i] for i in indices]
                affinity_list = [affinity_list[i] for i in indices]

        print(f"Loaded {len(smiles_list)} Davis samples")
        return smiles_list, protein_sequences, protein_keys_list, affinity_list

    except FileNotFoundError as e:
        print(f"Error loading Davis data: {e}")
        return None, None, None, None


def load_kiba_data(data_dir=None, use_full=None):
    """
    Load KIBA kinase inhibitor bioactivity dataset.
    Returns: (smiles_list, protein_sequences, protein_keys, affinities)
    """
    if data_dir is None:
        data_dir = f"{CONFIG['data_dir']}/kiba"
    if use_full is None:
        use_full = CONFIG['use_full_dataset']
    try:
        with open(os.path.join(data_dir, 'ligands_can.txt'), 'r') as f:
            ligands = json.load(f, object_pairs_hook=OrderedDict)
        with open(os.path.join(data_dir, 'proteins.txt'), 'r') as f:
            proteins = json.load(f, object_pairs_hook=OrderedDict)
        with open(os.path.join(data_dir, 'Y'), 'rb') as f:
            affinity = pickle.load(f, encoding='latin1')
        affinity = np.asarray(affinity)

        drugs = [Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
                 for d in ligands.keys()]
        prots = list(proteins.values())
        prot_keys = list(proteins.keys())

        rows, cols = np.where(~np.isnan(affinity))
        smiles_list = [drugs[i] for i in rows]
        protein_sequences = [prots[i] for i in cols]
        protein_keys_list = [prot_keys[i] for i in cols]
        affinity_list = [affinity[i, j] for i, j in zip(rows, cols)]

        if not use_full:
            max_samples = CONFIG['max_samples']
            if len(smiles_list) > max_samples:
                indices = np.random.choice(len(smiles_list), max_samples, replace=False)
                smiles_list = [smiles_list[i] for i in indices]
                protein_sequences = [protein_sequences[i] for i in indices]
                protein_keys_list = [protein_keys_list[i] for i in indices]
                affinity_list = [affinity_list[i] for i in indices]

        print(f"Loaded {len(smiles_list)} KIBA samples")
        return smiles_list, protein_sequences, protein_keys_list, affinity_list

    except FileNotFoundError as e:
        print(f"Error loading KIBA data: {e}")
        return None, None, None, None


# ============================================================
# Evaluation Metrics
# ============================================================

def get_mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def get_rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def get_pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def get_spearman(y, f):
    return stats.spearmanr(y, f)[0]


def concordance_index(y_true, y_pred):
    """
    Concordance Index (CI) — measures ranking quality.
    CI=1.0 is perfect, CI=0.5 is random.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ind = np.argsort(y_true)
    y_true = y_true[ind]
    y_pred = y_pred[ind]

    concordant = 0
    total = 0
    for i in range(len(y_true) - 1):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
                   (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
    return concordant / total if total > 0 else 0.5


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    y_pred_mean = [np.mean(y_pred) for _ in y_pred]
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    upp = sum((y_obs - (k * y_pred)) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    """Modified R² metric accounting for systematic error."""
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def calculate_metrics(Y, P, dataset_name=''):
    """Calculate all DTA evaluation metrics."""
    metrics = {
        'mse': get_mse(Y, P),
        'rmse': get_rmse(Y, P),
        'pearson': get_pearson(Y, P),
        'spearman': get_spearman(Y, P),
        'ci': concordance_index(Y, P),
        'rm2': get_rm2(Y, P)
    }
    if dataset_name:
        print(f"\nMetrics for {dataset_name}:")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v:.4f}")
    return metrics


# ============================================================
# Training and Prediction
# ============================================================

def train_epoch(model, device, train_loader, optimizer, epoch, log_interval=None):
    """Train model for one epoch."""
    if log_interval is None:
        log_interval = CONFIG['log_interval']
    print(f'Training on {len(train_loader.dataset)} samples...')
    model.train()
    loss_fn = nn.MSELoss()

    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'  Epoch {epoch} [{batch_idx * len(data_mol.y)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def predicting(model, device, loader):
    """Generate predictions for a data loader. Returns (true_labels, predictions)."""
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f'Predicting on {len(loader.dataset)} samples...')

    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# ============================================================
# Demo Dataset Utilities
# ============================================================

def create_simplified_protein_features(protein_sequence: str) -> np.ndarray:
    """
    Create 33-dim protein node features without PSSM (no alignment files needed).
    Uses one-hot encoding + biochemical properties only.
    """
    return seq_feature(protein_sequence)


def create_demo_dataset(smiles_list, protein_sequences, affinity_values,
                        num_samples=None, train_ratio=None, seed=None):
    """
    Create reproducible demo dataset with simplified protein features (no PSSM).
    Returns: (train_dataset, val_dataset)
    """
    if num_samples is None:
        num_samples = CONFIG['demo_samples']
    if train_ratio is None:
        train_ratio = CONFIG['demo_train_ratio']
    if seed is None:
        seed = CONFIG['seed']

    np.random.seed(seed)
    random.seed(seed)

    total_samples = min(num_samples, len(smiles_list))
    indices = np.random.choice(len(smiles_list), total_samples, replace=False)

    demo_smiles = [smiles_list[i] for i in indices]
    demo_sequences = [protein_sequences[i] for i in indices]
    demo_affinities = [affinity_values[i] for i in indices]

    print(f"\n{'='*60}")
    print("Creating Demo Dataset")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")

    # Pre-compute molecular graphs
    smile_graph = {}
    print("Converting molecules to graphs...")
    for smiles in tqdm(set(demo_smiles), desc="Molecular graphs"):
        try:
            num_atoms, features, edges = smile_to_graph(smiles)
            smile_graph[smiles] = (num_atoms, features, edges)
        except Exception:
            pass

    # Filter out molecules that failed
    valid_mask = [s in smile_graph for s in demo_smiles]
    demo_smiles = [s for s, v in zip(demo_smiles, valid_mask) if v]
    demo_sequences = [s for s, v in zip(demo_sequences, valid_mask) if v]
    demo_affinities = [a for a, v in zip(demo_affinities, valid_mask) if v]

    # Pre-compute simplified protein graphs (k-NN on sequence position)
    protein_graph = {}
    print("Creating simplified protein graphs...")
    for seq in tqdm(set(demo_sequences), desc="Protein graphs"):
        target_size = len(seq)
        target_features = create_simplified_protein_features(seq)
        k = 3
        target_edge_index = []
        for i in range(target_size):
            target_edge_index.append([i, i])
            for offset in range(1, k + 1):
                if i + offset < target_size:
                    target_edge_index.append([i, i + offset])
                    target_edge_index.append([i + offset, i])
        protein_graph[seq] = (target_size, target_features, np.array(target_edge_index))

    split_idx = int(len(demo_smiles) * train_ratio)

    train_dataset = DTADataset(
        root='/tmp', dataset='demo_train',
        xd=demo_smiles[:split_idx],
        y=demo_affinities[:split_idx],
        smile_graph=smile_graph,
        target_key=demo_sequences[:split_idx],
        target_graph=protein_graph
    )

    val_dataset = DTADataset(
        root='/tmp', dataset='demo_val',
        xd=demo_smiles[split_idx:],
        y=demo_affinities[split_idx:],
        smile_graph=smile_graph,
        target_key=demo_sequences[split_idx:],
        target_graph=protein_graph
    )

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    return train_dataset, val_dataset


def train_demo_model(train_dataset, val_dataset, model_type='GCN',
                     num_epochs=None, batch_size=None, learning_rate=None):
    """
    Train a single GNN model on demo dataset with validation tracking.
    Returns: (trained_model, history_dict)
    """
    if num_epochs is None:
        num_epochs = CONFIG['demo_epochs']
    if batch_size is None:
        batch_size = CONFIG['demo_batch_size']
    if learning_rate is None:
        learning_rate = CONFIG['learning_rate']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    pro_dim = CONFIG['num_features_pro_simplified']
    mol_dim = CONFIG['num_features_mol']

    if model_type == 'GCN':
        model = GNNNet(num_features_pro=pro_dim, num_features_mol=mol_dim).to(device)
    elif model_type == 'GAT':
        model = GNNNet_GAT(num_features_pro=pro_dim, num_features_mol=mol_dim).to(device)
    elif model_type == 'GIN':
        model = GNNNet_GIN(num_features_pro=pro_dim, num_features_mol=mol_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_ci': [], 'val_rmse': [], 'val_pearson': []}

    print(f"\nTraining {model_type} model for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for data in train_loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            optimizer.zero_grad()
            output = model(data_mol, data_pro)
            loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        history['train_loss'].append(np.mean(train_losses))

        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for data in val_loader:
                data_mol = data[0].to(device)
                data_pro = data[1].to(device)
                output = model(data_mol, data_pro)
                loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
                val_losses.append(loss.item())
                val_preds.extend(output.cpu().numpy().flatten())
                val_labels.extend(data_mol.y.cpu().numpy().flatten())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_ci = concordance_index(val_labels, val_preds)
        val_rmse = get_rmse(val_labels, val_preds)
        val_pearson = get_pearson(val_labels, val_preds)

        history['val_loss'].append(np.mean(val_losses))
        history['val_ci'].append(val_ci)
        history['val_rmse'].append(val_rmse)
        history['val_pearson'].append(val_pearson)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={history['train_loss'][-1]:.4f}, "
                  f"val_CI={val_ci:.4f}, val_RMSE={val_rmse:.4f}")

    print(f"\nTraining complete. Final Val CI: {history['val_ci'][-1]:.4f}, "
          f"Val RMSE: {history['val_rmse'][-1]:.4f}")
    return model, history


def compare_architectures(smiles_list, protein_sequences, affinity_values, num_epochs=None):
    """
    Train and compare GCN, GAT, and GIN architectures on the same dataset.
    Returns: dict of results keyed by architecture name.
    """
    if num_epochs is None:
        num_epochs = CONFIG['demo_epochs']

    print(f"\n{'='*60}")
    print("Architecture Comparison: GCN vs GAT vs GIN")
    print(f"{'='*60}")

    train_dataset, val_dataset = create_demo_dataset(
        smiles_list, protein_sequences, affinity_values, num_samples=500
    )

    architectures = ['GCN', 'GAT', 'GIN']
    results = {}

    for arch in architectures:
        print(f"\n--- Training {arch} ---")
        model, history = train_demo_model(
            train_dataset, val_dataset, model_type=arch, num_epochs=num_epochs
        )
        results[arch] = {
            'model': model,
            'history': history,
            'final_ci': history['val_ci'][-1],
            'final_rmse': history['val_rmse'][-1],
            'final_loss': history['val_loss'][-1]
        }

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'GCN': '#4A8A7F', 'GAT': '#67AB9F', 'GIN': '#8BC9BD'}
    styles = {'GCN': '-', 'GAT': '--', 'GIN': ':'}

    for metric, ax, title in [
        ('val_loss', axes[0], 'Validation Loss'),
        ('val_ci', axes[1], 'Concordance Index'),
        ('val_rmse', axes[2], 'RMSE'),
    ]:
        for arch in architectures:
            vals = results[arch]['history'][metric]
            ax.plot(range(1, len(vals) + 1), vals,
                    color=colors[arch], linestyle=styles[arch],
                    linewidth=2, label=arch)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'figures/{CHAPTER}/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"\nComparison plot saved to figures/{CHAPTER}/architecture_comparison.png")

    print(f"\n{'Architecture':<15} {'Val Loss':<12} {'Val CI':<12} {'Val RMSE':<12}")
    print("-" * 50)
    for arch in architectures:
        print(f"{arch:<15} {results[arch]['final_loss']:<12.4f} "
              f"{results[arch]['final_ci']:<12.4f} {results[arch]['final_rmse']:<12.4f}")

    best_arch = max(architectures, key=lambda a: results[a]['final_ci'])
    print(f"\nBest architecture by CI: {best_arch} (CI={results[best_arch]['final_ci']:.4f})")

    return results


# ============================================================
# Pre-trained Model Utilities
# ============================================================

def convert_legacy_state_dict(old_state_dict):
    """
    Convert PyG 1.3.2 GCNConv state dict to PyG 2.x format.
    Remaps mol_conv/pro_conv weight -> .lin.weight and transposes.
    """
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if any(conv in key for conv in ['mol_conv', 'pro_conv']):
            if key.endswith('.weight'):
                parts = key.rsplit('.', 1)
                new_key = f"{parts[0]}.lin.{parts[1]}"
                new_state_dict[new_key] = value.t()
            elif key.endswith('.bias'):
                parts = key.rsplit('.', 1)
                new_key = f"{parts[0]}.lin.{parts[1]}"
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_pretrained_model(model_path, num_features_pro=None, num_features_mol=None):
    """Load pre-trained DGraphDTA model from checkpoint."""
    if num_features_pro is None:
        num_features_pro = CONFIG['num_features_pro']
    if num_features_mol is None:
        num_features_mol = CONFIG['num_features_mol']

    model = GNNNetLegacy(
        num_features_pro=num_features_pro,
        num_features_mol=num_features_mol
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    converted = convert_legacy_state_dict(checkpoint)
    model.load_state_dict(converted, strict=False)
    model.eval()
    print(f"Loaded legacy model from {model_path}")
    return model


def evaluate_pretrained_model(model_path, dataset_name='davis'):
    """
    Evaluate pre-trained model on test set.
    Requires preprocessing files (aln/, pconsc4/) in data/ch11/{dataset}/
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Pre-trained Model: {dataset_name.upper()}")
    print(f"{'='*60}")

    dataset_dir = f"{CONFIG['data_dir']}/{dataset_name}"
    aln_dir = os.path.join(dataset_dir, 'aln')
    pconsc4_dir = os.path.join(dataset_dir, 'pconsc4')

    if not (os.path.exists(aln_dir) and os.path.exists(pconsc4_dir)):
        print(f"\nPreprocessing files not found ({aln_dir}/ and {pconsc4_dir}/).")
        print("Download from https://drive.google.com/open?id=1rqAopf_IaH3jzFkwXObQ4i-6UUwizCv")
        return None

    model = load_pretrained_model(model_path)
    print(f"Loading {dataset_name} test data...")

    if dataset_name == 'davis':
        smiles_list, protein_sequences, protein_keys, affinities = load_davis_data(
            data_dir=dataset_dir, use_full=True
        )
    else:
        smiles_list, protein_sequences, protein_keys, affinities = load_kiba_data(
            data_dir=dataset_dir, use_full=True
        )

    if smiles_list is None:
        return None

    smile_graph = {}
    for smiles in tqdm(set(smiles_list), desc="Building mol graphs"):
        try:
            smile_graph[smiles] = smile_to_graph(smiles)
        except Exception:
            pass

    target_graph = {}
    for key, seq in zip(protein_keys, protein_sequences):
        if key not in target_graph and valid_target(key, dataset_name):
            target_graph[key] = target_to_graph(key, seq, pconsc4_dir, aln_dir)

    # Filter entries where both graphs exist
    valid = [(s, k, a) for s, k, a in zip(smiles_list, protein_keys, affinities)
             if s in smile_graph and k in target_graph]
    if not valid:
        print("No valid samples after filtering.")
        return None

    smiles_list, protein_keys, affinities = zip(*valid)
    test_dataset = DTADataset(
        root='/tmp', dataset=f'{dataset_name}_pretrained_test',
        xd=list(smiles_list), y=list(affinities),
        smile_graph=smile_graph,
        target_key=list(protein_keys),
        target_graph=target_graph
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=collate)

    Y_test, P_test = predicting(model, device, test_loader)
    metrics = calculate_metrics(Y_test, P_test, dataset_name.upper())

    # Save prediction plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(Y_test, P_test, alpha=0.4, color='#67AB9F', edgecolors='#4A8A7F', linewidths=0.5)
    min_val, max_val = min(Y_test.min(), P_test.min()), max(Y_test.max(), P_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax.set_xlabel('True Affinity')
    ax.set_ylabel('Predicted Affinity')
    ax.set_title(f'{dataset_name.upper()} Predictions (CI={metrics["ci"]:.4f})')
    plt.tight_layout()
    plt.savefig(f'figures/{CHAPTER}/{dataset_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close('all')

    return metrics


def plot_dataset_distributions(davis_data=None, kiba_data=None,
                               save_path=None):
    """Create violin plots for DAVIS and KIBA affinity distributions."""
    if save_path is None:
        save_path = f'figures/{CHAPTER}/dataset_distributions.png'

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    datasets = []
    if davis_data and davis_data[3] is not None:
        datasets.append(('DAVIS', davis_data[3]))
    if kiba_data and kiba_data[3] is not None:
        datasets.append(('KIBA', kiba_data[3]))

    if not datasets:
        print("No datasets available for visualization")
        plt.close()
        return

    for idx, (name, affinities) in enumerate(datasets):
        ax = axes[idx]
        parts = ax.violinplot([affinities], positions=[0], showmeans=False, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#67AB9F')
            pc.set_alpha(0.8)
        ax.set_title(f'{name} Dataset')
        ylabel = 'Affinity (pKd)' if name == 'DAVIS' else 'KIBA Score'
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)
        ax.text(0.97, 0.97, f"n={len(affinities):,}\nMean={np.mean(affinities):.2f}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if len(datasets) == 1:
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Distribution plot saved to {save_path}")


# ============================================================
# Demo / Synthetic Data
# ============================================================

def create_synthetic_demo_data(n_samples=200):
    """Create synthetic drug-target pairs for demonstration."""
    demo_smiles = [
        'CC(=O)Nc1ccc(O)cc1',
        'CC(=O)Oc1ccccc1C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',
        'c1ccc2ccccc2c1',
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        'OC(=O)c1ccccc1',
        'c1ccc(cc1)C(=O)O',
    ]
    demo_proteins = [
        'MKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY',
        'MGSSHHHHHHSSGLVPRGSHMGDTDLQLRQQAQRRMSE',
        'MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYD',
        'MTEYKLVVVGAVGVGKSALTIQLIQNHFVDEYDPT',
    ]

    smiles_list = [random.choice(demo_smiles) for _ in range(n_samples)]
    protein_sequences = [random.choice(demo_proteins) for _ in range(n_samples)]
    affinity_values = np.random.uniform(4.0, 12.0, n_samples).tolist()

    return smiles_list, protein_sequences, affinity_values


if __name__ == "__main__":
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Try to load real datasets
    print("\nAttempting to load Davis and KIBA datasets...")
    davis_smiles, davis_seqs, davis_keys, davis_affinities = load_davis_data()
    kiba_smiles, kiba_seqs, kiba_keys, kiba_affinities = load_kiba_data()

    if davis_smiles is not None:
        print(f"Davis: {len(davis_smiles)} samples, "
              f"affinity range [{min(davis_affinities):.2f}, {max(davis_affinities):.2f}] pKd")

    if kiba_smiles is not None:
        print(f"KIBA: {len(kiba_smiles)} samples, "
              f"affinity range [{min(kiba_affinities):.2f}, {max(kiba_affinities):.2f}]")

    # Plot distributions if data available
    if davis_smiles is not None or kiba_smiles is not None:
        davis_data = (davis_smiles, davis_seqs, davis_keys, davis_affinities) if davis_smiles else None
        kiba_data = (kiba_smiles, kiba_seqs, kiba_keys, kiba_affinities) if kiba_smiles else None
        plot_dataset_distributions(davis_data=davis_data, kiba_data=kiba_data)

    # Choose data source for architecture comparison
    if davis_smiles is not None:
        print("\nUsing Davis dataset for architecture comparison...")
        smiles_list = davis_smiles
        protein_sequences = davis_seqs
        affinity_values = davis_affinities
    elif kiba_smiles is not None:
        print("\nUsing KIBA dataset for architecture comparison...")
        smiles_list = kiba_smiles
        protein_sequences = kiba_seqs
        affinity_values = kiba_affinities
    else:
        print("\nNo real dataset found. Creating synthetic demo data...")
        smiles_list, protein_sequences, affinity_values = create_synthetic_demo_data(n_samples=200)
        print(f"Created {len(smiles_list)} synthetic drug-target pairs")

    # Compare architectures
    print("\nComparing GCN, GAT, and GIN architectures...")
    comparison_results = compare_architectures(
        smiles_list, protein_sequences, affinity_values, num_epochs=10
    )
    print("\nArchitecture comparison complete!")

    # Check for pre-trained models
    davis_model_path = f"{CONFIG['model_dir']}/model_GNNNet_davis.model"
    kiba_model_path = f"{CONFIG['model_dir']}/model_GNNNet_kiba.model"

    if os.path.exists(davis_model_path):
        print(f"\nFound pre-trained Davis model, evaluating...")
        evaluate_pretrained_model(davis_model_path, 'davis')
    else:
        print(f"\nNo pre-trained Davis model at {davis_model_path}")

    if os.path.exists(kiba_model_path):
        print(f"\nFound pre-trained KIBA model, evaluating...")
        evaluate_pretrained_model(kiba_model_path, 'kiba')
    else:
        print(f"\nNo pre-trained KIBA model at {kiba_model_path}")

    print(f"\nChapter 11 complete. Outputs saved to figures/{CHAPTER}/ and artifacts/{CHAPTER}/")
