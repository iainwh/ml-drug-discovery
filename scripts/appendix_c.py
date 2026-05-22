"""
Appendix C: Knowledge Distillation for Hierarchical Molecular Generation
- Teacher: pre-trained HierVAE (hgraph2graph) with large hidden dimensions
- Student: compressed HierVAE with smaller hidden dimensions
- Distillation losses: reconstruction, KL, latent alignment, feature matching
- OOV-robust generation evaluation

Requires:
    - hgraph2graph repository cloned and importable (https://github.com/wengong-jin/hgraph2graph)
    - Pre-trained teacher checkpoint at ckpt/chembl-pretrained/model.ckpt
    - Vocabulary file at data/chembl/vocab.txt
    - Training SMILES at data/chembl/all.txt
"""

import os
import json
import random
import time
import warnings
from argparse import Namespace
from collections import defaultdict
from functools import partial

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
import rdkit.RDLogger as RDLogger

warnings.filterwarnings('ignore')
RDLogger.logger().setLevel(RDLogger.CRITICAL)

CHAPTER = "appendix_c"
RANDOM_SEED = 42

os.makedirs("ckpt/distillation", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


class Config:
    """Configuration for knowledge distillation pipeline."""

    # Paths
    TEACHER_MODEL_PATH = "ckpt/chembl-pretrained/model.ckpt"
    VOCAB_PATH = "data/chembl/vocab.txt"
    OUTPUT_DIR = "ckpt/distillation"

    # Teacher architecture (matches pre-trained checkpoint)
    TEACHER_HIDDEN_SIZE = 250
    TEACHER_EMBED_SIZE = 250
    TEACHER_LATENT_SIZE = 32
    TEACHER_DEPTH_T = 15
    TEACHER_DEPTH_G = 15

    # Student architecture (compressed)
    STUDENT_HIDDEN_SIZE = 128
    STUDENT_EMBED_SIZE = 128
    STUDENT_LATENT_SIZE = 16
    STUDENT_DEPTH_T = 15
    STUDENT_DEPTH_G = 15

    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    WARMUP_EPOCHS = 5

    # Distillation loss weights
    ALPHA_RECONSTRUCTION = 1.0
    BETA_KL = 0.1
    GAMMA_LATENT = 0.3
    DELTA_FEATURE = 0.2

    # KL annealing
    KL_ANNEAL_START = 0.0
    KL_ANNEAL_RATE = 0.01

    # Training settings
    CLIP_NORM = 5.0
    SAVE_EVERY = 5
    EVAL_EVERY = 1

    # Evaluation
    NUM_EVAL_SAMPLES = 1000
    NUM_GEN_SAMPLES = 1000

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RNN_TYPE = 'LSTM'
    DROPOUT = 0.0


def _try_import_hgraph():
    """
    Attempt to import hgraph2graph modules.
    Returns (HierVAE, PairVocab, common_atom_vocab, MolGraph) or raises ImportError
    with a helpful message.
    """
    try:
        from hgraph import HierVAE, PairVocab, common_atom_vocab, MolGraph
        return HierVAE, PairVocab, common_atom_vocab, MolGraph
    except ImportError:
        raise ImportError(
            "hgraph2graph is required for Appendix C.\n"
            "Clone and install it:\n"
            "  git clone https://github.com/wengong-jin/hgraph2graph.git\n"
            "  cd hgraph2graph && pip install -e .\n"
            "Then run this script from inside the hgraph2graph directory."
        )


class TeacherModelLoader:
    """Utility class for loading and preparing the teacher HierVAE model."""

    @staticmethod
    def load_vocab(vocab_path):
        """Load PairVocab from a vocabulary file."""
        _, PairVocab, _, _ = _try_import_hgraph()
        print(f"\nLoading vocabulary from: {vocab_path}")
        with open(vocab_path) as f:
            vocab_list = [x.strip("\r\n ").split() for x in f]
        vocab = PairVocab(vocab_list)
        print(f"Vocabulary: {vocab.size()[0]} cluster types, {vocab.size()[1]} motifs")
        return vocab

    @staticmethod
    def create_teacher_args(config, vocab):
        """Build Namespace args for teacher HierVAE."""
        _, _, common_atom_vocab, _ = _try_import_hgraph()
        return Namespace(
            vocab=vocab,
            atom_vocab=common_atom_vocab,
            rnn_type=config.RNN_TYPE,
            hidden_size=config.TEACHER_HIDDEN_SIZE,
            embed_size=config.TEACHER_EMBED_SIZE,
            latent_size=config.TEACHER_LATENT_SIZE,
            depthT=config.TEACHER_DEPTH_T,
            depthG=config.TEACHER_DEPTH_G,
            diterT=1,
            diterG=3,
            dropout=config.DROPOUT
        )

    @staticmethod
    def load_teacher_model(config, vocab):
        """Load pre-trained teacher model (frozen, eval mode)."""
        HierVAE, _, _, _ = _try_import_hgraph()
        print(f"\nLoading teacher model from: {config.TEACHER_MODEL_PATH}")

        teacher_args = TeacherModelLoader.create_teacher_args(config, vocab)
        teacher_model = HierVAE(teacher_args).to(config.DEVICE)

        checkpoint = torch.load(config.TEACHER_MODEL_PATH, map_location=config.DEVICE)
        model_state = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint
        teacher_model.load_state_dict(model_state)

        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        num_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"Teacher loaded: {num_params:,} parameters ({num_params/1e6:.2f}M)")
        return teacher_model


class StudentModelFactory:
    """Factory for creating compressed student HierVAE models."""

    @staticmethod
    def create_student_args(config, vocab):
        """Build Namespace args for student HierVAE."""
        _, _, common_atom_vocab, _ = _try_import_hgraph()
        return Namespace(
            vocab=vocab,
            atom_vocab=common_atom_vocab,
            rnn_type=config.RNN_TYPE,
            hidden_size=config.STUDENT_HIDDEN_SIZE,
            embed_size=config.STUDENT_EMBED_SIZE,
            latent_size=config.STUDENT_LATENT_SIZE,
            depthT=config.STUDENT_DEPTH_T,
            depthG=config.STUDENT_DEPTH_G,
            diterT=1,
            diterG=3,
            dropout=config.DROPOUT
        )

    @staticmethod
    def create_student_model(config, vocab, teacher_model=None):
        """Create and initialize student model."""
        HierVAE, _, _, _ = _try_import_hgraph()
        print("\nCreating student model...")

        student_args = StudentModelFactory.create_student_args(config, vocab)
        student_model = HierVAE(student_args).to(config.DEVICE)

        for param in student_model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        num_params = sum(p.numel() for p in student_model.parameters())
        print(f"Student created: {num_params:,} parameters ({num_params/1e6:.2f}M)")

        if teacher_model is not None:
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            reduction = (1 - num_params / teacher_params) * 100
            print(f"Parameter reduction vs teacher: {reduction:.1f}%")

        return student_model

    @staticmethod
    def compare_architectures(teacher_model, student_model):
        """Print side-by-side parameter count comparison."""
        print("\n" + "=" * 70)
        print("ARCHITECTURE COMPARISON")
        print("=" * 70)

        def count_module_params(module):
            return sum(p.numel() for p in module.parameters())

        components = [
            ('Encoder', 'encoder'),
            ('Decoder', 'decoder'),
            ('Latent Projection', 'R_mean'),
        ]

        print(f"{'Component':<22} {'Teacher':>12} {'Student':>12} {'Reduction':>12}")
        print("-" * 60)

        for name, attr in components:
            if hasattr(teacher_model, attr) and hasattr(student_model, attr):
                t_params = count_module_params(getattr(teacher_model, attr))
                s_params = count_module_params(getattr(student_model, attr))
                red = (1 - s_params / t_params) * 100
                print(f"{name:<22} {t_params:>12,} {s_params:>12,} {red:>11.1f}%")

        t_total = sum(p.numel() for p in teacher_model.parameters())
        s_total = sum(p.numel() for p in student_model.parameters())
        total_red = (1 - s_total / t_total) * 100
        print("-" * 60)
        print(f"{'TOTAL':<22} {t_total:>12,} {s_total:>12,} {total_red:>11.1f}%")
        print("=" * 70)


class DistillationLoss(nn.Module):
    """Combined knowledge distillation loss (reconstruction + KL + latent + feature)."""

    def __init__(self, config):
        super(DistillationLoss, self).__init__()
        self.config = config

    def compute_latent_distillation_loss(self, student_latent, teacher_latent):
        """MSE between student and teacher latent means (trims teacher to student dim)."""
        teacher_reduced = teacher_latent[:, :student_latent.size(1)]
        return F.mse_loss(student_latent, teacher_reduced)

    def compute_feature_matching_loss(self, student_features, teacher_features):
        """MSE between intermediate encoder representations."""
        losses = []
        if 'root_vecs' in student_features and 'root_vecs' in teacher_features:
            s_root = student_features['root_vecs']
            t_root = teacher_features['root_vecs']
            min_dim = min(s_root.size(-1), t_root.size(-1))
            losses.append(F.mse_loss(s_root[..., :min_dim], t_root[..., :min_dim]))

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=self.config.DEVICE)

    def forward(self, student_outputs, teacher_outputs, beta):
        """
        Compute combined distillation loss.

        Args:
            student_outputs: dict with 'recon_loss', 'kl_loss', 'latent_mean', 'features'
            teacher_outputs: same structure for teacher
            beta: current KL annealing weight

        Returns:
            dict with 'total', 'reconstruction', 'kl', 'latent_distill', 'feature_match'
        """
        student_recon = student_outputs['recon_loss']
        student_kl = student_outputs['kl_loss']

        latent_distill_loss = self.compute_latent_distillation_loss(
            student_outputs['latent_mean'],
            teacher_outputs['latent_mean']
        )
        feature_match_loss = self.compute_feature_matching_loss(
            student_outputs['features'],
            teacher_outputs['features']
        )

        total_loss = (
            self.config.ALPHA_RECONSTRUCTION * student_recon +
            beta * self.config.BETA_KL * student_kl +
            self.config.GAMMA_LATENT * latent_distill_loss +
            self.config.DELTA_FEATURE * feature_match_loss
        )

        return {
            'total': total_loss,
            'reconstruction': student_recon.item(),
            'kl': student_kl.item(),
            'latent_distill': latent_distill_loss.item(),
            'feature_match': feature_match_loss.item(),
        }


class MoleculeDatasetForDistillation(Dataset):
    """Dataset that validates molecules against the hgraph vocabulary."""

    def __init__(self, smiles_list, vocab, atom_vocab, max_samples=None):
        _, _, _, MolGraph = _try_import_hgraph()
        self.vocab = vocab
        self.atom_vocab = atom_vocab
        self.valid_smiles = []

        candidates = smiles_list[:max_samples] if max_samples else smiles_list
        print(f"\nValidating {len(candidates)} molecules against vocabulary...")

        for smiles in tqdm(candidates, desc="Validating"):
            try:
                hmol = MolGraph(smiles)
                valid = True
                for node, attr in hmol.mol_tree.nodes(data=True):
                    if attr['label'] not in vocab.vmap:
                        valid = False
                        break
                    for i, s in attr['inter_label']:
                        if (attr['smiles'], s) not in vocab.vmap:
                            valid = False
                            break
                if valid:
                    self.valid_smiles.append(smiles)
            except Exception:
                continue

        print(f"Valid molecules: {len(self.valid_smiles)} / {len(candidates)}")

    def __len__(self):
        return len(self.valid_smiles)

    def __getitem__(self, idx):
        return self.valid_smiles[idx]


def collate_fn_distillation(smiles_batch, vocab, atom_vocab):
    """Collate function: tensorizes a batch of SMILES via MolGraph."""
    _, _, _, MolGraph = _try_import_hgraph()
    return MolGraph.tensorize(smiles_batch, vocab, atom_vocab)


def load_training_data(config, vocab, max_samples=5000):
    """
    Load ChEMBL training SMILES and create a DataLoader.

    Args:
        config: Config object
        vocab: PairVocab object
        max_samples: cap on training molecules

    Returns:
        DataLoader
    """
    _, _, common_atom_vocab, _ = _try_import_hgraph()

    data_file = "data/chembl/all.txt"
    print(f"\nLoading SMILES from: {data_file}")

    with open(data_file, 'r') as f:
        all_smiles = [line.strip() for line in f if line.strip()]

    print(f"Total SMILES: {len(all_smiles)}")

    if max_samples and max_samples < len(all_smiles):
        sampled_smiles = random.sample(all_smiles, max_samples)
    else:
        sampled_smiles = all_smiles

    dataset = MoleculeDatasetForDistillation(
        sampled_smiles, vocab, common_atom_vocab, max_samples=max_samples
    )

    collate_fn = partial(collate_fn_distillation, vocab=vocab, atom_vocab=common_atom_vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"DataLoader: batch_size={config.BATCH_SIZE}, batches={len(dataloader)}")
    return dataloader


class DistillationTrainer:
    """Training loop for knowledge distillation of HierVAE."""

    def __init__(self, config, teacher_model, student_model, vocab):
        self.config = config
        self.teacher = teacher_model
        self.student = student_model
        self.vocab = vocab

        self.criterion = DistillationLoss(config)
        self.optimizer = optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = defaultdict(list)

    def compute_beta(self, epoch, step, steps_per_epoch):
        """Compute KL annealing weight with linear warmup."""
        total_steps = epoch * steps_per_epoch + step
        warmup_steps = self.config.WARMUP_EPOCHS * steps_per_epoch

        if total_steps < warmup_steps:
            beta = self.config.KL_ANNEAL_START + (
                self.config.BETA_KL - self.config.KL_ANNEAL_START
            ) * (total_steps / warmup_steps)
        else:
            beta = self.config.BETA_KL
        return beta

    def extract_features(self, model, batch):
        """
        Run encoder + decoder for one batch, returning losses and features.
        Returns dict with 'recon_loss', 'kl_loss', 'latent_mean', 'features', 'accuracy'.
        """
        graphs, tensors, orders = batch

        tree_tensors, graph_tensors = tensors
        tree_tensors = [t.to(self.config.DEVICE) if isinstance(t, torch.Tensor) else t
                        for t in tree_tensors]
        graph_tensors = [t.to(self.config.DEVICE) if isinstance(t, torch.Tensor) else t
                         for t in graph_tensors]
        tensors = (tree_tensors, graph_tensors)

        root_vecs, tree_vecs, inter_vecs, graph_vecs = model.encoder(tree_tensors, graph_tensors)

        latent_mean = model.R_mean(root_vecs)
        latent_log_var = -torch.abs(model.R_var(root_vecs))

        batch_size = root_vecs.size(0)
        kl_loss = -0.5 * torch.sum(
            1.0 + latent_log_var - latent_mean * latent_mean - torch.exp(latent_log_var)
        ) / batch_size

        epsilon = torch.randn_like(latent_mean)
        latent_vec = latent_mean + torch.exp(latent_log_var / 2) * epsilon

        dec_loss, wacc, iacc, tacc, sacc = model.decoder(
            (latent_vec, latent_vec, latent_vec),
            graphs,
            tensors,
            orders
        )

        return {
            'recon_loss': dec_loss,
            'kl_loss': kl_loss,
            'latent_mean': latent_mean,
            'latent_vec': latent_vec,
            'features': {'root_vecs': root_vecs, 'tree_vecs': tree_vecs},
            'accuracy': {'word': wacc, 'inter': iacc, 'topo': tacc, 'assm': sacc}
        }

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch, returning average losses."""
        self.student.train()
        self.teacher.eval()

        epoch_losses = defaultdict(float)
        steps_per_epoch = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")

        for step, batch in enumerate(pbar):
            beta = self.compute_beta(epoch, step, steps_per_epoch)

            with torch.no_grad():
                teacher_outputs = self.extract_features(self.teacher, batch)

            student_outputs = self.extract_features(self.student, batch)
            losses = self.criterion(student_outputs, teacher_outputs, beta)

            self.optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), self.config.CLIP_NORM)
            self.optimizer.step()

            for key, value in losses.items():
                epoch_losses[key] += value if key == 'total' else value

            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['reconstruction']:.4f}",
                'beta': f"{beta:.4f}"
            })
            self.global_step += 1

        for key in epoch_losses:
            epoch_losses[key] /= steps_per_epoch

        return dict(epoch_losses)

    def save_checkpoint(self, epoch, is_best=False):
        """Save student model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': dict(self.history),
        }
        path = os.path.join(self.config.OUTPUT_DIR, f'student_model_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

        if is_best:
            best_path = os.path.join(self.config.OUTPUT_DIR, 'best_student_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def train(self, dataloader):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("STARTING DISTILLATION TRAINING")
        print("=" * 70)

        for epoch in range(self.config.NUM_EPOCHS):
            losses = self.train_epoch(dataloader, epoch)

            for key, value in losses.items():
                self.history[key].append(value)
            self.history['epoch'].append(epoch)

            print(f"\nEpoch {epoch+1} | "
                  f"Loss={losses['total']:.4f} | "
                  f"Recon={losses['reconstruction']:.4f} | "
                  f"KL={losses['kl']:.4f} | "
                  f"LatentDistill={losses['latent_distill']:.4f}")

            is_best = losses['total'] < self.best_loss
            if is_best:
                self.best_loss = losses['total']

            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(epoch, is_best)

            if epoch >= self.config.WARMUP_EPOCHS:
                self.scheduler.step()

        self.save_checkpoint(self.config.NUM_EPOCHS - 1)
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)


class ModelEvaluator:
    """Evaluation suite for comparing teacher and student generation quality."""

    def __init__(self, config, teacher_model, student_model, vocab):
        self.config = config
        self.teacher = teacher_model
        self.student = student_model
        self.vocab = vocab
        self.oov_stats = {
            'teacher_oov_count': 0, 'teacher_total_attempts': 0,
            'student_oov_count': 0, 'student_total_attempts': 0,
            'oov_motifs': set()
        }

    def safe_sample(self, model, num_samples, model_name='model'):
        """
        Generate molecules with OOV error handling.
        Falls back to batch_size=1 on KeyError.
        """
        generated_smiles = []
        oov_count = 0
        attempts = 0
        batch_size = min(self.config.BATCH_SIZE, 8)

        with torch.no_grad():
            pbar = tqdm(total=num_samples, desc=f"Generating {model_name}")

            while len(generated_smiles) < num_samples and attempts < num_samples * 3:
                try:
                    current_batch = min(batch_size, num_samples - len(generated_smiles))
                    attempts += current_batch
                    smiles_batch = model.sample(current_batch, greedy=True)
                    for smiles in smiles_batch:
                        generated_smiles.append(smiles)
                        pbar.update(1)

                except KeyError as e:
                    oov_count += current_batch
                    self.oov_stats['oov_motifs'].add(str(e))
                    if current_batch > 1:
                        batch_size = 1
                    continue

                except Exception:
                    attempts += current_batch
                    continue

            pbar.close()

        if model_name == 'teacher':
            self.oov_stats['teacher_oov_count'] = oov_count
            self.oov_stats['teacher_total_attempts'] = attempts
        else:
            self.oov_stats['student_oov_count'] = oov_count
            self.oov_stats['student_total_attempts'] = attempts

        return generated_smiles

    def evaluate_validity(self, model, num_samples=1000, model_name='model'):
        """Evaluate fraction of generated SMILES that are valid molecules."""
        model.eval()
        generated_smiles = self.safe_sample(model, num_samples, model_name)

        valid_count = sum(1 for s in generated_smiles if Chem.MolFromSmiles(s) is not None)
        validity_rate = valid_count / len(generated_smiles) if generated_smiles else 0.0

        oov_count = self.oov_stats.get(f'{model_name}_oov_count', 0)
        total_attempts = self.oov_stats.get(f'{model_name}_total_attempts', len(generated_smiles))
        oov_rate = oov_count / total_attempts if total_attempts > 0 else 0.0

        return {
            'validity_rate': validity_rate,
            'valid_count': valid_count,
            'total_count': len(generated_smiles),
            'generated_smiles': generated_smiles,
            'oov_count': oov_count,
            'oov_rate': oov_rate,
            'total_attempts': total_attempts
        }

    def evaluate_uniqueness(self, generated_smiles):
        """Fraction of valid generated SMILES that are unique (canonicalized)."""
        canonical = []
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical.append(Chem.MolToSmiles(mol))
        unique = set(canonical)
        return {
            'uniqueness_rate': len(unique) / len(canonical) if canonical else 0.0,
            'unique_count': len(unique),
            'total_valid': len(canonical)
        }

    def evaluate_reconstruction(self, model, test_smiles, num_samples=100):
        """Measure exact-match and valid reconstruction rates."""
        _, _, common_atom_vocab, MolGraph = _try_import_hgraph()
        model.eval()

        test_sample = random.sample(test_smiles, min(num_samples, len(test_smiles)))
        exact_matches = 0
        valid_reconstructions = 0

        with torch.no_grad():
            for smiles in tqdm(test_sample, desc="Testing reconstruction"):
                try:
                    batch = MolGraph.tensorize([smiles], self.vocab, common_atom_vocab)
                    graphs, tensors, orders = batch
                    tree_tensors, graph_tensors = tensors
                    tree_tensors = [t.to(self.config.DEVICE) if isinstance(t, torch.Tensor) else t
                                    for t in tree_tensors]
                    graph_tensors = [t.to(self.config.DEVICE) if isinstance(t, torch.Tensor) else t
                                     for t in graph_tensors]
                    tensors = (tree_tensors, graph_tensors)
                    batch = (graphs, tensors, orders)

                    reconstructed = model.reconstruct(batch)
                    if reconstructed and len(reconstructed) > 0:
                        recon_smiles = reconstructed[0]
                        mol = Chem.MolFromSmiles(recon_smiles)
                        if mol is not None:
                            valid_reconstructions += 1
                            orig_mol = Chem.MolFromSmiles(smiles)
                            if orig_mol and Chem.MolToSmiles(orig_mol) == Chem.MolToSmiles(mol):
                                exact_matches += 1
                except Exception:
                    continue

        n = len(test_sample)
        return {
            'exact_match_rate': exact_matches / n,
            'valid_reconstruction_rate': valid_reconstructions / n,
            'tested_count': n
        }

    def compute_property_statistics(self, smiles_list):
        """Compute molecular property statistics for a list of SMILES."""
        properties = {'mol_weight': [], 'logp': [], 'qed': [], 'num_atoms': [], 'num_rings': []}

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties['mol_weight'].append(Descriptors.MolWt(mol))
                properties['logp'].append(Descriptors.MolLogP(mol))
                try:
                    properties['qed'].append(QED.qed(mol))
                except Exception:
                    pass
                properties['num_atoms'].append(mol.GetNumAtoms())
                properties['num_rings'].append(Descriptors.RingCount(mol))

        stats = {}
        for prop, values in properties.items():
            if values:
                stats[prop] = {
                    'mean': np.mean(values), 'std': np.std(values),
                    'min': np.min(values), 'max': np.max(values),
                    'values': values
                }
        return stats

    def comprehensive_evaluation(self, test_smiles):
        """Run complete teacher/student evaluation comparison."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)

        results = {}

        for model_name, model in [('teacher', self.teacher), ('student', self.student)]:
            print(f"\n--- {model_name.upper()} MODEL ---")

            validity = self.evaluate_validity(model, self.config.NUM_GEN_SAMPLES, model_name)
            results[f'{model_name}_validity'] = validity
            print(f"  Validity: {validity['validity_rate']*100:.2f}%")

            uniqueness = self.evaluate_uniqueness(validity['generated_smiles'])
            results[f'{model_name}_uniqueness'] = uniqueness
            print(f"  Uniqueness: {uniqueness['uniqueness_rate']*100:.2f}%")

            reconstruction = self.evaluate_reconstruction(model, test_smiles, num_samples=100)
            results[f'{model_name}_reconstruction'] = reconstruction
            print(f"  Exact match: {reconstruction['exact_match_rate']*100:.2f}%")
            print(f"  Valid reconstruction: {reconstruction['valid_reconstruction_rate']*100:.2f}%")

        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        metrics = [
            ('Validity', 'validity', 'validity_rate'),
            ('Uniqueness', 'uniqueness', 'uniqueness_rate'),
            ('Exact Reconstruction', 'reconstruction', 'exact_match_rate'),
        ]
        print(f"\n{'Metric':<30} {'Teacher':>12} {'Student':>12} {'Ratio':>12}")
        print("-" * 68)
        for name, key, field in metrics:
            t_val = results[f'teacher_{key}'][field]
            s_val = results[f'student_{key}'][field]
            ratio = s_val / t_val if t_val > 0 else 0.0
            print(f"{name:<30} {t_val:>12.4f} {s_val:>12.4f} {ratio:>11.2f}x")

        t_params = sum(p.numel() for p in self.teacher.parameters())
        s_params = sum(p.numel() for p in self.student.parameters())
        print(f"{'Model Size (params)':<30} {t_params:>12,} {s_params:>12,} "
              f"{s_params/t_params:>11.2f}x")

        print("\nOOV Statistics:")
        for model_name in ['teacher', 'student']:
            oov = results[f'{model_name}_validity']['oov_count']
            attempts = results[f'{model_name}_validity']['total_attempts']
            rate = results[f'{model_name}_validity']['oov_rate']
            print(f"  {model_name}: {oov} OOV / {attempts} attempts ({rate*100:.2f}%)")

        if self.oov_stats['oov_motifs']:
            print(f"\n  Unique OOV motifs: {len(self.oov_stats['oov_motifs'])}")
            for i, motif in enumerate(list(self.oov_stats['oov_motifs'])[:5]):
                print(f"    {i+1}. {motif}")

        return results


class Visualizer:
    """Visualization utilities for distillation training and evaluation."""

    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distillation Training History', fontsize=16, fontweight='bold')

        epochs = history['epoch']

        axes[0, 0].plot(epochs, history['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, history['reconstruction'], 'g-', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, history['kl'], 'r-', linewidth=2)
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, history['latent_distill'], label='Latent', linewidth=2)
        axes[1, 1].plot(epochs, history['feature_match'], label='Feature', linewidth=2)
        axes[1, 1].set_title('Distillation Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved: {save_path}")
        plt.close('all')

    @staticmethod
    def plot_property_distributions(teacher_props, student_props, save_path=None):
        """Overlay histograms of teacher vs student molecular properties."""
        properties = ['mol_weight', 'logp', 'qed', 'num_atoms']
        titles = ['Molecular Weight', 'LogP', 'QED', 'Number of Atoms']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Property Distributions: Teacher vs Student', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for idx, (prop, title) in enumerate(zip(properties, titles)):
            ax = axes[idx]
            if prop in teacher_props and prop in student_props:
                ax.hist(teacher_props[prop]['values'], bins=30, alpha=0.5,
                        label='Teacher', color='blue', density=True)
                ax.hist(student_props[prop]['values'], bins=30, alpha=0.5,
                        label='Student', color='red', density=True)
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Property distributions saved: {save_path}")
        plt.close('all')

    @staticmethod
    def plot_comparison_bars(evaluation_results, save_path=None):
        """Bar chart comparing teacher vs student key metrics."""
        metrics = [
            ('Validity', 'validity', 'validity_rate'),
            ('Uniqueness', 'uniqueness', 'uniqueness_rate'),
            ('Exact Reconstruction', 'reconstruction', 'exact_match_rate'),
        ]

        metric_names = [m[0] for m in metrics]
        teacher_vals = [evaluation_results[f'teacher_{m[1]}'][m[2]] * 100 for m in metrics]
        student_vals = [evaluation_results[f'student_{m[1]}'][m[2]] * 100 for m in metrics]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        bars1 = ax.bar(x - width/2, teacher_vals, width, label='Teacher', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, student_vals, width, label='Student', color='coral', alpha=0.8)

        ax.set_ylabel('Percentage (%)')
        ax.set_title('Teacher vs Student Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison bar chart saved: {save_path}")
        plt.close('all')


def run_knowledge_distillation_pipeline(config, max_train_samples=5000):
    """
    Execute the complete knowledge distillation pipeline.

    1. Load vocab and teacher
    2. Create student
    3. Load training data
    4. Train via distillation
    5. Evaluate and visualize
    """
    print("\n" + "=" * 70)
    print("KNOWLEDGE DISTILLATION PIPELINE")
    print("=" * 70)

    # Verify prerequisites
    for path in [config.TEACHER_MODEL_PATH, config.VOCAB_PATH]:
        if not os.path.exists(path):
            print(f"\nRequired file not found: {path}")
            print("Cannot run distillation pipeline without pre-trained teacher and vocabulary.")
            return None

    _, _, common_atom_vocab, _ = _try_import_hgraph()

    # Load vocab and models
    vocab = TeacherModelLoader.load_vocab(config.VOCAB_PATH)
    teacher_model = TeacherModelLoader.load_teacher_model(config, vocab)
    student_model = StudentModelFactory.create_student_model(config, vocab, teacher_model)
    StudentModelFactory.compare_architectures(teacher_model, student_model)

    # Load training data
    print("\n[Step 1/5] Loading training data...")
    if not os.path.exists("data/chembl/all.txt"):
        print("Training data not found at data/chembl/all.txt")
        return None
    train_loader = load_training_data(config, vocab, max_samples=max_train_samples)

    # Train
    print("\n[Step 2/5] Training student model...")
    trainer = DistillationTrainer(config, teacher_model, student_model, vocab)
    trainer.train(train_loader)

    # Plot training history
    print("\n[Step 3/5] Plotting training history...")
    Visualizer.plot_training_history(
        trainer.history,
        save_path=os.path.join(config.OUTPUT_DIR, 'training_history.png')
    )

    # Evaluate
    print("\n[Step 4/5] Running evaluation...")
    with open("data/chembl/all.txt", 'r') as f:
        test_smiles = [line.strip() for line in f if line.strip()]
    test_smiles = random.sample(test_smiles, min(1000, len(test_smiles)))

    evaluator = ModelEvaluator(config, teacher_model, student_model, vocab)
    results = evaluator.comprehensive_evaluation(test_smiles)

    # Plot comparison
    print("\n[Step 5/5] Saving comparison plots...")
    Visualizer.plot_comparison_bars(
        results,
        save_path=f'figures/{CHAPTER}/model_comparison.png'
    )

    teacher_props = evaluator.compute_property_statistics(
        [s for s in results['teacher_validity']['generated_smiles']
         if Chem.MolFromSmiles(s) is not None]
    )
    student_props = evaluator.compute_property_statistics(
        [s for s in results['student_validity']['generated_smiles']
         if Chem.MolFromSmiles(s) is not None]
    )
    Visualizer.plot_property_distributions(
        teacher_props, student_props,
        save_path=f'figures/{CHAPTER}/property_distributions.png'
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return results


if __name__ == "__main__":
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Save configuration
    config_dict = {k: str(v) if isinstance(v, torch.device) else v
                   for k, v in config.__dict__.items()
                   if not k.startswith('_') and not callable(v)}
    config_path = os.path.join(config.OUTPUT_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    # Check if hgraph2graph is available
    try:
        _try_import_hgraph()
        hgraph_available = True
    except ImportError as e:
        print(f"\n{e}")
        hgraph_available = False

    if not hgraph_available:
        print("\nAppendix C requires hgraph2graph. Demonstrating pipeline structure only.")
        print("\nDistillation loss components:")
        print("  - ALPHA_RECONSTRUCTION = 1.0  (reconstruction loss weight)")
        print("  - BETA_KL = 0.1               (KL divergence weight)")
        print("  - GAMMA_LATENT = 0.3           (latent space alignment weight)")
        print("  - DELTA_FEATURE = 0.2          (feature matching weight)")
        print("\nTeacher architecture: hidden=250, embed=250, latent=32")
        print("Student architecture:  hidden=128, embed=128, latent=16")
        print(f"\nTo run: ensure hgraph2graph is installed and pre-trained checkpoint")
        print(f"  exists at {config.TEACHER_MODEL_PATH}")
    else:
        print(f"\nUsing device: {config.DEVICE}")
        results = run_knowledge_distillation_pipeline(config, max_train_samples=5000)
        if results is not None:
            print(f"\nAppendix C complete. Outputs saved to figures/{CHAPTER}/ and {config.OUTPUT_DIR}/")
