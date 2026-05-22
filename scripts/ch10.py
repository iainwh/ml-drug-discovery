"""
Chapter 10: Molecular Autoencoders for Generative Chemistry
- Vocab and SmilesPEVocab for character-level and BPE tokenization
- BasicAutoencoder (MLP encoder-decoder for SMILES)
- VAECYC: Variational Autoencoder with Cyclical KL Annealing
- Trainer with early stopping and checkpointing
- LearningRateScheduler
- MolecularEvaluator: reconstruction, latent space, generation metrics
- SMILESDataset PyTorch Dataset
"""

import os
import json
import logging
import math
import random
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import codecs
try:
    from SmilesPE.pretokenizer import atomwise_tokenizer
    from SmilesPE.tokenizer import SPE_Tokenizer
    HAS_SMILESPE = True
except ImportError:
    HAS_SMILESPE = False
    print("SmilesPE not installed. SmilesPEVocab will be unavailable.")

warnings.filterwarnings("ignore")

CHAPTER = "ch10"
RANDOM_SEED = 42

os.makedirs(f"artifacts/{CHAPTER}/checkpoints/autoencoder", exist_ok=True)
os.makedirs(f"artifacts/{CHAPTER}/checkpoints/vae_cyc_spe", exist_ok=True)
os.makedirs(f"artifacts/{CHAPTER}/logs/autoencoder", exist_ok=True)
os.makedirs(f"artifacts/{CHAPTER}/logs/vae_cyc_spe", exist_ok=True)
os.makedirs(f"artifacts/{CHAPTER}/evaluation_results", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

CHECKPOINT_DIR_BASIC = f'artifacts/{CHAPTER}/checkpoints/autoencoder'
LOG_DIR_BASIC = f'artifacts/{CHAPTER}/logs/autoencoder'
CHECKPOINT_DIR_VAECYCSPE = f'artifacts/{CHAPTER}/checkpoints/vae_cyc_spe'
LOG_DIR_VAECYCSPE = f'artifacts/{CHAPTER}/logs/vae_cyc_spe'
SPE_MODEL_PATH = f'data/{CHAPTER}/SPE_ChEMBL.txt'
TRAIN_DATA_PATH = f'data/{CHAPTER}/moses_train.csv'
TEST_DATA_PATH = f'data/{CHAPTER}/moses_test.csv'
EVAL_PATH = f'artifacts/{CHAPTER}/evaluation_results'


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_visualization_style():
    """Configure visualization style."""
    colors = ["#A20025", "#6C8EBF", "#F9CBCD", "#D4E1F5", "#A680B8", "#C3ABD0", "#12AAB5", "#9AC7BF"]
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16


class Vocab:
    """Vocabulary class for character-level SMILES tokenization."""

    def __init__(self, smiles_list: Optional[List[str]] = None):
        self.char_map = {
            'Cl': 'Q',
            'Br': 'W',
            '[nH]': 'X',
            '[H]': 'Y'
        }
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self._token_counts = Counter()
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        if smiles_list:
            self.build_vocab(smiles_list)

    def _apply_substitutions(self, smiles: str, substitution_map: dict) -> str:
        for source, target in substitution_map.items():
            smiles = smiles.replace(source, target)
        return smiles

    def build_vocab(self, smiles_list: List[str]) -> None:
        if not smiles_list:
            raise ValueError("Cannot build vocabulary from empty SMILES list")
        all_chars = set()
        char_counts = Counter()
        for smiles in smiles_list:
            if not isinstance(smiles, str) or not smiles.strip():
                continue
            preprocessed = self._apply_substitutions(smiles, self.char_map)
            chars = list(preprocessed)
            all_chars.update(chars)
            char_counts.update(chars)
        if not all_chars:
            raise ValueError("No valid characters found in SMILES list")
        vocab_chars = self.special_tokens + sorted(list(all_chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(vocab_chars)
        self._token_counts = char_counts
        self.pad_idx = self.char_to_idx['<pad>']
        self.sos_idx = self.char_to_idx['<sos>']
        self.eos_idx = self.char_to_idx['<eos>']
        self.unk_idx = self.char_to_idx['<unk>']

    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        if not self.char_to_idx:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")
        preprocessed = self._apply_substitutions(smiles, self.char_map)
        tokens = []
        if add_special_tokens:
            tokens.append(self.sos_idx)
        for char in preprocessed:
            tokens.append(self.char_to_idx.get(char, self.unk_idx))
        if add_special_tokens:
            tokens.append(self.eos_idx)
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        if not self.idx_to_char:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")
        chars = []
        for token in tokens:
            if token == self.eos_idx:
                break
            if skip_special_tokens and token in [self.pad_idx, self.sos_idx]:
                continue
            char = self.idx_to_char.get(token, '')
            if char:
                chars.append(char)
        smiles = ''.join(chars)
        reverse_map = {v: k for k, v in self.char_map.items()}
        return self._apply_substitutions(smiles, reverse_map)

    def encode_batch(self, smiles_list: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        if not smiles_list:
            return torch.empty(0, 0, dtype=torch.long)
        encoded_sequences = [self.encode(smiles) for smiles in smiles_list]
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_sequences)
        batch_size = len(encoded_sequences)
        padded_tensor = torch.full((batch_size, max_length), self.pad_idx, dtype=torch.long)
        for i, sequence in enumerate(encoded_sequences):
            seq_len = min(len(sequence), max_length)
            padded_tensor[i, :seq_len] = torch.tensor(sequence[:seq_len])
        return padded_tensor

    def get_vocab_stats(self) -> dict:
        return {
            'vocab_size': self.vocab_size,
            'num_special_tokens': len(self.special_tokens),
            'num_characters': self.vocab_size - len(self.special_tokens),
            'most_common_chars': self._token_counts.most_common(10) if self._token_counts else [],
            'special_token_indices': {
                'pad': self.pad_idx,
                'sos': self.sos_idx,
                'eos': self.eos_idx,
                'unk': self.unk_idx
            }
        }

    def validate_smiles_coverage(self, test_smiles: List[str]) -> Tuple[float, List[str]]:
        total_chars = 0
        unknown_chars = set()
        for smiles in test_smiles:
            preprocessed = self._apply_substitutions(smiles, self.char_map)
            for char in preprocessed:
                total_chars += 1
                if char not in self.char_to_idx:
                    unknown_chars.add(char)
        coverage = 1.0 - (len(unknown_chars) / max(1, total_chars))
        return coverage, list(unknown_chars)

    def __len__(self) -> int:
        return self.vocab_size

    def __str__(self) -> str:
        if self.vocab_size == 0:
            return "Vocab(empty - not built yet)"
        return f"Vocab(size={self.vocab_size}, chars={self.vocab_size - len(self.special_tokens)})"


class SmilesPEVocab:
    """SmilesPE-based vocabulary for BPE SMILES tokenization."""

    def __init__(self, spe_model_path: str = None, smiles_list: List[str] = None):
        if not HAS_SMILESPE:
            raise ImportError("SmilesPE is not installed. Install with: pip install SmilesPE")
        spe_vob = codecs.open(spe_model_path)
        self.spe_model_path = spe_model_path
        self.spe_tokenizer = SPE_Tokenizer(spe_vob)
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self._build_vocab_from_spe()
        self._token_counts = Counter()
        print(f"SmilesPE vocabulary loaded: {self.vocab_size} tokens")

    def _build_vocab_from_spe(self):
        with open(self.spe_model_path, "r") as ins:
            lines = ins.readlines()
        spe_vocab = []
        individual_tokens = set()
        for line in lines:
            merged_token = line.strip().replace(' ', '')
            spe_vocab.append(merged_token)
            parts = line.strip().split()
            for part in parts:
                individual_tokens.add(part)
        self.token_to_idx = {}
        self.idx_to_token = {}
        for idx, token in enumerate(self.special_tokens):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
        next_idx = len(self.special_tokens)
        for token in sorted(individual_tokens):
            if token not in self.token_to_idx:
                self.token_to_idx[token] = next_idx
                self.idx_to_token[next_idx] = token
                next_idx += 1
        for token in spe_vocab:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = next_idx
                self.idx_to_token[next_idx] = token
                next_idx += 1
        common_smiles_tokens = [
            '[nH]', '[NH]', '[NH2]', '[NH3+]', '[N+]', '[N-]', '[n+]',
            '[O-]', '[O+]', '[OH]', '[S-]', '[S+]', '[P+]', '[P-]',
            'Cl', 'Br', 'F', 'I', 'B', 'P', 'S', 'N', 'O', 'C',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            '(', ')', '[', ']', '=', '#', '-', '+', '/', '\\', '.', '@',
            'c', 'n', 'o', 's', 'p'
        ]
        for token in common_smiles_tokens:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = next_idx
                self.idx_to_token[next_idx] = token
                next_idx += 1
        self.vocab_size = len(self.token_to_idx)
        self.char_to_idx = self.token_to_idx
        self.idx_to_char = self.idx_to_token

    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        spe_tokens = self.spe_tokenizer.tokenize(smiles).split()
        indices = []
        if add_special_tokens:
            indices.append(self.sos_idx)
        for token in spe_tokens:
            idx = self.token_to_idx.get(token, self.unk_idx)
            indices.append(idx)
            self._token_counts[token] += 1
        if add_special_tokens:
            indices.append(self.eos_idx)
        return indices

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if skip_special_tokens and idx in [self.pad_idx, self.sos_idx]:
                continue
            token = self.idx_to_token.get(idx, '')
            if token and token not in self.special_tokens:
                tokens.append(token)
        return ''.join(tokens)

    def encode_batch(self, smiles_list: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        if not smiles_list:
            return torch.empty(0, 0, dtype=torch.long)
        encoded_sequences = [self.encode(smiles) for smiles in smiles_list]
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_sequences)
        batch_size = len(encoded_sequences)
        padded_tensor = torch.full((batch_size, max_length), self.pad_idx, dtype=torch.long)
        for i, sequence in enumerate(encoded_sequences):
            seq_len = min(len(sequence), max_length)
            padded_tensor[i, :seq_len] = torch.tensor(sequence[:seq_len])
        return padded_tensor

    def get_vocab_stats(self) -> Dict:
        num_special = len(self.special_tokens)
        num_subwords = self.vocab_size - num_special
        most_common = self._token_counts.most_common(20)
        avg_token_length = np.mean([len(token) for token in self.token_to_idx.keys()
                                    if token not in self.special_tokens])
        return {
            'vocab_size': self.vocab_size,
            'num_special_tokens': num_special,
            'num_subword_tokens': num_subwords,
            'most_common_tokens': most_common,
            'average_token_length': avg_token_length,
            'special_token_indices': {
                'pad': self.pad_idx,
                'sos': self.sos_idx,
                'eos': self.eos_idx,
                'unk': self.unk_idx
            }
        }

    def __len__(self) -> int:
        return self.vocab_size

    def __str__(self) -> str:
        return f"SmilesPEVocab(size={self.vocab_size})"


class AbstractMolecularAutoencoder(nn.Module, ABC):
    """Abstract base class for molecular autoencoders."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def loss_function(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def generate_from_latent(self, z: torch.Tensor, vocab=None) -> List[str]:
        pass

    def interpolate_molecules(self, x1: torch.Tensor, x2: torch.Tensor,
                              steps: int = 10, vocab=None) -> List[List[str]]:
        self.eval()
        with torch.no_grad():
            if self.__class__.__name__ == 'VAECYC':
                mu1, _ = self.encode(x1)
                mu2, _ = self.encode(x2)
                z1, z2 = mu1, mu2
            else:
                z1 = self.encode(x1)
                z2 = self.encode(x2)
            interpolated = []
            for alpha in np.linspace(0, 1, steps):
                z_interp = (1 - alpha) * z1 + alpha * z2
                smiles = self.generate_from_latent(z_interp, vocab)
                interpolated.append(smiles)
        return interpolated

    def get_latent_representation(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if self.__class__.__name__ == 'VAECYC':
                mu, _ = self.encode(x)
                z = mu
            else:
                z = self.encode(x)
            return z.cpu().numpy()

    def _pad_or_truncate(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        current_length = x.size(1)
        if current_length > self.max_length:
            return x[:, :self.max_length]
        elif current_length < self.max_length:
            padding = torch.zeros(
                batch_size,
                self.max_length - current_length,
                dtype=torch.long,
                device=x.device
            )
            return torch.cat([x, padding], dim=1)
        return x


class BasicAutoencoder(AbstractMolecularAutoencoder):
    """Basic MLP Autoencoder for SMILES molecular representation learning."""

    def __init__(self, vocab_size: int, latent_dim: int = 64, max_length: int = 100,
                 embed_dim: int = 128, hidden_dims: List[int] = [512, 256]):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        encoder_layers = []
        input_dim = embed_dim * max_length
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        input_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        decoder_layers.extend([
            nn.Linear(input_dim, vocab_size * max_length),
            nn.Unflatten(1, (max_length, vocab_size))
        ])
        self.decoder = nn.Sequential(*decoder_layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self._pad_or_truncate(x)
        embedded = self.embedding(x)
        embedded_flat = embedded.view(batch_size, -1)
        z = self.encoder(embedded_flat)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def loss_function(self, recon: torch.Tensor, target: torch.Tensor,
                      z: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        target = self._pad_or_truncate(target)
        recon_flat = recon.view(-1, self.vocab_size)
        target_flat = target.view(-1)
        recon_loss = F.cross_entropy(recon_flat, target_flat, ignore_index=0)
        with torch.no_grad():
            perplexity = torch.exp(recon_loss).clamp(max=1000.0).item()
        return {
            'loss': recon_loss,
            'recon_loss': recon_loss,
            'perplexity': perplexity
        }

    def generate_from_latent(self, z: torch.Tensor, vocab=None) -> List[str]:
        self.eval()
        with torch.no_grad():
            logits = self.decode(z)
            tokens = torch.argmax(logits, dim=-1)
            generated = []
            for i in range(z.size(0)):
                if vocab is not None:
                    smiles = vocab.decode(tokens[i].cpu().tolist())
                    generated.append(smiles)
                else:
                    generated.append(tokens[i].cpu().tolist())
        return generated


class VAECYC(AbstractMolecularAutoencoder):
    """Variational Autoencoder with Cyclical KL Annealing for Molecular Generation."""

    def __init__(self, vocab: Vocab, latent_dim: int = 64, max_length: int = 100,
                 embed_dim: int = 128, hidden_dim: int = 512, encoder_num_layers: int = 2,
                 encoder_bidirectional: bool = True, encoder_dropout: float = 0.2,
                 decoder_bidirectional: bool = False, decoder_num_layers: int = 4,
                 decoder_dropout: float = 0.2, max_kl_weight: float = 0.6,
                 n_cycles: int = 20, cycle_ratio: float = 0.7, word_dropout: float = 0.2,
                 free_bits: float = 0.25, max_epochs: int = 10, dataset_size: int = None,
                 batch_size: int = None):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.encoder_num_layers = encoder_num_layers
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_dropout = encoder_dropout
        self.decoder_num_layers = decoder_num_layers
        self.decoder_bidirectional = decoder_bidirectional
        self.decoder_dropout = decoder_dropout
        self.max_length = max_length
        self.max_kl_weight = max_kl_weight
        self.n_cycles = n_cycles
        self.cycle_ratio = cycle_ratio
        self.word_dropout = word_dropout
        self.free_bits = free_bits
        self.max_epochs = max_epochs
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=vocab.pad_idx
        )
        self.encoder = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.encoder_num_layers,
            batch_first=True,
            bidirectional=self.encoder_bidirectional,
            dropout=self.encoder_dropout if self.encoder_num_layers > 1 else 0
        )
        encoder_output_dim = self.hidden_dim * (2 if self.encoder_bidirectional else 1)
        self.fc_mu = nn.Linear(encoder_output_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, self.latent_dim)
        self.decoder = nn.GRU(
            input_size=self.embed_dim + self.latent_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.decoder_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=self.decoder_dropout if self.decoder_num_layers > 1 else 0
        )
        self.latent_to_hidden = nn.Linear(
            self.latent_dim,
            self.hidden_dim * self.decoder_num_layers
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def encode(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        embedded = self.embedding(x)
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        output, hidden = self.encoder(embedded)
        if self.encoder_bidirectional:
            hidden = hidden.view(self.encoder_num_layers, 2, batch_size, self.hidden_dim)
            final_hidden = hidden[-1]
            combined_hidden = torch.cat([final_hidden[0], final_hidden[1]], dim=1)
        else:
            combined_hidden = hidden[-1]
        mu = self.fc_mu(combined_hidden)
        logvar = self.fc_logvar(combined_hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        h_init = self.latent_to_hidden(z)
        h = h_init.view(batch_size, self.decoder_num_layers, self.hidden_dim)
        h = h.transpose(0, 1).contiguous()
        embedded = self.embedding(x)
        z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, self.latent_dim)
        decoder_input = torch.cat([embedded, z_expanded], dim=2)
        output, _ = self.decoder(decoder_input, h)
        logits = self.output_projection(output)
        return logits

    def forward(self, x: torch.Tensor, target: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            self.step_count += 1
        x_dropout = x
        if self.training and self.word_dropout > 0:
            dropout_mask = torch.rand_like(x.float()) < self.word_dropout
            x_dropout = x.masked_fill(dropout_mask, self.vocab.unk_idx)
        mu, logvar = self.encode(x_dropout, lengths)
        z = self.reparameterize(mu, logvar)
        decoder_input = target[:, :-1]
        logits = self.decode(decoder_input, z)
        return logits, mu, logvar

    def get_kl_weight(self) -> float:
        step = self.step_count.item()
        if self.dataset_size is None or self.batch_size is None:
            return self.max_kl_weight
        total_expected_steps = (self.dataset_size / self.batch_size) * self.max_epochs
        cycle_steps = max(1, int(total_expected_steps // self.n_cycles))
        cycle_position = (step - 1) % cycle_steps
        progress = cycle_position / cycle_steps
        if progress <= self.cycle_ratio:
            weight = (progress / self.cycle_ratio) * self.max_kl_weight
        else:
            weight = self.max_kl_weight
        return min(weight, self.max_kl_weight)

    def loss_function(self, logits: torch.Tensor, target: torch.Tensor,
                      mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_shifted = target[:, 1:]
        logits_flat = logits.reshape(-1, self.vocab_size)
        target_flat = target_shifted.reshape(-1)
        recon_loss = F.cross_entropy(
            logits_flat, target_flat,
            ignore_index=self.vocab.pad_idx,
            reduction='mean'
        )
        free_bits = self.free_bits if hasattr(self, 'free_bits') else 0.25
        kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0:
            kl_per_dim = kl_divergence.mean(dim=0)
            kl_per_dim = torch.maximum(kl_per_dim, torch.tensor(free_bits, device=kl_per_dim.device))
            kl_loss = kl_per_dim.sum()
        else:
            kl_loss = kl_divergence.sum(dim=1).mean()
        kl_weight = self.get_kl_weight()
        total_loss = recon_loss + kl_weight * kl_loss
        with torch.no_grad():
            perplexity = torch.exp(recon_loss).clamp(max=1000.0)
            kl_per_dim_avg = kl_divergence.mean()
            if free_bits > 0:
                active_dims = (kl_divergence.mean(dim=0) > free_bits).sum().float()
            else:
                active_dims = torch.tensor(self.latent_dim, dtype=torch.float)
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_weight': torch.tensor(kl_weight),
            'perplexity': perplexity,
            'kl_per_dim': kl_per_dim_avg,
            'active_dims': active_dims,
        }

    def generate_from_latent(self, z: torch.Tensor, vocab=None, max_length: int = 100,
                             temperature: float = 1.0) -> List[str]:
        self.eval()
        vocab = vocab or self.vocab
        batch_size = z.size(0)
        device = z.device
        with torch.no_grad():
            generated_smiles = []
            for i in range(batch_size):
                sequence = [self.vocab.sos_idx]
                current_latent = z[i:i+1]
                h = self.latent_to_hidden(current_latent)
                h = h.view(1, self.decoder_num_layers, self.hidden_dim)
                h = h.transpose(0, 1).contiguous()
                for step in range(max_length - 1):
                    current_token_idx = sequence[-1]
                    current_token = torch.tensor([[current_token_idx]], device=device)
                    embedded = self.embedding(current_token)
                    z_current = current_latent.unsqueeze(1)
                    decoder_input = torch.cat([embedded, z_current], dim=2)
                    output, h = self.decoder(decoder_input, h)
                    logits = self.output_projection(output)
                    token_logits = logits[0, 0]
                    if temperature != 1.0:
                        token_logits = token_logits / temperature
                    probs = F.softmax(token_logits, dim=0)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    sequence.append(next_token)
                    if next_token == self.vocab.eos_idx:
                        break
                try:
                    smiles = vocab.decode(sequence)
                    generated_smiles.append(smiles)
                except Exception:
                    generated_smiles.append("")
        return generated_smiles

    def interpolate_molecules(self, x1: torch.Tensor, x2: torch.Tensor,
                              steps: int = 10, vocab=None,
                              temperature: float = 1.0) -> List[List[str]]:
        self.eval()
        vocab = vocab or self.vocab
        with torch.no_grad():
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)
            interpolated = []
            for alpha in np.linspace(0, 1, steps):
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                smiles = self.generate_from_latent(
                    z_interp, vocab=vocab, max_length=self.max_length, temperature=temperature
                )
                interpolated.append(smiles)
        return interpolated


class SMILESDataset(Dataset):
    """PyTorch Dataset for SMILES strings."""

    def __init__(self, smiles_list: List[str], vocab, max_length: int = 100):
        self.vocab = vocab
        self.max_length = max_length
        self.is_smilespe = isinstance(vocab, SmilesPEVocab)
        self.encoded_data = []
        self.valid_smiles = []
        print(f"Processing {len(smiles_list)} SMILES strings...")
        for smiles in smiles_list:
            if len(smiles) > 0 and len(smiles) < max_length - 2:
                try:
                    encoded = vocab.encode(smiles)
                    if len(encoded) <= max_length:
                        self.encoded_data.append(encoded)
                        self.valid_smiles.append(smiles)
                except Exception:
                    continue
        print(f"Dataset contains {len(self.encoded_data)} valid SMILES")
        if self.encoded_data:
            avg_len = np.mean([len(seq) for seq in self.encoded_data])
            print(f"Average sequence length: {avg_len:.1f} tokens")

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        encoded = self.encoded_data[idx]
        input_seq = encoded[:-1]
        target_seq = encoded[1:]
        input_padded = input_seq + [self.vocab.pad_idx] * (self.max_length - len(input_seq))
        target_padded = target_seq + [self.vocab.pad_idx] * (self.max_length - len(target_seq))
        return {
            'input': torch.tensor(input_padded[:self.max_length], dtype=torch.long),
            'target': torch.tensor(target_padded[:self.max_length], dtype=torch.long),
            'length': len(input_seq),
            'smiles': self.valid_smiles[idx]
        }


class LearningRateScheduler:
    """Custom learning rate scheduler with multiple modes."""

    def __init__(self, optimizer: torch.optim.Optimizer, mode: str = 'step',
                 step_size: int = 50, gamma: float = 0.5, patience: int = 10):
        self.optimizer = optimizer
        self.mode = mode
        self.step_size = step_size
        self.gamma = gamma
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.initial_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch: int, val_loss: Optional[float] = None):
        if self.mode == 'step':
            if epoch > 0 and epoch % self.step_size == 0:
                self._reduce_lr()
        elif self.mode == 'exponential':
            new_lr = self.initial_lr * (self.gamma ** epoch)
            self._set_lr(new_lr)
        elif self.mode == 'cosine':
            new_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / 300))
            self._set_lr(new_lr)
        elif self.mode == 'plateau' and val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self._reduce_lr()
                self.epochs_without_improvement = 0

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.gamma

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class Trainer:
    """Comprehensive trainer for molecular autoencoders."""

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda',
                 max_epochs: int = 300,
                 gradient_clip_val: float = 5.0,
                 early_stopping_patience: int = 10,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs',
                 save_every_n_epochs: int = 5,
                 verbose: bool = True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        self.save_every_n_epochs = save_every_n_epochs
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_losses': [], 'val_losses': [],
            'train_recon_losses': [], 'val_recon_losses': [],
            'train_perplexities': [], 'val_perplexities': [],
            'learning_rates': [], 'epoch_times': []
        }
        self.callbacks = []

    def _setup_logging(self):
        log_file = self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        if self.verbose:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            self.logger.addHandler(ch)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: Optional[int] = None) -> Dict:
        num_epochs = num_epochs or self.max_epochs
        self.logger.info(f"Starting training for {num_epochs} epochs")
        for epoch in range(self.current_epoch, min(self.current_epoch + num_epochs, self.max_epochs)):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader, epoch)
            self._update_history(train_metrics, val_metrics)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['learning_rates'].append(current_lr)
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_times'].append(epoch_time)
            self._log_epoch_results(epoch, train_metrics, val_metrics, current_lr, epoch_time)
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self._save_checkpoint('best_model.pt', is_best=True)
            else:
                self.epochs_without_improvement += 1
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            for callback in self.callbacks:
                callback(self, epoch, train_metrics, val_metrics)
        self._save_checkpoint('final_model.pt')
        self._save_training_history()
        self.logger.info("Training completed!")
        return self.training_history

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_loss = epoch_recon_loss = epoch_perplexity = 0.0
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.max_epochs} [Train]',
                    disable=not self.verbose)
        for batch_idx, batch in enumerate(pbar):
            input_seq = batch['input'].to(self.device)
            target_seq = batch['target'].to(self.device)
            self.optimizer.zero_grad()
            if self.model.__class__.__name__ == 'VAECYC':
                recon, mu, logvar = self.model(input_seq, target_seq)
                loss_dict = self.model.loss_function(recon, target_seq, mu, logvar)
            else:
                recon, z = self.model(input_seq)
                loss_dict = self.model.loss_function(recon, target_seq, z)
            loss = loss_dict['loss']
            loss.backward()
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict['recon_loss'].item()
            if 'perplexity' in loss_dict:
                val = loss_dict['perplexity']
                epoch_perplexity += val.item() if isinstance(val, torch.Tensor) else val
            self.global_step += 1
        metrics = {
            'loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon_loss / num_batches
        }
        if epoch_perplexity > 0:
            metrics['perplexity'] = epoch_perplexity / num_batches
        return metrics

    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        epoch_loss = epoch_recon_loss = epoch_perplexity = 0.0
        num_batches = len(val_loader)
        with torch.no_grad():
            for batch in val_loader:
                input_seq = batch['input'].to(self.device)
                target_seq = batch['target'].to(self.device)
                if self.model.__class__.__name__ == 'VAECYC':
                    recon, mu, logvar = self.model(input_seq, target_seq)
                    loss_dict = self.model.loss_function(recon, target_seq, mu, logvar)
                else:
                    recon, z = self.model(input_seq)
                    loss_dict = self.model.loss_function(recon, target_seq, z)
                epoch_loss += loss_dict['loss'].item()
                epoch_recon_loss += loss_dict['recon_loss'].item()
                if 'perplexity' in loss_dict:
                    val = loss_dict['perplexity']
                    epoch_perplexity += val.item() if isinstance(val, torch.Tensor) else val
        return {
            'loss': epoch_loss / num_batches,
            'recon_loss': epoch_recon_loss / num_batches,
            'perplexity': epoch_perplexity / num_batches if epoch_perplexity > 0 else None
        }

    def _update_history(self, train_metrics: Dict, val_metrics: Dict):
        self.training_history['train_losses'].append(train_metrics['loss'])
        self.training_history['val_losses'].append(val_metrics['loss'])
        self.training_history['train_recon_losses'].append(train_metrics['recon_loss'])
        self.training_history['val_recon_losses'].append(val_metrics['recon_loss'])
        if 'perplexity' in train_metrics and train_metrics['perplexity']:
            self.training_history['train_perplexities'].append(train_metrics['perplexity'])
        if 'perplexity' in val_metrics and val_metrics['perplexity']:
            self.training_history['val_perplexities'].append(val_metrics['perplexity'])

    def _log_epoch_results(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        log_str = (
            f"Epoch {epoch+1}/{self.max_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"LR: {lr:.6f} | Time: {epoch_time:.1f}s"
        )
        self.logger.info(log_str)

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        model_config = {}
        if isinstance(self.model, BasicAutoencoder):
            model_config = {
                'model_type': 'basic',
                'vocab_size': self.model.vocab_size,
                'latent_dim': self.model.latent_dim,
                'max_length': self.model.max_length,
                'embed_dim': self.model.embed_dim,
                'hidden_dims': self.model.hidden_dims
            }
        elif isinstance(self.model, VAECYC):
            model_config = {
                'model_type': 'vae-cyc',
                'vocab': self.model.vocab,
                'latent_dim': self.model.latent_dim,
                'embed_dim': self.model.embed_dim,
                'hidden_dim': self.model.hidden_dim,
                'max_length': self.model.max_length,
            }
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'epochs_without_improvement': self.epochs_without_improvement,
            'model_config': model_config,
        }
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']

    def _save_training_history(self):
        history_file = self.log_dir / 'training_history.json'
        json_history = {}
        for key, values in self.training_history.items():
            json_history[key] = [float(v) if isinstance(v, (np.ndarray, np.floating)) else v
                                  for v in values]
        with open(history_file, 'w') as f:
            json.dump(json_history, f, indent=2)

    def plot_training_curves(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        epochs = range(1, len(self.training_history['train_losses']) + 1)
        ax = axes[0, 0]
        ax.plot(epochs, self.training_history['train_losses'], label='Train', linewidth=2)
        ax.plot(epochs, self.training_history['val_losses'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        ax.plot(epochs, self.training_history['train_recon_losses'], label='Train', linewidth=2)
        ax.plot(epochs, self.training_history['val_recon_losses'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title('Reconstruction Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        ax.plot(epochs, self.training_history['learning_rates'], linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        ax.plot(epochs, self.training_history['epoch_times'], linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Training Time per Epoch')
        ax.grid(True, alpha=0.3)
        plt.suptitle('Training Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        return fig

    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)

    def get_best_model(self) -> nn.Module:
        best_checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model


class MolecularEvaluator:
    """Comprehensive evaluation suite for molecular autoencoders."""

    def __init__(self, model: nn.Module, vocab, device: str = 'cuda'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model.eval()

    def evaluate_reconstruction(self, test_smiles: List[str],
                                num_samples: int = 1000,
                                verbose: bool = True) -> Dict:
        if len(test_smiles) > num_samples:
            test_smiles = random.sample(test_smiles, num_samples)
        metrics = {
            'token_accuracies': [], 'exact_matches': [],
            'tanimoto_similarities': [], 'valid_reconstructions': [], 'examples': []
        }
        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(test_smiles, desc="Evaluating reconstruction") if verbose else test_smiles
            for smiles in iterator:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                try:
                    tokens = self.vocab.encode(smiles)
                    if len(tokens) > self.model.max_length:
                        continue
                    input_tensor = torch.tensor([tokens[:-1]], device=self.device)
                    target_tensor = torch.tensor([tokens[1:]], device=self.device)
                    if self.model.__class__.__name__ == 'VAECYC':
                        recon, mu, logvar = self.model(input_tensor, target_tensor)
                    else:
                        recon, z = self.model(input_tensor)
                    pred_tokens = torch.argmax(recon, dim=-1)
                    pred_tokens_trimmed = pred_tokens[:, :target_tensor.size(1)]
                    target_trimmed = target_tensor[:, :pred_tokens.size(1)]
                    mask = target_trimmed != self.vocab.pad_idx
                    if mask.sum() > 0:
                        correct = (pred_tokens_trimmed == target_trimmed).float()
                        token_acc = (correct * mask).sum() / mask.sum()
                        metrics['token_accuracies'].append(token_acc.item())
                    reconstructed_smiles = self.vocab.decode(pred_tokens[0].cpu().tolist())
                    exact_match = smiles == reconstructed_smiles
                    metrics['exact_matches'].append(exact_match)
                    tanimoto = self._calculate_tanimoto_similarity(smiles, reconstructed_smiles)
                    metrics['tanimoto_similarities'].append(tanimoto)
                    recon_mol = Chem.MolFromSmiles(reconstructed_smiles)
                    metrics['valid_reconstructions'].append(recon_mol is not None)
                    if len(metrics['examples']) < 10:
                        metrics['examples'].append({
                            'original': smiles,
                            'reconstructed': reconstructed_smiles,
                            'tanimoto': tanimoto,
                            'exact_match': exact_match
                        })
                except Exception:
                    continue
        return {
            'num_evaluated': len(metrics['token_accuracies']),
            'avg_token_accuracy': np.mean(metrics['token_accuracies']) if metrics['token_accuracies'] else 0,
            'exact_match_rate': np.mean(metrics['exact_matches']) if metrics['exact_matches'] else 0,
            'avg_tanimoto_similarity': np.mean(metrics['tanimoto_similarities']) if metrics['tanimoto_similarities'] else 0,
            'valid_reconstruction_rate': np.mean(metrics['valid_reconstructions']) if metrics['valid_reconstructions'] else 0,
            'examples': metrics['examples']
        }

    def evaluate_generation(self, num_samples: int = 1000) -> Dict:
        generated_smiles = []
        self.model.eval()
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="Generating molecules"):
                z = torch.randn(1, self.model.latent_dim, device=self.device)
                smiles_list = self.model.generate_from_latent(z, self.vocab)
                if smiles_list:
                    generated_smiles.append(smiles_list[0])
        valid_molecules = []
        valid_smiles = []
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_molecules.append(mol)
                valid_smiles.append(smiles)
        validity_rate = len(valid_molecules) / len(generated_smiles) if generated_smiles else 0
        unique_smiles = set(valid_smiles)
        uniqueness_rate = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
        return {
            'num_generated': len(generated_smiles),
            'validity_rate': validity_rate,
            'num_valid': len(valid_molecules),
            'uniqueness_rate': uniqueness_rate,
            'num_unique': len(unique_smiles),
            'sample_molecules': valid_smiles[:10]
        }

    def _calculate_tanimoto_similarity(self, smiles1: str, smiles2: str) -> float:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
        fp1 = morgan_gen.GetFingerprint(mol1)
        fp2 = morgan_gen.GetFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def _prepare_for_json(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(v) for v in obj]
        return obj

    def create_evaluation_report(self, test_smiles: List[str],
                                 save_path: Optional[str] = None) -> Dict:
        print("=" * 60)
        print("MOLECULAR AUTOENCODER EVALUATION REPORT")
        print("=" * 60)
        print("\n1. Evaluating Reconstruction Quality...")
        recon_results = self.evaluate_reconstruction(test_smiles, num_samples=1000)
        print(f"   Token Accuracy: {recon_results['avg_token_accuracy']:.3f}")
        print(f"   Exact Match Rate: {recon_results['exact_match_rate']:.3f}")
        print(f"   Tanimoto Similarity: {recon_results['avg_tanimoto_similarity']:.3f}")
        print(f"   Valid Reconstruction Rate: {recon_results['valid_reconstruction_rate']:.3f}")
        print("\n2. Evaluating Generation Quality...")
        gen_results = self.evaluate_generation(num_samples=200)
        print(f"   Validity Rate: {gen_results['validity_rate']:.3f}")
        print(f"   Uniqueness Rate: {gen_results['uniqueness_rate']:.3f}")
        report = {
            'reconstruction': recon_results,
            'generation': gen_results,
            'timestamp': str(np.datetime64('now'))
        }
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self._prepare_for_json(report), f, indent=2)
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        return report


def create_vocab(tokenizer_type: str, train_smiles: List[str], spe_model_path: str = None):
    if tokenizer_type.lower() == 'char':
        print("Building character-based vocabulary...")
        return Vocab(train_smiles)
    elif tokenizer_type.lower() == 'spe':
        print("Loading SmilesPE vocabulary...")
        return SmilesPEVocab(spe_model_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def load_data(train_path: str, test_path: str, sample_size: int = None) -> tuple:
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_smiles = train_df['SMILES'].dropna().tolist()
    test_smiles = test_df['SMILES'].dropna().tolist()
    print(f"Loaded {len(train_smiles)} training and {len(test_smiles)} test SMILES")
    if sample_size:
        train_smiles = random.sample(train_smiles, min(sample_size, len(train_smiles)))
        test_smiles = random.sample(test_smiles, min(sample_size // 10, len(test_smiles)))
        print(f"Sampled to {len(train_smiles)} training and {len(test_smiles)} test SMILES")
    return train_smiles, test_smiles


def create_data_loaders(train_smiles: List[str], test_smiles: List[str],
                        vocab, batch_size: int = 128,
                        max_length: int = 100) -> tuple:
    val_size = len(test_smiles) // 2
    val_smiles = test_smiles[:val_size]
    test_smiles_split = test_smiles[val_size:]
    train_dataset = SMILESDataset(train_smiles, vocab, max_length)
    val_dataset = SMILESDataset(val_smiles, vocab, max_length)
    test_dataset = SMILESDataset(test_smiles_split, vocab, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def initialize_model(model_type: str, vocab, config: Dict) -> nn.Module:
    if model_type.lower() in ['basic', 'basicautoencoder']:
        return BasicAutoencoder(
            vocab_size=vocab.vocab_size,
            latent_dim=config.get('latent_dim', 64),
            max_length=config.get('max_length', 150),
            embed_dim=config.get('embed_dim', 128),
            hidden_dims=config.get('hidden_dims', [512, 256])
        )
    elif model_type.lower() in ['vae-cyc', 'vaecyc']:
        dataset_size = config.get('dataset_size')
        if dataset_size is None and 'train_path' in config:
            try:
                train_df = pd.read_csv(config['train_path'])
                dataset_size = len(train_df)
            except Exception:
                dataset_size = None
        return VAECYC(
            vocab=vocab,
            latent_dim=config.get('latent_dim', 64),
            embed_dim=config.get('embed_dim', 128),
            hidden_dim=config.get('hidden_dim', 512),
            max_length=config.get('max_length', 100),
            word_dropout=config.get('word_dropout', 0.2),
            cycle_ratio=config.get('cycle_ratio', 0.7),
            max_kl_weight=config.get('max_kl_weight', 0.6),
            n_cycles=config.get('n_cycles', 10),
            encoder_bidirectional=config.get('encoder_bidirectional', True),
            decoder_num_layers=config.get('decoder_num_layers', 4),
            encoder_num_layers=config.get('encoder_num_layers', 2),
            decoder_dropout=config.get('decoder_dropout', 0.2),
            encoder_dropout=config.get('encoder_dropout', 0.2),
            free_bits=config.get('free_bits', 0.1),
            max_epochs=config.get('max_epochs', 300),
            dataset_size=dataset_size,
            batch_size=config.get('batch_size', 256),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_molecular_autoencoder(config: Dict) -> Dict:
    """Unified training function for molecular autoencoders."""
    if 'model_type' not in config:
        raise ValueError("Config must include 'model_type' key")
    model_type = config['model_type']
    tokenizer_type = config.get('tokenizer_type', 'char')
    print(f"Training {model_type.upper()} Molecular Autoencoder")
    set_random_seeds(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_smiles, test_smiles = load_data(
        config['train_path'], config['test_path'], config.get('sample_size', None)
    )
    vocab = create_vocab(tokenizer_type, train_smiles, config.get('spe_model_path', None))
    train_loader, val_loader, test_loader = create_data_loaders(
        train_smiles, test_smiles, vocab,
        batch_size=config.get('batch_size', 128),
        max_length=config.get('max_length', 100)
    )
    model = initialize_model(model_type, vocab, config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 0.0001),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    trainer = Trainer(
        model=model, optimizer=optimizer, device=device,
        max_epochs=config.get('max_epochs', 300),
        gradient_clip_val=config.get('gradient_clip_val', 5.0),
        early_stopping_patience=config.get('early_stopping_patience', 30),
        checkpoint_dir=config.get('checkpoint_dir', CHECKPOINT_DIR_BASIC),
        log_dir=config.get('log_dir', LOG_DIR_BASIC),
        save_every_n_epochs=config.get('save_every_n_epochs', 10),
        verbose=config.get('verbose', True)
    )
    if config.get('use_lr_scheduler', False):
        lr_scheduler = LearningRateScheduler(
            optimizer,
            mode=config.get('lr_schedule_mode', 'step'),
            step_size=config.get('lr_step_size', 50),
            gamma=config.get('lr_gamma', 0.5)
        )
        def lr_callback(trainer, epoch, train_metrics, val_metrics):
            lr_scheduler.step(epoch, val_metrics['loss'])
        trainer.add_callback(lr_callback)
    training_history = trainer.train(train_loader, val_loader)
    best_model = trainer.get_best_model()
    evaluator = MolecularEvaluator(best_model, vocab, device)
    test_smiles_for_eval = [item['smiles'] for item in test_loader.dataset]
    eval_report = evaluator.create_evaluation_report(
        test_smiles_for_eval,
        save_path=Path(config.get('log_dir', LOG_DIR_BASIC)) / 'evaluation_report.json'
    )
    results = {
        'config': config,
        'vocab_size': vocab.vocab_size,
        'training_history': training_history,
        'evaluation_report': evaluator._prepare_for_json(eval_report)
    }
    results_path = Path(config.get('log_dir', LOG_DIR_BASIC)) / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def analyze_smiles_dataset(smiles_list: List[str], name: str = "Dataset") -> Dict:
    """Analyze a dataset of SMILES strings."""
    print(f"\n{'='*60}")
    print(f"Analyzing {name}")
    print(f"{'='*60}")
    lengths = [len(s) for s in smiles_list]
    valid_count = 0
    valid_molecules = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1
            valid_molecules.append(mol)
    all_chars = Counter()
    for smiles in smiles_list:
        all_chars.update(smiles)
    stats = {
        'num_molecules': len(smiles_list),
        'validity_rate': valid_count / len(smiles_list) if smiles_list else 0,
        'length_stats': {
            'mean': np.mean(lengths) if lengths else 0,
            'std': np.std(lengths) if lengths else 0,
            'min': np.min(lengths) if lengths else 0,
            'max': np.max(lengths) if lengths else 0,
        },
        'unique_chars': len(all_chars),
        'most_common_chars': all_chars.most_common(10)
    }
    print(f"Total molecules: {stats['num_molecules']:,}")
    print(f"Validity rate: {stats['validity_rate']:.2%}")
    if lengths:
        print(f"Length: {stats['length_stats']['mean']:.1f} +/- {stats['length_stats']['std']:.1f}")
    return stats


if __name__ == "__main__":
    setup_visualization_style()
    set_random_seeds(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Demo: Vocabulary on example SMILES
    demo_smiles = [
        'CC(=O)Nc1ccc(O)cc1',
        'c1ccccc1',
        'CCO',
        'c1ccccc1C(=O)O',
        'NC(=O)c1ccccc1',
        'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
        'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',
        'OC(=O)c1ccccc1O'
    ]
    print("\nBuilding character vocabulary from demo SMILES...")
    vocab = Vocab(demo_smiles)
    print(f"Vocabulary: {vocab}")
    stats = vocab.get_vocab_stats()
    print(f"Vocab size: {stats['vocab_size']}")
    test_smi = demo_smiles[0]
    tokens = vocab.encode(test_smi)
    reconstructed = vocab.decode(tokens)
    print(f"Original: {test_smi}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Match: {test_smi == reconstructed}")

    # Check if training data exists
    if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH):
        print(f"\nTraining data found. Running quick test training...")
        test_config = {
            'model_type': 'basic',
            'tokenizer_type': 'char',
            'train_path': TRAIN_DATA_PATH,
            'test_path': TEST_DATA_PATH,
            'latent_dim': 32,
            'embed_dim': 64,
            'hidden_dims': [256, 128],
            'max_length': 100,
            'batch_size': 64,
            'learning_rate': 0.001,
            'max_epochs': 3,
            'early_stopping_patience': 5,
            'save_every_n_epochs': 1,
            'sample_size': 500,
            'checkpoint_dir': CHECKPOINT_DIR_BASIC,
            'log_dir': LOG_DIR_BASIC,
            'verbose': True
        }
        results = train_molecular_autoencoder(test_config)
        print(f"\nFinal val loss: {results['training_history']['val_losses'][-1]:.4f}")
    else:
        print(f"\nTraining data not found at {TRAIN_DATA_PATH}.")
        print("Demo: Instantiating BasicAutoencoder with demo vocabulary...")
        model = BasicAutoencoder(
            vocab_size=vocab.vocab_size,
            latent_dim=16,
            max_length=50,
            embed_dim=32,
            hidden_dims=[128, 64]
        )
        print(f"Model: {model.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

        print("\nDemo: Instantiating VAECYC with demo vocabulary...")
        vaecyc = VAECYC(
            vocab=vocab,
            latent_dim=16,
            max_length=50,
            embed_dim=32,
            hidden_dim=128,
            encoder_num_layers=1,
            decoder_num_layers=2,
            max_epochs=5,
            dataset_size=1000,
            batch_size=32
        )
        total_params = sum(p.numel() for p in vaecyc.parameters())
        print(f"VAECYC parameters: {total_params:,}")

    print(f"\nChapter 10 complete. Outputs saved to artifacts/{CHAPTER}/ and figures/{CHAPTER}/")
