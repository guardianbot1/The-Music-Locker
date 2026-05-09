"""
transformer.py
--------------
Decoder-only (GPT-style) Transformer with genre conditioning.

Architecture
------------
  ht = TokenEmb(xt) + PositionalEmb(t) + GenreEmb(genre)
  ht passes through N causal self-attention blocks.
  Final linear head projects to vocabulary logits.

This matches the project specification:
  p(X) = Π_{t=1}^{T} p(x_t | x_{<t})
  h_t  = Emb(x_t) + Emb(genre)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── project imports ──────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    """

    def __init__(self, d_model: int, max_len: int = Config.MAX_POSITION, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                          # [max_len, d]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )                                                            # [d/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                                         # [1, max_len, d]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Single Transformer Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One causal self-attention block.

    Sub-layers
    ----------
    1. Multi-head causal self-attention  (masked so t can only attend to t' < t)
    2. Position-wise Feed-Forward Network
    Both wrapped with Pre-LayerNorm and residual connections.
    """

    def __init__(self,
                 d_model:  int   = Config.D_MODEL,
                 n_heads:  int   = Config.N_HEADS,
                 d_ff:     int   = Config.D_FF,
                 dropout:  float = Config.DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Fused QKV projection
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,   # input shape: [B, T, D]
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                : [B, T, D]
        attn_mask        : [T, T]  causal mask (additive, -inf for future positions)
        key_padding_mask : [B, T]  True where position is PAD

        Returns
        -------
        x : [B, T, D]
        """
        # 1. Causal self-attention with Pre-LN
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + x

        # 2. Feed-forward with Pre-LN
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full Transformer Music Generator
# ─────────────────────────────────────────────────────────────────────────────

class MusicTransformer(nn.Module):
    """
    Decoder-only Transformer for autoregressive music generation.

    Input embeddings
    ----------------
    h_t = TokenEmb(x_t) + PositionalEnc(t) + GenreEmb(genre)

    The genre embedding is broadcast over the whole sequence (T positions),
    giving the model a persistent genre-conditioning signal at every step.

    Forward pass output
    -------------------
    logits : [B, T, vocab_size]
    """

    def __init__(self,
                 vocab_size:  int   = Config.VOCAB_SIZE,
                 d_model:     int   = Config.D_MODEL,
                 n_heads:     int   = Config.N_HEADS,
                 n_layers:    int   = Config.N_LAYERS,
                 d_ff:        int   = Config.D_FF,
                 dropout:     float = Config.DROPOUT,
                 max_len:     int   = Config.MAX_POSITION,
                 num_genres:  int   = Config.NUM_GENRES,
                 pad_token_id: int  = 0):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.d_model      = d_model
        self.vocab_size   = vocab_size

        # ── Embedding layers ──────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.genre_emb = nn.Embedding(num_genres, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Output projection ─────────────────────────────────────────────
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (token embedding ↔ LM head, à la Press & Wolf 2017)
        self.lm_head.weight = self.token_emb.weight

        # Initialise weights
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Causal mask ───────────────────────────────────────────────────────

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular additive mask (−∞ for future positions).

        Shape: [seq_len, seq_len]
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask   # [T, T]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self,
                input_ids:  torch.Tensor,
                genre_ids:  torch.Tensor,
                labels:     Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        input_ids  : [B, T]   token sequences (shifted right for teacher-forcing)
        genre_ids  : [B]      genre index per sample
        labels     : [B, T]   target tokens (optional; if given, loss is returned)

        Returns
        -------
        logits : [B, T, vocab_size]
        loss   : scalar cross-entropy (or None if labels not supplied)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ── Build causal mask ─────────────────────────────────────────────
        causal_mask = self.make_causal_mask(T, device)          # [T, T]

        # ── Padding mask (True = ignore this position) ────────────────────
        pad_mask = input_ids.eq(self.pad_token_id)              # [B, T]

        # ── Embeddings ────────────────────────────────────────────────────
        tok_emb   = self.token_emb(input_ids)                   # [B, T, D]
        genre_vec = self.genre_emb(genre_ids).unsqueeze(1)      # [B, 1, D]

        # h_t = Emb(x_t) + Emb(genre)  (broadcast over T)
        x = tok_emb + genre_vec                                  # [B, T, D]
        x = self.pos_enc(x)                                      # [B, T, D]

        # ── Transformer blocks ────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=pad_mask)

        # ── LM head ───────────────────────────────────────────────────────
        x      = self.norm_out(x)                                # [B, T, D]
        logits = self.lm_head(x)                                 # [B, T, V]

        # ── Loss (optional) ───────────────────────────────────────────────
        loss = None
        if labels is not None:
            # Flatten for cross-entropy, ignore PAD positions in labels
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),             # [B*T, V]
                labels.reshape(-1),                              # [B*T]
                ignore_index=self.pad_token_id,
            )

        return logits, loss

    # ── Parameter count utility ───────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MusicTransformer(vocab_size=512).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    B, T = 4, Config.MAX_SEQ_LEN - 1
    ids    = torch.randint(0, 512, (B, T), device=device)
    genres = torch.randint(0, Config.NUM_GENRES, (B,), device=device)
    labels = torch.randint(0, 512, (B, T), device=device)

    logits, loss = model(ids, genres, labels)
    print(f"Logits shape : {logits.shape}")   # [4, 511, 512]
    print(f"Loss         : {loss.item():.4f}")
