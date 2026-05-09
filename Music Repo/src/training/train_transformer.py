"""
train_transformer.py
---------------------
Full training loop for the Transformer music generator (Task 3).

Features
--------
* Train / Val / Test split  (indices are saved to disk for reproducibility)
* Cosine LR schedule with linear warm-up
* Gradient clipping
* Epoch-level perplexity tracking (train + val)
* Checkpoint saving (best val loss + latest)
* Loss curve PNG saved to outputs/plots/

Usage
-----
    python src/training/train_transformer.py
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, random_split

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from models.transformer import MusicTransformer
from preprocessing.tokenizer import load_dataset, build_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset split
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(dataset, val_frac: float = Config.VAL_SPLIT,
                test_frac: float = Config.TEST_SPLIT,
                seed: int = Config.SEED
                ) -> Tuple[Subset, Subset, Subset]:
    """
    Split *dataset* into train / val / test subsets.
    Split indices are saved to SPLIT_DIR for reproducibility.
    """
    n      = len(dataset)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_tr   = n - n_val - n_test

    gen  = torch.Generator().manual_seed(seed)
    tr, val, te = random_split(dataset, [n_tr, n_val, n_test], generator=gen)

    # Save indices
    Config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train", tr), ("val", val), ("test", te)]:
        idxs = list(subset.indices)
        with open(Config.SPLIT_DIR / f"{name}_indices.json", "w") as f:
            json.dump(idxs, f)

    log.info("Split sizes  →  train: %d  |  val: %d  |  test: %d", n_tr, n_val, n_test)
    return tr, val, te


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warm-up → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Cosine schedule with linear warm-up."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)                # linear warm-up
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))  # cosine decay

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model: nn.Module,
              loader: DataLoader,
              optimizer,
              scheduler,
              device: torch.device,
              training: bool = True,
              grad_clip: float = Config.GRAD_CLIP) -> Tuple[float, float]:
    """
    Run one full pass over loader.

    Returns
    -------
    avg_loss   : mean cross-entropy over all batches
    perplexity : exp(avg_loss)
    """
    model.train() if training else model.eval()

    total_loss = 0.0
    total_tokens = 0
    context = torch.enable_grad() if training else torch.no_grad()

    mode_name = "TRAIN" if training else "VAL"
    total_batches = len(loader)
    start_time = time.time()

    log.info("%s epoch started | total batches: %d", mode_name, total_batches)

    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            genre_ids = batch["genre_id"].to(device).long()

            _, loss = model(input_ids, genre_ids, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

            n_tokens = labels.ne(model.pad_token_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            # Print progress every 100 batches, and also on first/last batch
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                avg_loss_so_far = total_loss / max(total_tokens, 1)
                ppl_so_far = float(np.exp(min(avg_loss_so_far, 20)))

                batches_per_sec = batch_idx / max(elapsed, 1e-6)
                remaining_batches = total_batches - batch_idx
                eta_sec = remaining_batches / max(batches_per_sec, 1e-6)

                log.info(
                    "%s batch %d/%d | loss: %.4f | ppl: %.2f | elapsed: %.0fs | ETA: %.0fs",
                    mode_name,
                    batch_idx,
                    total_batches,
                    avg_loss_so_far,
                    ppl_so_far,
                    elapsed,
                    eta_sec,
                )

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = float(np.exp(min(avg_loss, 20)))

    log.info("%s epoch finished | avg loss: %.4f | ppl: %.2f", mode_name, avg_loss, perplexity)

    return avg_loss, perplexity


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history: Dict[str, List], out_dir: Path = Config.PLOTS_DIR):
    """Save loss and perplexity curves as PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Task 3 – Transformer Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "transformer_loss_curve.png", dpi=150)
    plt.close(fig)

    # Perplexity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_ppl"], label="Train Perplexity", marker="o")
    ax.plot(epochs, history["val_ppl"],   label="Val Perplexity",   marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Task 3 – Transformer Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "transformer_perplexity_curve.png", dpi=150)
    plt.close(fig)

    log.info("Plots saved → %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, val_loss: float,
                    path: Path, history: Dict):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
        "history":    history,
        "vocab_size": model.vocab_size,
    }, path)
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(model, optimizer, path: Path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    log.info("Checkpoint loaded from %s  (epoch %d)", path, ckpt["epoch"])
    return ckpt["epoch"], ckpt["val_loss"], ckpt.get("history", {})


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(resume: bool = False):
    """
    Full training loop.

    Parameters
    ----------
    resume : if True, load the latest checkpoint before training.
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on: %s", device)

    if torch.cuda.is_available():
        cuda_index = torch.cuda.current_device()
        cuda_name = torch.cuda.get_device_name(cuda_index)
        log.info("CUDA device index: %d", cuda_index)
        log.info("CUDA device name : %s", cuda_name)
    else:
        log.info("CUDA not available, running on CPU")


    # ── Dataset ──────────────────────────────────────────────────────────
    log.info("Loading dataset …")
    dataset = load_dataset()
    log.info("Dataset size: %d windows", len(dataset))

    tr_data, val_data, te_data = make_splits(dataset)

    tr_loader = DataLoader(tr_data,  batch_size=Config.BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ── Tokeniser (for vocab size) ────────────────────────────────────────
    tok = build_tokenizer()
    vocab_size = len(tok)
    pad_id     = tok["PAD_None"]
    log.info("Vocab size: %d  |  PAD id: %d", vocab_size, pad_id)

    # ── Model ─────────────────────────────────────────────────────────────
    model = MusicTransformer(
        vocab_size   = vocab_size,
        d_model      = Config.D_MODEL,
        n_heads      = Config.N_HEADS,
        n_layers     = Config.N_LAYERS,
        d_ff         = Config.D_FF,
        dropout      = Config.DROPOUT,
        max_len      = Config.MAX_POSITION,
        num_genres   = Config.NUM_GENRES,
        pad_token_id = pad_id,
    ).to(device)

    log.info("Model parameters: %s", f"{model.num_parameters():,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(),
                                     lr=Config.LEARNING_RATE, weight_decay=0.01)
    total_steps  = len(tr_loader) * Config.EPOCHS
    scheduler    = get_lr_scheduler(optimizer, Config.WARMUP_STEPS, total_steps)

    # ── Checkpoint directory ──────────────────────────────────────────────
    Config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt   = Config.CHECKPOINTS_DIR / "transformer_best.pt"
    latest_ckpt = Config.CHECKPOINTS_DIR / "transformer_latest.pt"

    start_epoch = 0
    best_val    = float("inf")
    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_ppl":  [], "val_ppl":  [],
    }

    if resume and latest_ckpt.exists():
        start_epoch, best_val, history = load_checkpoint(model, optimizer, latest_ckpt)

    # ── Training loop ─────────────────────────────────────────────────────
    log.info("Starting training for %d epochs …", Config.EPOCHS)

    for epoch in range(start_epoch + 1, Config.EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_ppl   = run_epoch(model, tr_loader,  optimizer, scheduler, device, training=True)
        val_loss, val_ppl = run_epoch(model, val_loader, optimizer, scheduler, device, training=False)

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d/%d | Train loss: %.4f  ppl: %7.2f | "
            "Val loss: %.4f  ppl: %7.2f | LR: %.2e | %.0fs",
            epoch, Config.EPOCHS,
            tr_loss, tr_ppl,
            val_loss, val_ppl,
            scheduler.get_last_lr()[0],
            elapsed,
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(tr_ppl)
        history["val_ppl"].append(val_ppl)

        # Save checkpoints
        save_checkpoint(model, optimizer, epoch, val_loss, latest_ckpt, history)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt, history)
            log.info("  ↑ New best val loss: %.4f", best_val)

    # ── Final test evaluation ─────────────────────────────────────────────
    te_loader = DataLoader(te_data, batch_size=Config.BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)
    ckpt_best = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt_best["model"])

    te_loss, te_ppl = run_epoch(model, te_loader, optimizer, scheduler, device, training=False)
    log.info("─" * 60)
    log.info("Test  loss: %.4f  |  Test perplexity: %.2f", te_loss, te_ppl)
    log.info("─" * 60)

    # Save test results
    results = {
        "test_loss":       te_loss,
        "test_perplexity": te_ppl,
        "best_val_loss":   best_val,
        "total_epochs":    Config.EPOCHS,
    }
    with open(Config.OUTPUTS_DIR / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Test results saved → %s", Config.OUTPUTS_DIR / "test_results.json")

    # ── Plot curves ───────────────────────────────────────────────────────
    plot_curves(history)

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
