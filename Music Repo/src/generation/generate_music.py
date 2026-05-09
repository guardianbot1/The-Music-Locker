"""
generate_music.py
-----------------
Autoregressive music generation using the trained Transformer.

Sampling strategy
-----------------
  1. Prime the model with a BOS token (and optional seed tokens).
  2. At each step compute logits, apply temperature scaling + top-k filtering.
  3. Sample the next token from the resulting distribution.
  4. Stop when EOS is produced or GEN_MAX_LEN tokens are reached.
  5. Convert the token sequence back to a MIDI file via the miditok tokeniser.

Produces 10 compositions (2 per genre × 5 genres) as required by the spec.

Usage
-----
    python generation/generate_music.py \
        --checkpoint outputs/checkpoints/transformer_best.pt \
        --n_per_genre 2 \
        --temperature 0.9 \
        --top_k 50
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import build_tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out all logits except the top-k.

    Parameters
    ----------
    logits : [V]  raw logits for one time-step
    k      : number of candidates to keep (k=0 means keep all)

    Returns
    -------
    filtered logits : [V]
    """
    if k == 0:
        return logits
    # kth largest value
    top_values, _ = torch.topk(logits, k)
    threshold      = top_values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def sample_next_token(logits: torch.Tensor,
                      temperature: float = 1.0,
                      top_k: int = 50) -> int:
    """
    Sample one token from the logit distribution.

    Parameters
    ----------
    logits      : [V]  raw (un-softmaxed) logits
    temperature : > 1 = more random, < 1 = more deterministic
    top_k       : restrict sampling to top-k tokens (0 = no restriction)

    Returns
    -------
    token_id : int
    """
    logits = logits / max(temperature, 1e-8)
    logits = top_k_filter(logits, top_k)
    probs  = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ─────────────────────────────────────────────────────────────────────────────
# Core generation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_sequence(model: MusicTransformer,
                      genre_id: int,
                      bos_id: int,
                      eos_id: int,
                      device: torch.device,
                      seed_ids: Optional[List[int]] = None,
                      max_len: int = Config.GEN_MAX_LEN,
                      temperature: float = Config.TEMPERATURE,
                      top_k: int = Config.TOP_K) -> List[int]:
    """
    Autoregressively generate a token sequence for a given genre.

    p(x_t | x_{<t})  repeated up to *max_len* steps.

    Parameters
    ----------
    model      : trained MusicTransformer (in eval mode)
    genre_id   : integer genre label
    bos_id     : BOS token id
    eos_id     : EOS token id
    device     : torch device
    seed_ids   : optional list of tokens to prime generation
    max_len    : maximum number of tokens to generate (incl. BOS)
    temperature: sampling temperature
    top_k      : top-k filtering parameter

    Returns
    -------
    generated token IDs (excluding BOS, truncated at first EOS)
    """
    model.eval()

    # Initialise sequence with BOS (+ optional seed)
    ids = [bos_id]
    if seed_ids:
        ids.extend(seed_ids)

    genre_tensor = torch.tensor([genre_id], dtype=torch.long, device=device)  # [1]

    for _ in range(max_len - len(ids)):
        # Truncate to max_position to avoid OOM
        context     = ids[-Config.MAX_POSITION:]
        input_ids   = torch.tensor([context], dtype=torch.long, device=device)  # [1, T]
        genre_broad = genre_tensor  # scalar genre per sample

        logits, _ = model(input_ids, genre_broad)  # logits: [1, T, V]
        next_logits = logits[0, -1]                              # [V]

        next_id = sample_next_token(next_logits, temperature, top_k)
        ids.append(next_id)

        if next_id == eos_id:
            break

    # Strip BOS
    return ids[1:]


# ─────────────────────────────────────────────────────────────────────────────
# Token sequence → MIDI
# ─────────────────────────────────────────────────────────────────────────────

def tokens_to_midi(token_ids: List[int],
                   tok,
                   out_path: Path) -> bool:
    """
    Convert a list of token IDs back to a MIDI file using miditok.

    Parameters
    ----------
    token_ids : generated token ID list
    tok       : miditok REMI tokeniser
    out_path  : where to write the .mid file

    Returns
    -------
    True on success, False on error.
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # miditok ≥ 2.1: decode takes a TokSequence or list of ids
        midi = tok.decode(token_ids)  
        midi.dump_midi(str(out_path))
        log.debug("MIDI saved → %s", out_path)
        return True
    except Exception as exc:
        log.warning("Failed to decode tokens to MIDI: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main generation entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(checkpoint_path: Path,
                 n_per_genre: int = Config.NUM_SAMPLES_PER_GENRE,
                 temperature: float = Config.TEMPERATURE,
                 top_k: int = Config.TOP_K,
                 out_dir: Path = Config.GENERATED_MIDIS_DIR) -> List[Path]:
    """
    Generate *n_per_genre* MIDI files for each genre (= 10 total by default).

    Parameters
    ----------
    checkpoint_path : path to the best checkpoint .pt file
    n_per_genre     : compositions per genre
    temperature     : sampling temperature
    top_k           : top-k filter size
    out_dir         : directory to write .mid files

    Returns
    -------
    List of generated MIDI file paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Generating on: %s", device)

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tok    = build_tokenizer()
    bos_id = tok["BOS_None"]
    eos_id = tok["EOS_None"]
    pad_id = tok["PAD_None"]

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt       = torch.load(checkpoint_path, map_location=device)
    vocab_size = ckpt.get("vocab_size", len(tok))

    model = MusicTransformer(
        vocab_size   = vocab_size,
        d_model      = Config.D_MODEL,
        n_heads      = Config.N_HEADS,
        n_layers     = Config.N_LAYERS,
        d_ff         = Config.D_FF,
        dropout      = 0.0,
        num_genres   = Config.NUM_GENRES,
        pad_token_id = pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Checkpoint loaded (epoch %d, val loss %.4f)",
             ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    metadata: List[Dict] = []

    # ── Generate ──────────────────────────────────────────────────────────
    for genre_id, genre_name in Config.ID2GENRE.items():
        log.info("Generating %d compositions for genre: %s", n_per_genre, genre_name)
        for sample_idx in range(n_per_genre):
            log.info("  [%s] sample %d/%d …", genre_name, sample_idx + 1, n_per_genre)

            ids = generate_sequence(
                model        = model,
                genre_id     = genre_id,
                bos_id       = bos_id,
                eos_id       = eos_id,
                device       = device,
                max_len      = Config.GEN_MAX_LEN,
                temperature  = temperature,
                top_k        = top_k,
            )

            fname    = f"task3_{genre_name}_{sample_idx + 1:02d}.mid"
            out_path = out_dir / fname

            success = tokens_to_midi(ids, tok, out_path)
            if success:
                generated.append(out_path)
                meta = {
                    "file":        fname,
                    "genre":       genre_name,
                    "genre_id":    genre_id,
                    "sample_idx":  sample_idx,
                    "n_tokens":    len(ids),
                    "temperature": temperature,
                    "top_k":       top_k,
                }
                metadata.append(meta)
                log.info("  ✓ Saved %s  (%d tokens)", fname, len(ids))
            else:
                log.warning("  ✗ Failed to write %s", fname)

    # ── Save metadata ─────────────────────────────────────────────────────
    meta_path = out_dir / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved → %s", meta_path)

    log.info("─" * 50)
    log.info("Generation complete.  %d / %d files written.",
             len(generated), n_per_genre * Config.NUM_GENRES)
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate a single MIDI for quick testing
# ─────────────────────────────────────────────────────────────────────────────

def quick_generate(checkpoint_path: Path,
                   genre: str = "classical",
                   out_path: Optional[Path] = None,
                   temperature: float = 0.9,
                   top_k: int = 50) -> Optional[Path]:
    """
    Generate one MIDI for a given genre. Useful for rapid iteration.

    Parameters
    ----------
    checkpoint_path : .pt checkpoint path
    genre           : genre string (must be in Config.GENRES)
    out_path        : output file (defaults to outputs/generated_midis/quick_test.mid)
    """
    if genre not in Config.GENRE2ID:
        raise ValueError(f"Unknown genre '{genre}'. Choose from: {Config.GENRES}")

    if out_path is None:
        out_path = Config.GENERATED_MIDIS_DIR / f"quick_{genre}.mid"

    results = generate_all(
        checkpoint_path = checkpoint_path,
        n_per_genre     = 1,
        temperature     = temperature,
        top_k           = top_k,
        out_dir         = out_path.parent,
    )
    return results[0] if results else None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate music with trained Transformer")
    parser.add_argument("--checkpoint",  type=str,
                        default=str(Config.CHECKPOINTS_DIR / "transformer_best.pt"))
    parser.add_argument("--n_per_genre", type=int,  default=Config.NUM_SAMPLES_PER_GENRE)
    parser.add_argument("--temperature", type=float, default=Config.TEMPERATURE)
    parser.add_argument("--top_k",       type=int,  default=Config.TOP_K)
    parser.add_argument("--out_dir",     type=str,  default=str(Config.GENERATED_MIDIS_DIR))
    args = parser.parse_args()

    generate_all(
        checkpoint_path = Path(args.checkpoint),
        n_per_genre     = args.n_per_genre,
        temperature     = args.temperature,
        top_k           = args.top_k,
        out_dir         = Path(args.out_dir),
    )
