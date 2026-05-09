"""
metrics.py
----------
All quantitative evaluation metrics for Task 3.

Implements every metric specified in the project rubric:

  1.  Perplexity      — from the autoregressive NLL loss on the test set
  2.  Pitch Histogram Similarity  H(p, q) = Σ |p_i − q_i|
  3.  Rhythm Diversity Score      D = #unique_durations / #total_notes
  4.  Repetition Ratio            R = #repeated_patterns / #total_patterns

Usage (standalone)
------------------
    python evaluation/metrics.py \
        --checkpoint outputs/checkpoints/transformer_best.pt \
        --generated  outputs/generated_midis/
"""

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
import torch
from torch.utils.data import DataLoader, Subset

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import build_tokenizer, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model Perplexity on the test set
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_perplexity(model: MusicTransformer,
                             test_loader: DataLoader,
                             device: torch.device) -> Dict[str, float]:
    """
    Compute token-level average NLL and perplexity on the held-out test set.

    Perplexity = exp( (1/T) * L_TR )
    where L_TR = − Σ log p_θ(x_t | x_{<t})

    Returns
    -------
    dict with keys: avg_nll, perplexity, n_tokens
    """
    model.eval()
    total_nll    = 0.0
    total_tokens = 0

    pad_id = model.pad_token_id

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            genre_ids = torch.tensor(batch["genre_id"], dtype=torch.long, device=device)

            _, loss = model(input_ids, genre_ids, labels)

            # loss is already mean; re-scale to sum for correct averaging
            n_tok         = labels.ne(pad_id).sum().item()
            total_nll    += loss.item() * n_tok
            total_tokens += n_tok

    avg_nll    = total_nll / max(total_tokens, 1)
    perplexity = math.exp(min(avg_nll, 20))

    result = {
        "avg_nll":    round(avg_nll,    4),
        "perplexity": round(perplexity, 4),
        "n_tokens":   total_tokens,
    }
    log.info("Test NLL: %.4f  |  Perplexity: %.2f  (over %d tokens)",
             avg_nll, perplexity, total_tokens)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pitch Histogram Similarity
# ─────────────────────────────────────────────────────────────────────────────

def pitch_histogram(midi_path: Path) -> np.ndarray:
    """
    Compute the 12-bin pitch-class histogram (normalised) for a MIDI file.

    Returns
    -------
    hist : [12]  float array, sums to 1.
    """
    pm   = pretty_midi.PrettyMIDI(str(midi_path))
    hist = np.zeros(12, dtype=float)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            hist[note.pitch % 12] += 1

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def pitch_histogram_similarity(ref_midi: Path,
                               gen_midi: Path) -> float:
    """
    H(p, q) = Σ_{i=1}^{12} |p_i − q_i|

    Lower is more similar (0 = identical distributions, 2 = completely different).

    Returns
    -------
    similarity_score : float in [0, 2]
    """
    p = pitch_histogram(ref_midi)
    q = pitch_histogram(gen_midi)
    return float(np.sum(np.abs(p - q)))


def batch_pitch_histogram_similarity(ref_dir: Path,
                                     gen_dir: Path) -> Dict[str, float]:
    """
    Average pitch histogram similarity between two folders of MIDI files.

    Pairwise matching is done by sorted filename order.
    """
    refs = sorted(ref_dir.glob("*.mid"))
    gens = sorted(gen_dir.glob("*.mid"))
    n    = min(len(refs), len(gens))

    if n == 0:
        log.warning("No MIDI files found for pitch histogram comparison.")
        return {"mean": None, "std": None}

    scores = [pitch_histogram_similarity(refs[i], gens[i]) for i in range(n)]
    result = {
        "mean":   round(float(np.mean(scores)), 4),
        "std":    round(float(np.std(scores)), 4),
        "values": [round(s, 4) for s in scores],
    }
    log.info("Pitch Histogram Similarity — mean: %.4f  std: %.4f", result["mean"], result["std"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Rhythm Diversity Score
# ─────────────────────────────────────────────────────────────────────────────

def rhythm_diversity(midi_path: Path, duration_bins: int = 16) -> float:
    """
    D_rhythm = #unique_durations / #total_notes

    Durations are quantised to *duration_bins* equal-width bins between
    [0, max_duration] to avoid floating-point uniqueness issues.

    Returns
    -------
    score : float in (0, 1].  Higher = more rhythmically diverse.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    all_durations: List[float] = []

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            dur = note.end - note.start
            if dur > 0:
                all_durations.append(dur)

    if not all_durations:
        return 0.0

    max_dur = max(all_durations)
    bin_w   = max_dur / duration_bins if max_dur > 0 else 1.0
    binned  = [int(d / bin_w) for d in all_durations]

    unique = len(set(binned))
    total  = len(binned)
    return round(unique / total, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Repetition Ratio
# ─────────────────────────────────────────────────────────────────────────────

def repetition_ratio(midi_path: Path, pattern_len: int = 4) -> float:
    """
    R = #repeated_patterns / #total_patterns

    A pattern is a tuple of *pattern_len* consecutive pitch-class values.
    A pattern is "repeated" if it appears more than once in the sequence.

    Returns
    -------
    score : float in [0, 1].  Lower = more creative (less repetition).
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    pitches: List[int] = []

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in sorted(inst.notes, key=lambda n: n.start):
            pitches.append(note.pitch % 12)

    if len(pitches) < pattern_len + 1:
        return 0.0

    patterns = [
        tuple(pitches[i: i + pattern_len])
        for i in range(len(pitches) - pattern_len + 1)
    ]
    counts  = Counter(patterns)
    total   = len(patterns)
    repeated = sum(c for c in counts.values() if c > 1)
    return round(repeated / total, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def midi_metrics(midi_path: Path) -> Dict:
    """Compute all MIDI-level metrics for a single generated file."""
    return {
        "file":             midi_path.name,
        "rhythm_diversity": rhythm_diversity(midi_path),
        "repetition_ratio": repetition_ratio(midi_path),
    }


def generate_perplexity_report(checkpoint_path: Path,
                               out_path: Optional[Path] = None) -> Dict:
    """
    Load a saved checkpoint, evaluate on the test split, and write a JSON report.

    Parameters
    ----------
    checkpoint_path : path to .pt checkpoint
    out_path        : where to save the report JSON (default: outputs/)

    Returns
    -------
    report dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load tokeniser ────────────────────────────────────────────────────
    tok    = build_tokenizer()
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
        dropout      = 0.0,           # eval mode
        num_genres   = Config.NUM_GENRES,
        pad_token_id = pad_id,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Test DataLoader ───────────────────────────────────────────────────
    dataset   = load_dataset()
    split_f   = Config.SPLIT_DIR / "test_indices.json"
    with open(split_f) as f:
        test_idx = json.load(f)
    test_data   = Subset(dataset, test_idx)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=2)

    # ── Perplexity ────────────────────────────────────────────────────────
    ppl_results = compute_model_perplexity(model, test_loader, device)

    # ── Per-genre perplexity ──────────────────────────────────────────────
    genre_results: Dict[str, Dict] = {}
    for genre_id, genre_name in Config.ID2GENRE.items():
        genre_idx = [i for i in test_idx
                     if dataset.genres[i].item() == genre_id]
        if not genre_idx:
            continue
        g_data   = Subset(dataset, genre_idx)
        g_loader = DataLoader(g_data, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=2)
        g_res = compute_model_perplexity(model, g_loader, device)
        genre_results[genre_name] = g_res
        log.info("  %-12s perplexity: %.2f", genre_name, g_res["perplexity"])

    # ── MIDI-level metrics on generated files ─────────────────────────────
    gen_dir   = Config.GENERATED_MIDIS_DIR
    midi_rows = []
    if gen_dir.exists():
        for mpath in sorted(gen_dir.glob("*.mid")):
            midi_rows.append(midi_metrics(mpath))

    # ── Assemble report ───────────────────────────────────────────────────
    report = {
        "checkpoint":        str(checkpoint_path),
        "overall":           ppl_results,
        "per_genre":         genre_results,
        "generated_metrics": midi_rows,
        "training_history":  ckpt.get("history", {}),
    }

    if out_path is None:
        out_path = Config.OUTPUTS_DIR / "perplexity_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Perplexity report saved → %s", out_path)

    # ── Pretty print summary ──────────────────────────────────────────────
    _print_report(report)
    return report


def _print_report(report: Dict):
    """Pretty-print the evaluation report to stdout."""
    print("\n" + "═" * 60)
    print("  TASK 3 — TRANSFORMER EVALUATION REPORT")
    print("═" * 60)
    o = report["overall"]
    print(f"  Overall Test NLL    : {o['avg_nll']}")
    print(f"  Overall Perplexity  : {o['perplexity']}")
    print(f"  Evaluated on tokens : {o['n_tokens']:,}")
    print()
    print("  Per-genre Perplexity")
    print("  " + "─" * 30)
    for genre, gr in report["per_genre"].items():
        print(f"  {genre:<14}  {gr['perplexity']:>8.2f}")
    if report["generated_metrics"]:
        print()
        print("  Generated MIDI Metrics")
        print("  " + "─" * 50)
        print(f"  {'File':<30} {'Rhythm Div.':>12} {'Repetition':>12}")
        for row in report["generated_metrics"]:
            print(f"  {row['file']:<30} {row['rhythm_diversity']:>12.4f}"
                  f" {row['repetition_ratio']:>12.4f}")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Transformer music generator")
    parser.add_argument("--checkpoint", type=str,
                        default=str(Config.CHECKPOINTS_DIR / "transformer_best.pt"))
    parser.add_argument("--out", type=str, default=None,
                        help="Path for output JSON report")
    args = parser.parse_args()

    out = Path(args.out) if args.out else None
    generate_perplexity_report(Path(args.checkpoint), out_path=out)
