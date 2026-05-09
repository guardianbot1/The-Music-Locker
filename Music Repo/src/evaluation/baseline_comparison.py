"""
baseline_comparison.py
----------------------
Task 3 — Baseline Comparison

Implements two baseline music generators and compares them against the
trained Transformer using the same metrics as metrics.py.

Baselines
---------
  1. Random Note Generator  — uniformly samples pitches, velocities, durations
  2. Markov Chain           — 2nd-order pitch Markov chain trained on your MIDI data

Output
------
  outputs/baselines/random/          ← 10 generated MIDI files
  outputs/baselines/markov/          ← 10 generated MIDI files
  outputs/baseline_comparison.json   ← full metric table
  (also prints a comparison table to the terminal)

Usage
-----
  # Full run (generate + evaluate all three models)
  python evaluation/baseline_comparison.py \
      --checkpoint outputs/checkpoints/transformer_best.pt

  # Skip Transformer evaluation (if you haven't trained yet)
  python evaluation/baseline_comparison.py --no_transformer
"""

import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import Config
from src.evaluation.metrics_trans import rhythm_diversity, repetition_ratio, pitch_histogram

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

# Fixed musical constants for both baselines
PITCHES        = list(range(48, 85))   # comfortable piano range C3–C6
VELOCITIES     = [40, 55, 70, 85, 100] # ppp → ff
DURATIONS      = [0.125, 0.25, 0.5, 1.0, 2.0]  # 32nd → half note @ 120 bpm
NOTES_PER_FILE = 120                   # ~30 bars of 4/4 at 120 bpm

def _write_midi(notes: List[Tuple[int, float, float, int]],
                out_path: Path,
                tempo: int = 120) -> bool:
    """
    Write a list of (pitch, start, end, velocity) tuples to a MIDI file.

    Parameters
    ----------
    notes    : [(pitch, start_sec, end_sec, velocity), ...]
    out_path : destination .mid path
    tempo    : BPM

    Returns
    -------
    True on success.
    """
    try:
        pm   = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        inst = pretty_midi.Instrument(program=0)   # Acoustic Grand Piano

        for pitch, start, end, vel in notes:
            note = pretty_midi.Note(
                velocity=int(vel),
                pitch=int(pitch),
                start=float(start),
                end=float(end),
            )
            inst.notes.append(note)

        pm.instruments.append(inst)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(out_path))
        return True
    except Exception as exc:
        log.warning("MIDI write failed for %s: %s", out_path, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Random Note Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_random_midi(out_path: Path,
                         n_notes: int = NOTES_PER_FILE,
                         seed: int = 0) -> bool:
    """
    Generate a MIDI file by uniformly sampling pitch, duration, and velocity.

    This is the naive baseline — no musical knowledge whatsoever.
    """
    rng = random.Random(seed)
    notes = []
    t = 0.0

    for _ in range(n_notes):
        pitch    = rng.choice(PITCHES)
        duration = rng.choice(DURATIONS)
        velocity = rng.choice(VELOCITIES)
        notes.append((pitch, t, t + duration, velocity))
        t += duration

    return _write_midi(notes, out_path)


def run_random_baseline(n_files: int = 10,
                        out_dir: Path = None) -> List[Path]:
    """Generate *n_files* random MIDI files."""
    if out_dir is None:
        out_dir = Config.OUTPUTS_DIR / "baselines" / "random"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for i in range(n_files):
        out_path = out_dir / f"random_{i + 1:02d}.mid"
        ok = generate_random_midi(out_path, seed=i * 7)
        if ok:
            generated.append(out_path)
            log.info("Random [%d/%d] → %s", i + 1, n_files, out_path.name)
        else:
            log.warning("Random [%d/%d] failed", i + 1, n_files)

    log.info("Random baseline: %d / %d files written.", len(generated), n_files)
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Markov Chain Music Model
# ─────────────────────────────────────────────────────────────────────────────

class MarkovMusicModel:
    """
    2nd-order pitch Markov chain.

    State  : (prev_pitch, curr_pitch)
    Predict: next_pitch

    Also models duration using a 1st-order duration Markov chain:
    State  : quantised_prev_duration
    Predict: quantised_next_duration
    """

    def __init__(self, order: int = 2):
        self.order        = order
        # pitch transitions: {(p1, p2): Counter({p3: count})}
        self.pitch_trans  = defaultdict(Counter)
        # duration transitions: {d1_bin: Counter({d2_bin: count})}
        self.dur_trans    = defaultdict(Counter)
        self._dur_bins    = DURATIONS
        self._trained     = False

    # ── Training ─────────────────────────────────────────────────────────

    def _quantise_dur(self, dur: float) -> float:
        """Snap a duration to the nearest bin."""
        return min(self._dur_bins, key=lambda b: abs(b - dur))

    def train_on_midi(self, midi_path: Path) -> int:
        """
        Accumulate transition counts from one MIDI file.

        Returns the number of notes processed.
        """
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception:
            return 0

        pitches   = []
        durations = []

        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in sorted(inst.notes, key=lambda n: n.start):
                pitches.append(note.pitch)
                dur = self._quantise_dur(note.end - note.start)
                durations.append(dur)

        # Pitch transitions (order-2)
        for i in range(len(pitches) - self.order):
            state = tuple(pitches[i: i + self.order])
            nxt   = pitches[i + self.order]
            self.pitch_trans[state][nxt] += 1

        # Duration transitions (order-1)
        for i in range(len(durations) - 1):
            self.dur_trans[durations[i]][durations[i + 1]] += 1

        return len(pitches)

    def train(self, midi_dir: Path, max_files: int = 200) -> int:
        """Train on all MIDI files found under *midi_dir*."""
        midi_paths = list(midi_dir.rglob("*.mid"))[:max_files]
        if not midi_paths:
            log.warning("No MIDI files found in %s", midi_dir)
            return 0

        total = 0
        for i, mp in enumerate(midi_paths):
            n = self.train_on_midi(mp)
            total += n
            if (i + 1) % 50 == 0:
                log.info("  Markov training: %d / %d files …", i + 1, len(midi_paths))

        self._trained = total > 0
        log.info("Markov chain trained on %d notes from %d files.",
                 total, len(midi_paths))
        return total

    # ── Sampling helpers ─────────────────────────────────────────────────

    def _sample_counter(self, counter: Counter, rng: random.Random) -> Optional:
        """Sample from a Counter proportionally to counts."""
        if not counter:
            return None
        items, weights = zip(*counter.items())
        total = sum(weights)
        r = rng.uniform(0, total)
        cumulative = 0
        for item, w in zip(items, weights):
            cumulative += w
            if r <= cumulative:
                return item
        return items[-1]

    def _random_pitch(self, rng: random.Random) -> int:
        return rng.choice(PITCHES)

    def _random_dur(self, rng: random.Random) -> float:
        return rng.choice(self._dur_bins)

    # ── Generation ───────────────────────────────────────────────────────

    def generate(self, n_notes: int = NOTES_PER_FILE,
                 seed: int = 0) -> List[Tuple[int, float, float, int]]:
        """
        Sample a sequence of notes from the trained Markov chain.

        Returns
        -------
        list of (pitch, start_sec, end_sec, velocity)
        """
        rng = random.Random(seed)

        # Seed the pitch history
        history = [self._random_pitch(rng) for _ in range(self.order)]
        prev_dur = self._random_dur(rng)

        notes = []
        t     = 0.0

        for _ in range(n_notes):
            # ── Pitch ──────────────────────────────────────────────────
            state    = tuple(history[-self.order:])
            counter  = self.pitch_trans.get(state, Counter())
            if counter:
                pitch = self._sample_counter(counter, rng)
            else:
                pitch = self._random_pitch(rng)   # back-off to uniform
            history.append(pitch)

            # ── Duration ───────────────────────────────────────────────
            dur_counter = self.dur_trans.get(prev_dur, Counter())
            if dur_counter:
                dur = self._sample_counter(dur_counter, rng)
            else:
                dur = self._random_dur(rng)
            prev_dur = dur

            velocity = rng.choice(VELOCITIES)
            notes.append((pitch, t, t + dur, velocity))
            t += dur

        return notes


def run_markov_baseline(midi_data_dir: Path = None,
                        n_files: int = 10,
                        out_dir: Path = None) -> List[Path]:
    """Train Markov model and generate *n_files* MIDI files."""
    if midi_data_dir is None:
        midi_data_dir = Config.RAW_MIDI_DIR
    if out_dir is None:
        out_dir = Config.OUTPUTS_DIR / "baselines" / "markov"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Training Markov chain on MIDI files in %s …", midi_data_dir)
    model = MarkovMusicModel(order=2)
    n_trained = model.train(midi_data_dir, max_files=200)

    if n_trained == 0:
        log.error("Markov model could not be trained — no MIDI files found.")
        return []

    generated = []
    for i in range(n_files):
        out_path = out_dir / f"markov_{i + 1:02d}.mid"
        notes    = model.generate(seed=i * 13)
        ok       = _write_midi(notes, out_path)
        if ok:
            generated.append(out_path)
            log.info("Markov [%d/%d] → %s", i + 1, n_files, out_path.name)
        else:
            log.warning("Markov [%d/%d] failed", i + 1, n_files)

    log.info("Markov baseline: %d / %d files written.", len(generated), n_files)
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Metric aggregation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_midi_folder(folder: Path) -> Dict:
    """
    Compute mean rhythm diversity and repetition ratio for all .mid files
    in *folder*.

    Returns
    -------
    dict with keys: rhythm_diversity_mean, rhythm_diversity_std,
                    repetition_ratio_mean, repetition_ratio_std,
                    n_files
    """
    midi_files = sorted(folder.glob("*.mid"))
    if not midi_files:
        log.warning("No MIDI files found in %s", folder)
        return {}

    rd_scores  = []
    rep_scores = []

    for mp in midi_files:
        try:
            rd_scores.append(rhythm_diversity(mp))
            rep_scores.append(repetition_ratio(mp))
        except Exception as exc:
            log.debug("Metric error for %s: %s", mp.name, exc)

    return {
        "n_files":                len(midi_files),
        "rhythm_diversity_mean":  round(float(np.mean(rd_scores)),  4) if rd_scores  else None,
        "rhythm_diversity_std":   round(float(np.std(rd_scores)),   4) if rd_scores  else None,
        "repetition_ratio_mean":  round(float(np.mean(rep_scores)), 4) if rep_scores else None,
        "repetition_ratio_std":   round(float(np.std(rep_scores)),  4) if rep_scores else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Transformer perplexity retrieval
# ─────────────────────────────────────────────────────────────────────────────

def get_transformer_perplexity(checkpoint_path: Path) -> Optional[float]:
    """
    Pull the validation perplexity stored in the checkpoint.
    Falls back to running metrics.py if not found.
    """
    try:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        val_loss = ckpt.get("val_loss")
        if val_loss is not None:
            return round(math.exp(min(float(val_loss), 20)), 4)
    except Exception as exc:
        log.warning("Could not load checkpoint: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(report: Dict):
    """Pretty-print the comparison table to stdout."""
    rows = report["models"]

    print("\n" + "═" * 78)
    print("  TASK 3 — BASELINE COMPARISON REPORT")
    print("═" * 78)
    print(f"  {'Model':<28} {'Perplexity':>12} {'Rhythm Div.':>13} {'Repetition':>12}")
    print("  " + "─" * 74)

    for row in rows:
        ppl = f"{row['perplexity']:.2f}"  if row.get("perplexity") else   "  N/A  "
        rd  = f"{row['rhythm_diversity_mean']:.4f}" if row.get("rhythm_diversity_mean") else "  N/A  "
        rep = f"{row['repetition_ratio_mean']:.4f}" if row.get("repetition_ratio_mean") else "  N/A  "
        print(f"  {row['model']:<28} {ppl:>12} {rd:>13} {rep:>12}")

    print("═" * 78)
    print()
    print("  Notes:")
    print("  • Perplexity  — lower is better (not applicable to baselines)")
    print("  • Rhythm Div. — higher means more diverse note lengths (0–1)")
    print("  • Repetition  — lower means more creative (less repetitive)")
    print("═" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all(checkpoint_path: Optional[Path] = None,
            no_transformer: bool = False) -> Dict:
    """
    Full pipeline:
      1. Generate random baseline MIDIs
      2. Train Markov model + generate MIDIs
      3. Evaluate all three with MIDI metrics
      4. Pull Transformer perplexity from checkpoint
      5. Write comparison JSON + print table
    """
    report = {"models": []}

    # ── 1. Random Baseline ────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 1: Random Note Generator Baseline")
    log.info("=" * 50)
    random_dir   = Config.OUTPUTS_DIR / "baselines" / "random"
    random_files = run_random_baseline(n_files=10, out_dir=random_dir)
    random_metrics = evaluate_midi_folder(random_dir)
    report["models"].append({
        "model":       "Random Generator",
        "perplexity":  None,     # not applicable
        **random_metrics,
    })

    # ── 2. Markov Chain Baseline ──────────────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 2: Markov Chain Baseline")
    log.info("=" * 50)
    markov_dir   = Config.OUTPUTS_DIR / "baselines" / "markov"
    markov_files = run_markov_baseline(n_files=10, out_dir=markov_dir)
    markov_metrics = evaluate_midi_folder(markov_dir)
    report["models"].append({
        "model":       "Markov Chain",
        "perplexity":  None,     # not applicable
        **markov_metrics,
    })

    # ── 3. Transformer ────────────────────────────────────────────────────
    if not no_transformer and checkpoint_path and checkpoint_path.exists():
        log.info("=" * 50)
        log.info("STEP 3: Transformer")
        log.info("=" * 50)
        transformer_dir     = Config.GENERATED_MIDIS_DIR
        transformer_metrics = evaluate_midi_folder(transformer_dir)
        transformer_ppl     = get_transformer_perplexity(checkpoint_path)
        report["models"].append({
            "model":       "Task 3: Transformer",
            "perplexity":  transformer_ppl,
            **transformer_metrics,
        })
    elif not no_transformer:
        log.warning("Checkpoint not found — skipping Transformer evaluation.")

    # ── 4. Save JSON ──────────────────────────────────────────────────────
    out_json = Config.OUTPUTS_DIR / "baseline_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Comparison report saved → %s", out_json)

    # ── 5. Print table ────────────────────────────────────────────────────
    print_comparison_table(report)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline comparison for Task 3")
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(Config.CHECKPOINTS_DIR / "transformer_best.pt"),
        help="Path to transformer checkpoint .pt file",
    )
    parser.add_argument(
        "--no_transformer", action="store_true",
        help="Skip Transformer evaluation (useful before training is complete)",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if not args.no_transformer else None
    run_all(checkpoint_path=ckpt, no_transformer=args.no_transformer)
