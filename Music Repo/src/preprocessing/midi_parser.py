"""
midi_parser.py
--------------
Scans the Lakh clean_midi (or any genre-organised MIDI folder) and
produces a unified genre map that all downstream modules consume.

Expected folder layout
----------------------
data/raw_midi/
    classical/
        *.mid
    jazz/
        *.mid
    rock/
        *.mid
    pop/
        *.mid
    electronic/
        *.mid

If you downloaded the Lakh clean_midi subset you can move / symlink
files into these genre folders manually, or point a script at the
MSD genre annotations to sort them automatically.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pretty_midi

# ── project imports ─────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def scan_midi_files(root: Path = Config.RAW_MIDI_DIR) -> Dict[str, str]:
    """
    Walk *root* and collect every .mid / .midi file.

    Returns
    -------
    genre_map : dict
        {relative_path_str : genre_str}
        Only genres present in Config.GENRES are kept.
    """
    genre_map: Dict[str, str] = {}
    unknown: List[Path] = []

    for genre in Config.GENRES:
        genre_dir = root / genre
        if not genre_dir.exists():
            log.warning("Genre folder not found: %s (skipping)", genre_dir)
            continue

        midi_files = list(genre_dir.rglob("*.mid")) + list(genre_dir.rglob("*.midi"))
        log.info("  %-12s  %d files", genre, len(midi_files))

        for fpath in midi_files:
            rel = str(fpath.relative_to(root))
            genre_map[rel] = genre

    if unknown:
        log.warning("%d files had unrecognised genre folders and were skipped.", len(unknown))

    log.info("Total files found: %d", len(genre_map))
    return genre_map


def validate_midi(fpath: Path,
                  min_duration: float = 5.0,
                  max_duration: float = 250000.0,
                  min_notes: int = 50) -> Tuple[bool, str]:
    """
    Basic sanity check on a single MIDI file.

    Parameters
    ----------
    fpath        : path to .mid file
    min_duration : seconds – files shorter than this are discarded
    max_duration : seconds – files longer than this are discarded
    min_notes    : minimum total note count

    Returns
    -------
    (is_valid, reason)
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(fpath))
    except Exception as exc:
        return False, f"parse error: {exc}"

    duration = pm.get_end_time()
    if duration < min_duration:
        return False, f"too short ({duration:.1f}s < {min_duration}s)"
    if duration > max_duration:
        return False, f"too long ({duration:.1f}s > {max_duration}s)"

    note_count = sum(len(inst.notes) for inst in pm.instruments if not inst.is_drum)
    if note_count < min_notes:
        return False, f"too few notes ({note_count} < {min_notes})"

    return True, "ok"


def build_and_save_genre_map(root: Path = Config.RAW_MIDI_DIR,
                              out_path: Path = Config.GENRE_MAP_PATH,
                              validate: bool = True) -> Dict[str, str]:
    """
    Full pipeline: scan → (optionally) validate → save JSON.

    Parameters
    ----------
    root      : MIDI root folder
    out_path  : where to write genre_map.json
    validate  : whether to run MIDI validation (slower but safer)

    Returns
    -------
    Filtered genre_map dict.
    """
    log.info("Scanning MIDI files under: %s", root)
    genre_map = scan_midi_files(root)

    if validate:
        log.info("Validating %d files …", len(genre_map))
        valid_map: Dict[str, str] = {}
        skipped = 0
        for rel, genre in genre_map.items():
            fpath = root / rel
            ok, reason = validate_midi(fpath)
            if ok:
                valid_map[rel] = genre
            else:
                log.debug("SKIP %s — %s", rel, reason)
                skipped += 1
        log.info("Kept: %d  |  Skipped: %d", len(valid_map), skipped)
        genre_map = valid_map

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(genre_map, f, indent=2)
    log.info("Genre map saved → %s", out_path)

    # Print per-genre stats
    from collections import Counter
    counts = Counter(genre_map.values())
    log.info("Per-genre counts: %s", dict(counts))

    return genre_map


def load_genre_map(path: Path = Config.GENRE_MAP_PATH) -> Dict[str, str]:
    """Load a previously saved genre_map.json."""
    with open(path) as f:
        return json.load(f)


def get_midi_path(rel: str, root: Path = Config.RAW_MIDI_DIR) -> Path:
    """Resolve a relative path string back to an absolute Path."""
    return root / rel


# ─────────────────────────────────────────────────────────────────────────────
# MIDI info utility
# ─────────────────────────────────────────────────────────────────────────────

def midi_info(fpath: Path) -> Dict:
    """Return a summary dict for a single MIDI (useful for notebooks)."""
    pm = pretty_midi.PrettyMIDI(str(fpath))
    instruments = [
        {"name": inst.name, "program": inst.program, "is_drum": inst.is_drum,
         "notes": len(inst.notes)}
        for inst in pm.instruments
    ]
    return {
        "path":        str(fpath),
        "duration_s":  pm.get_end_time(),
        "tempo_bpm":   pm.estimate_tempo(),
        "n_instruments": len(pm.instruments),
        "total_notes": sum(i["notes"] for i in instruments),
        "instruments": instruments,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (run as script to build genre map)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_and_save_genre_map(validate=True)
