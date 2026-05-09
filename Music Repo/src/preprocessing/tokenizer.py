"""
tokenizer.py
------------
Wraps the miditok REMI tokeniser to:
  1. Build / train the tokeniser vocabulary on the MIDI corpus.
  2. Tokenise every MIDI → a list of integer token IDs.
  3. Slide a fixed-length window (MAX_SEQ_LEN) over each sequence.
  4. Persist windowed sequences as PyTorch tensors (.pt) for fast loading.

Token format per training sample
---------------------------------
[BOS, t1, t2, ..., t_{SEQ_LEN-2}, EOS]   (length = MAX_SEQ_LEN)

Genre label is stored alongside but NOT prepended to the token sequence;
the Transformer adds it via a dedicated genre embedding (see transformer.py).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI      # miditok >= 2.1
from miditoolkit import MidiFile

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
from preprocessing.midi_parser import load_genre_map, get_midi_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build / load the REMI tokeniser
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer(midi_paths: Optional[List[Path]] = None,
                    save_path: Path = Config.TOKENIZER_PATH) -> REMI:
    """
    Construct and train a REMI tokeniser.

    If *save_path* already exists the tokeniser is loaded from disk;
    otherwise it is trained from scratch and saved.

    Parameters
    ----------
    midi_paths : list of MIDI Paths to build vocabulary from.
                 If None, a default tokeniser with hard-coded vocab is built
                 (fine for REMI which has a fixed vocab).
    save_path  : where to serialise the tokeniser JSON.

    Returns
    -------
    tok : REMI tokeniser instance
    """
    if save_path.exists():
        log.info("Loading existing tokeniser from %s", save_path)
        tok = REMI(params=save_path)
        return tok

    log.info("Building REMI tokeniser …")

    tokenizer_config = TokenizerConfig(
        num_velocities=Config.NB_VELOCITIES,
        use_chords=False,         # keeps vocabulary small
        use_programs=True,        # encode instrument programme number
        use_pitch_bends=False,
        beat_res=Config.BEAT_RES,
        special_tokens=Config.SPECIAL_TOKENS,
    )

    tok = REMI(tokenizer_config)

    # For REMI the vocabulary is rule-based (no BPE needed), but if you
    # want BPE / Unigram compression uncomment the two lines below:
    # if midi_paths:
    #     tok.train(vocab_size=10_000, model="BPE", files_paths=midi_paths)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(save_path)
    log.info("Tokeniser saved → %s  (vocab size: %d)", save_path, len(tok))

    return tok


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Tokenise a single MIDI file
# ─────────────────────────────────────────────────────────────────────────────

def tokenise_midi(fpath: Path, tok: REMI) -> Optional[List[int]]:
    """
    Convert one MIDI file to a flat list of integer token IDs.
    Works with both single-stream and multi-track miditok outputs.
    """
    try:
        tokens_obj = tok(fpath)

        # Case 1: miditok returns one TokSequence
        if hasattr(tokens_obj, "ids"):
            return tokens_obj.ids

        # Case 2: miditok returns a list of TokSequence
        ids: List[int] = []
        for seq in tokens_obj:
            if hasattr(seq, "ids"):
                ids.extend(seq.ids)

        return ids

    except Exception as exc:
        log.warning("Tokenisation failed for %s: %s", fpath, exc)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sliding-window segmentation
# ─────────────────────────────────────────────────────────────────────────────

def sliding_windows(ids: List[int],
                    bos_id: int,
                    eos_id: int,
                    max_len: int = Config.MAX_SEQ_LEN,
                    stride: int = Config.STRIDE) -> List[List[int]]:
    """
    Slice *ids* into overlapping windows of length *max_len*.

    Each window is padded with BOS at the start and EOS at the end;
    the interior length is therefore max_len - 2.

    Parameters
    ----------
    ids     : full token ID sequence for one MIDI file
    bos_id  : BOS token index
    eos_id  : EOS token index
    max_len : total window length including BOS/EOS
    stride  : step between consecutive windows

    Returns
    -------
    List of token-ID lists, each of length *max_len*.
    """
    inner = max_len - 2       # slots available between BOS and EOS
    windows: List[List[int]] = []

    for start in range(0, max(1, len(ids) - inner + 1), stride):
        chunk = ids[start: start + inner]
        if len(chunk) < 4:    # skip trivially short tails
            continue
        # pad right if tail chunk is shorter than inner
        chunk = chunk + [eos_id] * (inner - len(chunk))
        window = [bos_id] + chunk + [eos_id]
        windows.append(window)

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_all(genre_map: Optional[Dict[str, str]] = None,
                   tok: Optional[REMI] = None,
                   out_dir: Path = Config.PROCESSED_DIR) -> Path:
    """
    Tokenise every MIDI in genre_map and write windowed tensors to disk.

    Output layout
    -------------
    processed/
        windows.pt        – LongTensor  [N, MAX_SEQ_LEN]
        genres.pt         – LongTensor  [N]   (genre IDs)
        index.json        – [{rel_path, genre, window_idx}, …]

    Parameters
    ----------
    genre_map : {rel_path: genre_str}  (loaded if None)
    tok       : REMI tokeniser         (built if None)
    out_dir   : where to write outputs

    Returns
    -------
    out_dir Path
    """
    if genre_map is None:
        genre_map = load_genre_map()
    if tok is None:
        tok = build_tokenizer()

    bos_id = tok["BOS_None"]
    eos_id = tok["EOS_None"]

    out_dir.mkdir(parents=True, exist_ok=True)

    all_windows: List[List[int]] = []
    all_genres:  List[int]       = []
    index:       List[Dict]      = []

    total = len(genre_map)
    log.info("Tokenising %d MIDI files …", total)

    for i, (rel, genre_str) in enumerate(genre_map.items()):
        if (i + 1) % 500 == 0:
            log.info("  %d / %d processed …", i + 1, total)

        fpath    = get_midi_path(rel)
        genre_id = Config.GENRE2ID.get(genre_str)
        if genre_id is None:
            continue

        ids = tokenise_midi(fpath, tok)
        if ids is None or len(ids) < 10:
            continue

        windows = sliding_windows(ids, bos_id, eos_id)
        for w_idx, win in enumerate(windows):
            all_windows.append(win)
            all_genres.append(genre_id)
            index.append({"rel_path": rel, "genre": genre_str, "window_idx": w_idx})

    log.info("Total training windows: %d", len(all_windows))

    # Save tensors
    windows_t = torch.tensor(all_windows, dtype=torch.long)  # [N, SEQ_LEN]
    genres_t  = torch.tensor(all_genres,  dtype=torch.long)  # [N]

    torch.save(windows_t, out_dir / "windows.pt")
    torch.save(genres_t,  out_dir / "genres.pt")

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)

    log.info("Preprocessed data saved → %s", out_dir)
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MIDITokenDataset(torch.utils.data.Dataset):
    """
    Loads pre-tokenised windows from disk.

    Each item is:
        input_ids  : LongTensor [SEQ_LEN - 1]   (tokens 0 .. T-2)
        labels     : LongTensor [SEQ_LEN - 1]   (tokens 1 .. T-1)
        genre_id   : int
    """

    def __init__(self, windows: torch.Tensor, genres: torch.Tensor):
        assert windows.shape[0] == genres.shape[0]
        self.windows = windows   # [N, SEQ_LEN]
        self.genres  = genres    # [N]

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        seq      = self.windows[idx]          # [SEQ_LEN]
        genre_id = self.genres[idx].item()
        # Teacher-forcing: input = seq[:-1],  target = seq[1:]
        return {
            "input_ids": seq[:-1],            # [SEQ_LEN - 1]
            "labels":    seq[1:],             # [SEQ_LEN - 1]
            "genre_id":  genre_id,
        }


def load_dataset(processed_dir: Path = Config.PROCESSED_DIR) -> MIDITokenDataset:
    """Load the full pre-tokenised dataset from disk."""
    windows = torch.load(processed_dir / "windows.pt")
    genres  = torch.load(processed_dir / "genres.pt")
    return MIDITokenDataset(windows, genres)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing.midi_parser import load_genre_map
    genre_map = load_genre_map()

    # Build tokeniser first (uses all MIDI files for vocabulary)
    midi_paths = [get_midi_path(rel) for rel in genre_map]
    tok = build_tokenizer(midi_paths=midi_paths)

    # Run full tokenisation pipeline
    preprocess_all(genre_map=genre_map, tok=tok)
