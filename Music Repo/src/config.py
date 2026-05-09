"""
config.py
---------
Central configuration for Task 3: Transformer-Based Music Generator.
All hyperparameters and paths are defined here so every other module
imports from a single source of truth.
"""

from pathlib import Path


class Config:
    # ------------------------------------------------------------------ #
    # Paths                                                                 #
    # ------------------------------------------------------------------ #
    ROOT_DIR        = Path(__file__).resolve().parent.parent
    DATA_DIR        = Path(r"D:\Work\CODE\Music Repo\data\raw_midi\genre_sorted")
    RAW_MIDI_DIR = Path(r"D:\Work\CODE\Music Repo\data\raw_midi\genre_sorted")          # raw .mid files organised by genre
    PROCESSED_DIR = Path(r"D:\Work\CODE\Music Repo\data\processed")         # tokenised .pt tensors
    SPLIT_DIR = Path(r"D:\Work\CODE\Music Repo\data\train_test_split\trans_splits")   # split index files

    OUTPUTS_DIR         = Path(r"D:\Work\CODE\Music Repo\outputs\trans")
    GENERATED_MIDIS_DIR = Path(r"D:\Work\CODE\Music Repo\outputs\generated_midis\transformer_generations")
    PLOTS_DIR           = OUTPUTS_DIR / "plots"
    CHECKPOINTS_DIR     = Path(r"D:\Work\CODE\Music Repo\outputs\trans\checkpoints")

    TOKENIZER_PATH  = PROCESSED_DIR / "tokenizer.json"
    GENRE_MAP_PATH  = Path(r"D:\Work\CODE\Music Repo\data\processed\genre_map.json")  # {rel_path: genre_str}

    # ------------------------------------------------------------------ #
    # Genre Setup                                                           #
    # ------------------------------------------------------------------ #
    # Folder names inside RAW_MIDI_DIR must match these strings exactly.
    # e.g.  data/raw_midi/classical/beethoven_op27.mid
    GENRES     = ["classical"]
    GENRE2ID   = {g: i for i, g in enumerate(GENRES)}
    ID2GENRE   = {i: g for i, g in enumerate(GENRES)}
    NUM_GENRES = len(GENRES)

    # ------------------------------------------------------------------ #
    # Tokeniser                                                             #
    # ------------------------------------------------------------------ #
    # REMI tokenisation scheme (miditok)
    TOKENIZER_TYPE   = "REMI"          # options: REMI | MIDILike | TSD
    BEAT_RES         = {(0, 4): 8, (4, 12): 4}   # ticks-per-beat resolution
    NB_VELOCITIES    = 32              # velocity bins
    SPECIAL_TOKENS   = ["PAD", "BOS", "EOS", "MASK"]
    MAX_SEQ_LEN      = 512            # tokens per training window (stride = 256)
    STRIDE           = 256

    # ------------------------------------------------------------------ #
    # Model Architecture                                                    #
    # ------------------------------------------------------------------ #
    # Vocabulary size is updated at runtime after the tokeniser is built.
    VOCAB_SIZE    = 30_000
    D_MODEL       = 256        # embedding dimension
    N_HEADS       = 8          # attention heads  (D_MODEL must be divisible)
    N_LAYERS      = 6          # decoder blocks
    D_FF          = 1024       # feed-forward inner dimension
    DROPOUT       = 0.1
    MAX_POSITION  = 1024       # maximum positional encoding length

    # ------------------------------------------------------------------ #
    # Training                                                              #
    # ------------------------------------------------------------------ #
    BATCH_SIZE      = 32
    LEARNING_RATE   = 3e-4
    WARMUP_STEPS    = 1_000
    EPOCHS          = 11
    GRAD_CLIP       = 1.0
    VAL_SPLIT       = 0.10     # 10 % of data for validation
    TEST_SPLIT      = 0.10     # 10 % of data for test
    SEED            = 42

    # ------------------------------------------------------------------ #
    # Generation                                                            #
    # ------------------------------------------------------------------ #
    GEN_MAX_LEN           = 1_024   # tokens to generate
    TEMPERATURE           = 0.9
    TOP_K                 = 50
    NUM_SAMPLES_PER_GENRE = 10       # 2 × 5 genres = 10 compositions
