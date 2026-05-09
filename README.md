# The-Music-Locker
# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425 / EEE474 — Neural Networks  
**Institution:** Brac University, Dhaka, Bangladesh

**This project represents a collaborative effort with equal contributions from Abrar Masud and Sadib Jaman.**

---
##DRIVE LINK FOR REPORT AND MIDI FILE
https://drive.google.com/drive/folders/16rpAEPi8G3_5zmhd-RVarfDxDWjb2tNy?usp=sharing

##Youtube guide
https://www.youtube.com/watch?v=mxVp65-NcgU


## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Data Directory Layout](#data-directory-layout)
5. [File-by-File Reference](#file-by-file-reference)
   - [config.py](#configpy)
   - [preprocessing/](#preprocessing)
   - [models/](#models)
   - [training/](#training)
   - [evaluation/](#evaluation)
   - [generation/](#generation)
6. [Pipeline: How to Run End-to-End](#pipeline-how-to-run-end-to-end)
7. [Outputs Reference](#outputs-reference)
8. [Key Results](#key-results)
9. [Dependencies](#dependencies)

---

## Project Overview

This project implements a progressive four-stage pipeline of unsupervised deep learning architectures for symbolic multi-genre music generation using MIDI data. No genre labels are required during training.

| Task | Model | Dataset | Purpose |
|------|-------|---------|---------|
| Task 1 | LSTM Autoencoder | MAESTRO | Single-genre reconstruction baseline |
| Task 2 | Variational Autoencoder (VAE) | Lakh MIDI | Multi-genre probabilistic generation |
| Task 3 | Causal Transformer | MAESTRO (tokenized) | Long-horizon autoregressive generation |

---

## Project Structure

```
src/
├── config.py                          # Central configuration for all modules
│
├── preprocessing/
│   ├── get_files_from_mae.py          # Copies MAESTRO .mid files into genre folder
│   ├── piano_roll_maes.py             # MAESTRO → binary piano-roll (Task 1)
│   ├── piano_roll_lakh.py             # Lakh MIDI → normalized piano-roll (Task 2)
│   ├── midi_parser.py                 # Scans & validates MIDI files, builds genre map
│   ├── midi_parser_lakh.py            # Lakh MIDI dataset loader class
│   ├── dataset_split_maes.py          # Train/val/test split for MAESTRO (.npy)
│   ├── dataset_split_lakh.py          # Train/val/test split for Lakh MIDI (.pt)
│   └── tokenizer.py                   # REMI tokenization pipeline (Task 3)
│
├── models/
│   ├── autoencoder.py                 # LSTM Autoencoder architecture (Task 1)
│   ├── vae.py                         # VAE architecture + loss function (Task 2)
│   └── transformer.py                 # Decoder-only Transformer architecture (Task 3)
│
├── training/
│   ├── train_ae.py                    # Training loop for LSTM Autoencoder
│   ├── train_vae.py                   # Training loop for VAE
│   └── train_transformer.py           # Training loop for Transformer
│
├── evaluation/
│   ├── loss_plot_mae.py               # Plots Task 1 reconstruction loss curve
│   ├── loss_plot_lakh.py              # Plots Task 2 VAE loss curves (3 plots)
│   ├── metrics_lakh.py                # Computes Task 1 vs Task 2 comparison table
│   ├── metrics_trans.py               # Computes perplexity, rhythm diversity, repetition ratio
│   └── baseline_comparison.py         # Generates baselines and comparison table (Task 3)
│
└── generation/
    ├── generate_music_mae.py          # Generates 5 MIDI samples from LSTM Autoencoder
    ├── generate_music_lakh.py         # Generates 8 MIDI samples from VAE
    ├── latent_interpolation.py        # VAE latent space interpolation experiment
    ├── generate_music.py              # Generates 10 MIDI samples from Transformer
    └── midi_export.py                 # Shared utility: piano-roll → PrettyMIDI object
```

---

## Setup and Installation

### 1. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install pretty_midi numpy matplotlib
pip install miditok miditoolkit
```

### 2. Download datasets

- **MAESTRO v3:** https://magenta.tensorflow.org/datasets/maestro  
  Extract into `data/raw_midi/maestro-v3.0.0/`

- **Lakh MIDI Dataset:** https://colinraffel.com/projects/lmd/  
  Extract the clean_midi subset into `data/raw_midi/lakh/`

---

## Data Directory Layout

```
data/
├── raw_midi/
│   ├── maestro-v3.0.0/         ← Original MAESTRO files (nested by year)
│   ├── lakh/                   ← Lakh clean_midi files
│   └── genre_sorted/
│       └── classical/          ← MAESTRO .mid files copied here by get_files_from_mae.py
│
├── processed/
│   ├── maestro_processed.npy   ← Task 1 piano-roll tensor (88892, 128, 128)
│   ├── lakh_matrix.pt          ← Task 2 piano-roll tensor (139297, 128, 88)
│   ├── windows.pt              ← Task 3 tokenized windows (125626, 512)
│   ├── genres.pt               ← Task 3 genre labels (125626,)
│   ├── tokenizer.json          ← Saved REMI tokenizer vocabulary
│   └── genre_map.json          ← {relative_midi_path: genre_string}
│
└── train_test_split/
    ├── maestro_splits/
    │   ├── train.npy           ← Task 1 train  (71113, 128, 128)
    │   ├── val.npy             ← Task 1 val    (8889, 128, 128)
    │   └── test.npy            ← Task 1 test   (8890, 128, 128)
    ├── lakh_train.pt           ← Task 2 train  (111437, 128, 88)
    ├── lakh_val.pt             ← Task 2 val    (13929, 128, 88)
    ├── lakh_test.pt            ← Task 2 test   (13931, 128, 88)
    └── trans_splits/
        ├── train_indices.json  ← Task 3 train indices
        ├── val_indices.json    ← Task 3 val indices
        └── test_indices.json   ← Task 3 test indices
```

---

## File-by-File Reference

---

### `config.py`

**Purpose:** Single source of truth for every hyperparameter and file path used across the entire project. Every other module imports from here — you should never hardcode a path or hyperparameter anywhere else.

**Key settings:**

| Section | Setting | Value | Description |
|---------|---------|-------|-------------|
| Paths | `DATA_DIR` | (your path) | Root data directory |
| Paths | `PROCESSED_DIR` | `data/processed/` | Preprocessed tensor outputs |
| Paths | `SPLIT_DIR` | `data/train_test_split/trans_splits/` | Transformer split indices |
| Paths | `OUTPUTS_DIR` | `outputs/trans/` | Training outputs and plots |
| Genre | `GENRES` | `["classical"]` | Active genre list |
| Tokenizer | `MAX_SEQ_LEN` | `512` | Token window length |
| Tokenizer | `STRIDE` | `256` | Sliding window stride |
| Model | `D_MODEL` | `256` | Transformer embedding dimension |
| Model | `N_HEADS` | `8` | Number of attention heads |
| Model | `N_LAYERS` | `6` | Number of Transformer decoder layers |
| Model | `D_FF` | `1024` | Feed-forward inner dimension |
| Training | `BATCH_SIZE` | `32` | Batch size (Transformer) |
| Training | `LEARNING_RATE` | `3e-4` | AdamW learning rate |
| Training | `EPOCHS` | `11` | Training epochs (Transformer) |
| Training | `WARMUP_STEPS` | `1000` | LR warmup steps |
| Generation | `TEMPERATURE` | `0.9` | Sampling temperature |
| Generation | `TOP_K` | `50` | Top-k filtering |

**How to use:** Open this file first. Change `DATA_DIR`, `RAW_MIDI_DIR`, `PROCESSED_DIR`, `OUTPUTS_DIR`, and `GENERATED_MIDIS_DIR` to your local paths before running anything else.

---

### `preprocessing/`

---

#### `preprocessing/get_files_from_mae.py`

**Purpose:** One-time utility that copies all `.mid` and `.midi` files from the nested MAESTRO v3 folder structure into a flat `genre_sorted/classical/` folder. This is needed because `midi_parser.py` expects MIDI files organized by genre subfolder.

**Input:** `data/raw_midi/maestro-v3.0.0/` (MAESTRO's original year-nested structure)  
**Output:** `data/raw_midi/genre_sorted/classical/` (flat folder of all MAESTRO .mid files)

**Run once before any other preprocessing:**
```bash
python src/preprocessing/get_files_from_mae.py
```

**Note:** Handles filename collisions by appending a counter suffix to avoid overwriting.

---

#### `preprocessing/piano_roll_maes.py`

**Purpose:** Converts all MAESTRO MIDI files into binary piano-roll matrices for **Task 1** (LSTM Autoencoder). This is the first preprocessing step for Task 1.

**What it does step-by-step:**
1. Walks the raw MIDI directory recursively to find all `.mid` files
2. Loads each file with `pretty_midi` and calls `get_piano_roll(fs=16)` — this samples the music 16 times per second
3. Binarizes velocities: any active note becomes `1.0`, silence becomes `0.0`
4. Transposes to shape `(time_steps, 128_pitches)`
5. Slices into non-overlapping windows of length 128 (one training sample = 8 bars)
6. Stacks all windows into one large array and saves it

**Input:** Raw `.mid` files from MAESTRO  
**Output:** `data/processed/maestro_processed.npy` — shape `(88892, 128, 128)` — dtype `float32`

**Representation:** Each sample is a `128 × 128` grid where rows = time steps and columns = MIDI pitches 0–127.

```bash
python src/preprocessing/piano_roll_maes.py
```

---

#### `preprocessing/piano_roll_lakh.py`

**Purpose:** Converts Lakh MIDI files into normalized piano-roll matrices for **Task 2** (VAE). Uses the standard 88-key piano range rather than the full 128-pitch MIDI range.

**What it does step-by-step:**
1. Loads a MIDI file and extracts the piano roll at `fs=16` Hz
2. Slices pitch axis to keep only MIDI pitches 21–108 (the 88-key piano range)
3. Transposes to `(time_steps, 88)`
4. Normalizes velocity values to `[0, 1]` by dividing by 127 (not binarized — keeps dynamics)
5. Segments into non-overlapping windows of length 128
6. Returns windows as a numpy array of shape `(n_windows, 128, 88)`

**Input:** A single MIDI file path  
**Output:** Numpy array `(n_windows, 128, 88)` or `None` if the file is corrupted/too short

**Used by:** `midi_parser_lakh.py` which calls this function for every file in the dataset.

**Key difference from `piano_roll_maes.py`:** Uses 88 pitches instead of 128, and keeps continuous velocity values instead of binarizing.

---

#### `preprocessing/midi_parser.py`

**Purpose:** Scans and validates MIDI files organized by genre subfolder, then builds and saves a `genre_map.json` file. This JSON is the master index consumed by `tokenizer.py` for Task 3 preprocessing.

**What it does:**
- `scan_midi_files()` — walks `data/raw_midi/genre_sorted/` and maps each file's relative path to its genre string (folder name)
- `validate_midi()` — filters out files shorter than 5 seconds, longer than 250,000 seconds, or with fewer than 50 notes
- `build_and_save_genre_map()` — runs scan + validate, then saves `{relative_path: genre_string}` to `genre_map.json`
- `midi_info()` — utility that returns a summary dict (duration, tempo, instrument list) for a single MIDI file

**Expected folder layout:**
```
data/raw_midi/genre_sorted/
    classical/  *.mid
    jazz/       *.mid
    rock/       *.mid
```

**Input:** Genre-sorted MIDI folder  
**Output:** `data/processed/genre_map.json`

```bash
python src/preprocessing/midi_parser.py
```

---

#### `preprocessing/midi_parser_lakh.py`

**Purpose:** PyTorch-compatible dataset class that loads and preprocesses the entire Lakh MIDI collection for **Task 2**. Wraps `piano_roll_lakh.py`'s conversion function and stacks all windows into one large tensor.

**Class: `LakhMidiDataset`**
- Constructor walks the Lakh MIDI directory, processes up to `max_files` files (default 5000, to prevent RAM overflow), and concatenates all windows into a single `FloatTensor`
- `max_files` parameter is critical — processing all 170,000 Lakh files at once would exhaust system RAM

**Input:** Path to Lakh MIDI directory  
**Output:** Saves `data/processed/lakh_matrix.pt` — shape `(139297, 128, 88)`

```bash
python src/preprocessing/midi_parser_lakh.py
```

---

#### `preprocessing/dataset_split_maes.py`

**Purpose:** Splits the MAESTRO piano-roll numpy array into train/validation/test sets for **Task 1**.

**Split ratio:** 80% train / 10% val / 10% test (shuffled with `np.random.permutation`)

**Input:** `data/processed/maestro_processed.npy`  
**Output:**
- `data/train_test_split/maestro_splits/train.npy` — shape `(71113, 128, 128)`
- `data/train_test_split/maestro_splits/val.npy` — shape `(8889, 128, 128)`
- `data/train_test_split/maestro_splits/test.npy` — shape `(8890, 128, 128)`

```bash
python src/preprocessing/dataset_split_maes.py
```

---

#### `preprocessing/dataset_split_lakh.py`

**Purpose:** Splits the Lakh MIDI piano-roll tensor into train/validation/test sets for **Task 2**.

**Split ratio:** 80% train / 10% val / 10% test (shuffled with `torch.randperm`)

**Input:** `data/processed/lakh_matrix.pt`  
**Output:**
- `data/train_test_split/lakh_train.pt` — shape `(111437, 128, 88)`
- `data/train_test_split/lakh_val.pt` — shape `(13929, 128, 88)`
- `data/train_test_split/lakh_test.pt` — shape `(13931, 128, 88)`

```bash
python src/preprocessing/dataset_split_lakh.py
```

---

#### `preprocessing/tokenizer.py`

**Purpose:** Full REMI tokenization pipeline for **Task 3**. Converts MIDI files into integer token sequences, applies sliding-window segmentation, and saves the result as PyTorch tensors ready for Transformer training.

**Key components:**

| Function / Class | Description |
|-----------------|-------------|
| `build_tokenizer()` | Builds or loads a REMI tokenizer from `miditok`. Saves vocab to `tokenizer.json`. Vocabulary is rule-based (no BPE needed for REMI). |
| `tokenise_midi()` | Converts one MIDI file to a flat list of integer token IDs using the REMI scheme. Handles both single-stream and multi-track miditok outputs. |
| `sliding_windows()` | Slices a token sequence into overlapping windows of length `MAX_SEQ_LEN=512` with stride `STRIDE=256`. Each window is bookended by `[BOS]...[EOS]`. |
| `preprocess_all()` | Master pipeline: loads genre map → tokenizes every MIDI → segments into windows → saves `windows.pt`, `genres.pt`, `index.json`. |
| `MIDITokenDataset` | PyTorch `Dataset` that loads pre-tokenized windows and returns `{input_ids, labels, genre_id}` dicts for teacher-forcing training. `input_ids = seq[:-1]`, `labels = seq[1:]`. |
| `load_dataset()` | Convenience function that loads `windows.pt` and `genres.pt` and returns a `MIDITokenDataset`. |

**Token encoding:**
- Pitches: tokens 0–127 (MIDI pitch values)
- Rest: token 128
- Sequence start: token 129 (BOS)
- Special tokens: PAD, BOS, EOS, MASK

**Input:** `data/processed/genre_map.json` + raw MIDI files  
**Output:**
- `data/processed/windows.pt` — LongTensor `(125626, 512)`
- `data/processed/genres.pt` — LongTensor `(125626,)`
- `data/processed/index.json` — metadata list per window

```bash
python src/preprocessing/tokenizer.py
```

---

### `models/`

---

#### `models/autoencoder.py`

**Purpose:** Defines the `LSTMAutoencoder` class used in **Task 1**.

**Architecture:**

```
Input: (Batch, 128, 128)  [batch × time_steps × pitches]
  ↓
Encoder LSTM  (input=128, hidden=64, batch_first=True)
  → takes the final hidden state as the latent vector z ∈ R^64
  ↓
Repeat z → (Batch, 128, 64)  [broadcast across 128 time steps]
  ↓
Decoder LSTM  (input=64, hidden=64, batch_first=True)
  ↓
Linear(64 → 128)
  ↓
Sigmoid activation
Output: (Batch, 128, 128)  — reconstructed piano roll
```

**Key design choices:**
- The latent vector `z` is repeated 128 times along the time axis to feed the decoder, making reconstruction conditioned entirely on the compressed representation
- Sigmoid output constrains values to `[0, 1]`, matching the binary piano-roll target
- Loss function: MSE (Mean Squared Error) between input and reconstruction

**Constructor parameters:**
- `input_dim=128` — number of MIDI pitches
- `hidden_dim=64` — size of the latent bottleneck
- `seq_len=128` — number of time steps per sequence

**Saved to:** `outputs/lstm_autoencoder.pth`

---

#### `models/vae.py`

**Purpose:** Defines the `VAE` class and `vae_loss_function` used in **Task 2**.

**Architecture:**

```
Input: (Batch, 128, 88)  [batch × time_steps × pitches]
  ↓
Encoder LSTM  (input=88, hidden=256, batch_first=True)
  → final hidden state h_last ∈ R^256
  ↓
fc_mu(256 → 64)     → μ ∈ R^64
fc_logvar(256 → 64) → log σ² ∈ R^64
  ↓
Reparameterization: z = μ + σ ⊙ ε,  ε ~ N(0, I)
  ↓
decoder_fc(64 → 256)  → initial hidden state h_0
  ↓
Decoder LSTM  (input=88, hidden=256, batch_first=True)
  [initialized with h_0; input is zeros of shape (Batch, seq_len, 88)]
  ↓
Linear(256 → 88) + Sigmoid
Output: (Batch, 128, 88)  — reconstructed piano roll
```

**`vae_loss_function(recon_x, x, mu, logvar, beta=1.0)`:**
- `L_recon` = MSE loss (sum reduction)
- `L_KL` = −½ Σ(1 + log σ² − μ² − σ²)
- `L_VAE` = `L_recon` + `beta × L_KL`
- `beta` is annealed from 0.1 to 1.0 during training to prevent posterior collapse

**Constructor parameters:**
- `input_dim=88` — standard 88-key piano range
- `hidden_dim=256` — LSTM hidden size
- `latent_dim=64` — latent space dimension

**Methods:**
- `encode(x)` → `(mu, logvar)`
- `reparameterize(mu, logvar)` → `z`
- `decode(z, seq_len)` → reconstructed piano roll
- `forward(x)` → `(reconstruction, mu, logvar)`

**Saved to:** `outputs/vae_model.pth`

---

#### `models/transformer.py`

**Purpose:** Defines the `MusicTransformer` class — a decoder-only (GPT-style) causal Transformer for **Task 3**.

**Sub-modules:**

`PositionalEncoding` — Sinusoidal positional encoding:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

`TransformerBlock` — One causal self-attention decoder block:
- Pre-LayerNorm multi-head self-attention with causal mask (upper-triangular `-inf` mask)
- Pre-LayerNorm position-wise feed-forward network (Linear → GELU → Dropout → Linear)
- Residual connections around both sub-layers

`MusicTransformer` — Full model:
```
Input tokens: (Batch, T)
  ↓
Token embedding  (vocab_size → d_model=256)
Genre embedding  (num_genres → d_model=256)  ← broadcast over all T positions
  h_t = TokenEmb(x_t) + PositionalEnc(t) + GenreEmb(genre)
  ↓
6 × TransformerBlock  (causal self-attention + FFN)
  ↓
LayerNorm
  ↓
LM head: Linear(d_model → vocab_size)  ← weight tied to token embedding
Output logits: (Batch, T, vocab_size)
```

**Key features:**
- **Weight tying:** LM head weights are shared with the token embedding matrix (reduces parameters and improves generalization)
- **Genre conditioning:** A single genre embedding vector is added to every token embedding position, giving the model a persistent genre signal throughout the sequence
- **Causal masking:** `make_causal_mask()` produces an upper-triangular `-inf` additive mask, ensuring token `t` can only attend to positions `t' < t`
- **Padding mask:** Automatically ignores PAD token positions in attention

**Constructor parameters (from `config.py`):**
- `vocab_size=30000`, `d_model=256`, `n_heads=8`, `n_layers=6`
- `d_ff=1024`, `dropout=0.1`, `max_len=1024`, `num_genres=1`

**`forward(input_ids, genre_ids, labels=None)`:**  
Returns `(logits, loss)`. Loss is cross-entropy computed only over non-PAD positions when `labels` is supplied.

**Saved to:** `outputs/trans/checkpoints/transformer_best.pt`

---

### `training/`

---

#### `training/train_ae.py`

**Purpose:** Full training loop for the **Task 1** LSTM Autoencoder.

**What it does:**
1. Loads `train.npy` and `val.npy` as `MidiDataset` PyTorch datasets
2. Instantiates `LSTMAutoencoder(input_dim=128, hidden_dim=64, seq_len=128)`
3. Trains with MSE loss and Adam optimizer (`lr=0.001`) for 20 epochs, batch size 64
4. Tracks train and validation MSE loss per epoch
5. Prints epoch-level progress to console
6. Saves the trained model weights and full loss history

**Hyperparameters:**
- Epochs: 20
- Batch size: 64
- Learning rate: 0.001
- Loss: `nn.MSELoss()`
- Optimizer: `Adam`

**Outputs:**
- `outputs/lstm_autoencoder.pth` — trained model weights
- `outputs/loss_history_ae.npy` — dictionary with `train_loss` and `val_loss` lists

```bash
python src/training/train_ae.py
```

---

#### `training/train_vae.py`

**Purpose:** Full training loop for the **Task 2** Variational Autoencoder.

**What it does:**
1. Loads `lakh_train.pt` and `lakh_val.pt` as `TensorDataset` objects
2. Instantiates `VAE(input_dim=88, hidden_dim=256, latent_dim=64)`
3. Trains for 50 epochs with `beta=1.0` (KL weight) and Adam optimizer (`lr=0.001`), batch size 64
4. Tracks four loss components per epoch: total, reconstruction, KL, and validation total
5. Saves a checkpoint every 10 epochs
6. Saves the full loss history for plotting

**Hyperparameters:**
- Epochs: 50
- Batch size: 64
- Learning rate: 0.001
- Beta (KL weight): 1.0
- Loss: MSE reconstruction + β × KL divergence

**Note on posterior collapse:** If the model generates silence (rhythm diversity = 0.00), retrain with `beta` starting at 0.1 and gradually increasing to 1.0 (β-annealing). Modify the `BETA` variable or add a scheduler in the training loop.

**Outputs:**
- `outputs/vae_model.pth` — trained model weights
- `outputs/vae_loss_history.pt` — dictionary with `train_total_loss`, `train_recon_loss`, `train_kl_loss`, `val_total_loss`

```bash
python src/training/train_vae.py
```

---

#### `training/train_transformer.py`

**Purpose:** Full training loop for the **Task 3** Causal Transformer. The most complete training script in the project.

**What it does:**
1. Sets global random seed for reproducibility (`seed=42`)
2. Loads the pre-tokenized `MIDITokenDataset` and splits into train/val/test using `make_splits()`
3. Saves split indices to JSON files so the exact same split can be reproduced
4. Builds the `MusicTransformer` model and logs total parameter count
5. Trains with AdamW optimizer and a **cosine LR schedule with linear warmup**:
   - Steps 0 → `WARMUP_STEPS`: learning rate increases linearly from 0 to `LEARNING_RATE`
   - Steps `WARMUP_STEPS` → end: learning rate decreases following a cosine curve
6. Applies gradient clipping at norm 1.0 every step
7. Logs batch-level progress every 100 batches with loss, perplexity, elapsed time, and ETA
8. Saves two checkpoints after each epoch: `transformer_latest.pt` (always) and `transformer_best.pt` (only when validation loss improves)
9. After training, loads the best checkpoint and evaluates on the test set
10. Saves test results to `test_results.json` and plots training curves

**Hyperparameters (from config.py):**
- Epochs: 11
- Batch size: 32
- Learning rate: 3e-4
- Warmup steps: 1000
- Gradient clip: 1.0
- Val/Test split: 10% each

**Checkpoint format (`.pt` file contains):**
- `model` — model state dict
- `optimizer` — optimizer state dict
- `epoch` — epoch number
- `val_loss` — best validation loss
- `history` — full loss/perplexity history dict
- `vocab_size` — tokenizer vocabulary size

**Outputs:**
- `outputs/trans/checkpoints/transformer_best.pt`
- `outputs/trans/checkpoints/transformer_latest.pt`
- `outputs/trans/test_results.json`
- `outputs/trans/plots/transformer_loss_curve.png`
- `outputs/trans/plots/transformer_perplexity_curve.png`

```bash
# Normal training
python src/training/train_transformer.py

# Resume from latest checkpoint
python src/training/train_transformer.py --resume
```

---

### `evaluation/`

---

#### `evaluation/loss_plot_mae.py`

**Purpose:** Loads `loss_history_ae.npy` saved by `train_ae.py` and produces the **Task 1** reconstruction loss plot.

**What it plots:** Single figure with train MSE loss (blue solid) and validation MSE loss (red dashed) on the y-axis over epochs on the x-axis.

**Input:** `outputs/loss_history_ae.npy`  
**Output:** `outputs/plots/reconstruction_loss_curve.png`

```bash
python src/evaluation/loss_plot_mae.py
```

---

#### `evaluation/loss_plot_lakh.py`

**Purpose:** Loads `vae_loss_history.pt` saved by `train_vae.py` and produces three **Task 2** training plots.

**What it plots:**

| Plot | Filename | Content |
|------|----------|---------|
| Plot 1 | `reconstruction_loss.png` | Training reconstruction (MSE) loss per epoch |
| Plot 2 | `kl_divergence_loss.png` | KL divergence loss per epoch |
| Plot 3 | `total_vae_loss.png` | Total VAE objective (Recon + β·KL) per epoch |

**Input:** `outputs/lakh/vae_loss_history.pt`  
**Output:** Three `.png` files saved to `outputs/plots/`

The KL divergence plot is particularly important — a steadily rising KL confirms the model is learning a non-collapsed latent distribution. A flat KL near zero indicates posterior collapse.

```bash
python src/evaluation/loss_plot_lakh.py
```

---

#### `evaluation/metrics_lakh.py`

**Purpose:** Computes and prints the **Task 1 vs Task 2 comparison table** (Table 3 in the report).

**Metrics computed:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| Pitch Histogram | H(p,q) = Σ\|p_i − q_i\| | Difference between generated and test pitch distributions |
| Rhythm Diversity | D = unique\_durations / total\_notes | Higher = more varied note lengths |
| Repetition Ratio | R = repeated\_patterns / total\_patterns | Lower = more creative output |

**Functions:**
- `get_piano_roll_from_midi()` — loads a MIDI and converts to 88-key piano roll at 16 Hz
- `calculate_pitch_histogram()` — collapses 88 keys into 12 pitch classes, returns normalized histogram
- `calculate_rhythm_diversity()` — counts unique patterns of simultaneous note counts
- `calculate_repetition_ratio()` — compares 16-step windows for repeated content
- `evaluate_and_compare()` — loads loss histories and generated MIDIs, prints the full comparison table

**Input:** Generated MIDI samples from `outputs/generated_midis/`, loss histories  
**Output:** Formatted comparison table printed to console

```bash
python src/evaluation/metrics_lakh.py
```

---

#### `evaluation/metrics_trans.py`

**Purpose:** All quantitative evaluation metrics for **Task 3** — perplexity on the test set, per-file MIDI metrics, and a full evaluation report.

**Functions:**

| Function | Description |
|----------|-------------|
| `compute_model_perplexity()` | Runs the Transformer over the test DataLoader and computes token-level average NLL and perplexity = exp(NLL) |
| `pitch_histogram()` | Returns normalized 12-bin pitch-class histogram for a MIDI file |
| `pitch_histogram_similarity()` | H(p,q) = Σ\|p_i − q_i\| between two MIDI files (lower = more similar) |
| `rhythm_diversity()` | D = unique\_durations / total\_notes. Durations quantized into 16 bins |
| `repetition_ratio()` | R = repeated 4-note-pattern tuples / total patterns |
| `generate_perplexity_report()` | Loads checkpoint, evaluates on test set, writes full JSON report |

**Output:** `outputs/perplexity_report.json` + formatted table printed to console

```bash
python src/evaluation/metrics_trans.py --checkpoint outputs/trans/checkpoints/transformer_best.pt
```

---

#### `evaluation/baseline_comparison.py`

**Purpose:** Implements both baseline models and generates the **Task 3 baseline comparison table**.

**Baseline 1 — Random Note Generator (`generate_random_midi`):**  
Samples pitches uniformly from the comfortable piano range (C3–C6, MIDI 48–84), with random durations (32nd to half note) and velocities. Produces musically incoherent output by design. Serves as the absolute lower bound.

**Baseline 2 — Markov Chain (`MarkovMusicModel`):**  
A second-order pitch Markov chain trained on MIDI files from `data/raw_midi/`. Models transition probabilities P(x_t | x_{t-1}, x_{t-2}) for pitch and P(d_t | d_{t-1}) for duration independently. Falls back to uniform sampling for unseen state transitions. Note: perplexity is reported as N/A because unseen transitions in the test set produce probability zero, making perplexity mathematically infinite.

**Pipeline:**
1. Generates 10 random MIDI files → saves to `outputs/trans/baselines/random/`
2. Trains Markov model on training MIDIs → generates 10 files → saves to `outputs/trans/baselines/markov/`
3. Evaluates all folders with rhythm diversity and repetition ratio
4. Pulls Transformer perplexity from checkpoint
5. Prints and saves a formatted comparison table

**Output:** `outputs/trans/baseline_comparison.json` + table printed to console

```bash
python src/evaluation/baseline_comparison.py \
    --checkpoint outputs/trans/checkpoints/transformer_best.pt
```

---

### `generation/`

---

#### `generation/midi_export.py`

**Purpose:** Shared utility module that converts a 2D piano-roll numpy array into a `PrettyMIDI` object. Used by both `generate_music_lakh.py` and `latent_interpolation.py`.

**Function: `piano_roll_to_midi(piano_roll, fs=16, program=0)`**
- Input: `(seq_len, 88)` float array with values in `[0, 1]` (sigmoid outputs from decoder)
- Pads 88-key roll back to full 128-pitch MIDI range by mapping columns to pitches 21–108
- Applies threshold of 0.5 to binarize sigmoid outputs into note-on / note-off events
- Finds contiguous blocks of active frames per pitch using `_find_contiguous_blocks()`
- Creates `pretty_midi.Note` objects with `velocity=100`, `start=frame/fs`, `end=frame/fs`
- Returns a `PrettyMIDI` object ready to write with `.write(filename)`

**Note:** `fs=16` means each frame represents 1/16 of a second. A 128-frame sequence is 8 seconds of music.

---

#### `generation/generate_music_mae.py`

**Purpose:** Generates **5 MIDI samples** from the trained **Task 1** LSTM Autoencoder by sampling random latent vectors.

**How generation works:**
1. Loads trained `LSTMAutoencoder` from `outputs/lstm_autoencoder.pth`
2. Samples `z ~ N(0, 1)` of shape `(1, 64)` — a random point in latent space
3. Repeats `z` 128 times to create decoder input `(1, 128, 64)`
4. Passes through the decoder LSTM and output linear layer
5. Applies sigmoid to get piano-roll values in `[0, 1]`
6. Converts to MIDI using `matrix_to_midi()` (inline version of midi_export logic)
7. Saves each sample as `sample_1.mid` through `sample_5.mid`

**Output:** 5 `.mid` files in `outputs/generated_midis/`

```bash
python src/generation/generate_music_mae.py
```

---

#### `generation/generate_music_lakh.py`

**Purpose:** Generates **8 MIDI samples** from the trained **Task 2** VAE by sampling from the prior distribution.

**How generation works:**
1. Loads trained `VAE` from `outputs/lakh/vae_model.pth`
2. Samples `z ~ N(0, I)` of shape `(8, 64)` — 8 random points in the 64-dimensional latent space
3. Passes all 8 latent vectors through `model.decode(z, seq_len=128)` in one batch
4. Converts each decoded piano roll to MIDI using `piano_roll_to_midi()` from `midi_export.py`
5. Saves as `vae_sample_1.mid` through `vae_sample_8.mid`

**Output:** 8 `.mid` files in `outputs/generated_midis/`

```bash
python src/generation/generate_music_lakh.py
```

---

#### `generation/latent_interpolation.py`

**Purpose:** **Task 2** latent space interpolation experiment. Generates a sequence of MIDI files that smoothly transition between two random points in the VAE latent space.

**How it works:**
1. Loads trained VAE
2. Samples two random latent vectors `z_start` and `z_end` from `N(0, I)`
3. Computes `num_steps=8` interpolated latent vectors using spherical linear interpolation:
   `z_i = (1 - α) × z_start + α × z_end`  where α ∈ {0.00, 0.14, 0.28, ..., 1.00}
4. Decodes each interpolated `z_i` into a piano roll
5. Converts to MIDI and saves with the alpha value in the filename

**Purpose:** Demonstrates that the VAE latent space is smooth and structured — gradually changing `z` should produce gradually changing music rather than abrupt jumps.

**Output:** 8 MIDI files in `outputs/generated_midis/interpolation/`  
Filenames: `interp_step_1_alpha_0.00.mid` through `interp_step_8_alpha_1.00.mid`

```bash
python src/generation/latent_interpolation.py
```

---

#### `generation/generate_music.py`

**Purpose:** Generates **10 MIDI compositions** from the trained **Task 3** Transformer using autoregressive sampling with top-k filtering and temperature scaling.

**How generation works:**
1. Loads trained `MusicTransformer` from `transformer_best.pt` checkpoint
2. For each genre and each sample:
   - Initializes the sequence with `[BOS]`
   - At each step, runs a forward pass to get logits for the next token
   - Applies temperature scaling: `logits = logits / T` (T=0.9)
   - Applies top-k filtering: zeroes out all logits except the top-50
   - Samples next token from the resulting softmax distribution
   - Appends to sequence; stops at `[EOS]` or `GEN_MAX_LEN=1024` tokens
3. Decodes token sequence back to MIDI using the miditok REMI tokenizer
4. Saves generation metadata to `generation_metadata.json`

**Temperature effect:**
- `T < 1.0` (e.g., 0.85): more conservative, higher-probability tokens favored
- `T > 1.0`: more random and diverse but potentially incoherent
- `T = 0.9` balances coherence and variety

**Output:**
- 10 `.mid` files in `outputs/generated_midis/transformer_generations/`  
  Named `task3_classical_01.mid` through `task3_classical_10.mid`
- `generation_metadata.json` — records genre, token count, temperature, top-k per file

```bash
python src/generation/generate_music.py \
    --checkpoint outputs/trans/checkpoints/transformer_best.pt \
    --n_per_genre 10 \
    --temperature 0.9 \
    --top_k 50
```

---

## Pipeline: How to Run End-to-End

### Task 1 — LSTM Autoencoder

```bash
# Step 1: Copy MAESTRO files into genre folder
python src/preprocessing/get_files_from_mae.py

# Step 2: Convert to piano rolls
python src/preprocessing/piano_roll_maes.py

# Step 3: Split dataset
python src/preprocessing/dataset_split_maes.py

# Step 4: Train
python src/training/train_ae.py

# Step 5: Evaluate (plot loss curve)
python src/evaluation/loss_plot_mae.py

# Step 6: Generate MIDI samples
python src/generation/generate_music_mae.py
```

### Task 2 — VAE

```bash
# Step 1: Preprocess Lakh MIDI
python src/preprocessing/midi_parser_lakh.py

# Step 2: Split dataset
python src/preprocessing/dataset_split_lakh.py

# Step 3: Train
python src/training/train_vae.py

# Step 4: Evaluate (plot loss curves)
python src/evaluation/loss_plot_lakh.py
python src/evaluation/metrics_lakh.py

# Step 5: Generate samples
python src/generation/generate_music_lakh.py

# Step 6: Latent interpolation
python src/generation/latent_interpolation.py
```

### Task 3 — Transformer

```bash
# Step 1: Build genre map
python src/preprocessing/midi_parser.py

# Step 2: Tokenize MIDI files
python src/preprocessing/tokenizer.py

# Step 3: Train
python src/training/train_transformer.py

# Step 4: Evaluate perplexity
python src/evaluation/metrics_trans.py \
    --checkpoint outputs/trans/checkpoints/transformer_best.pt

# Step 5: Run baseline comparison
python src/evaluation/baseline_comparison.py \
    --checkpoint outputs/trans/checkpoints/transformer_best.pt

# Step 6: Generate MIDI samples
python src/generation/generate_music.py \
    --checkpoint outputs/trans/checkpoints/transformer_best.pt
```

---

## Outputs Reference

| File | Created by | Description |
|------|-----------|-------------|
| `outputs/lstm_autoencoder.pth` | `train_ae.py` | Task 1 model weights |
| `outputs/loss_history_ae.npy` | `train_ae.py` | Task 1 train/val MSE per epoch |
| `outputs/vae_model.pth` | `train_vae.py` | Task 2 model weights |
| `outputs/vae_loss_history.pt` | `train_vae.py` | Task 2 loss components per epoch |
| `outputs/trans/checkpoints/transformer_best.pt` | `train_transformer.py` | Task 3 best checkpoint |
| `outputs/trans/checkpoints/transformer_latest.pt` | `train_transformer.py` | Task 3 latest checkpoint |
| `outputs/trans/test_results.json` | `train_transformer.py` | Test loss and perplexity |
| `outputs/plots/reconstruction_loss_curve.png` | `loss_plot_mae.py` | Task 1 loss figure |
| `outputs/plots/reconstruction_loss.png` | `loss_plot_lakh.py` | Task 2 reconstruction loss figure |
| `outputs/plots/kl_divergence_loss.png` | `loss_plot_lakh.py` | Task 2 KL divergence figure |
| `outputs/plots/total_vae_loss.png` | `loss_plot_lakh.py` | Task 2 total ELBO figure |
| `outputs/trans/plots/transformer_loss_curve.png` | `train_transformer.py` | Task 3 NLL loss figure |
| `outputs/trans/plots/transformer_perplexity_curve.png` | `train_transformer.py` | Task 3 perplexity figure |
| `outputs/generated_midis/sample_*.mid` | `generate_music_mae.py` | Task 1 generated MIDI (×5) |
| `outputs/generated_midis/vae_sample_*.mid` | `generate_music_lakh.py` | Task 2 generated MIDI (×8) |
| `outputs/generated_midis/interpolation/*.mid` | `latent_interpolation.py` | Task 2 interpolation MIDI (×8) |
| `outputs/generated_midis/transformer_generations/*.mid` | `generate_music.py` | Task 3 generated MIDI (×10) |
| `outputs/trans/baseline_comparison.json` | `baseline_comparison.py` | Task 3 baseline metrics |
| `outputs/perplexity_report.json` | `metrics_trans.py` | Task 3 full evaluation report |

---

## Key Results

| Model | Loss | Perplexity | Rhythm Diversity | Repetition Ratio |
|-------|------|-----------|-----------------|-----------------|
| Random Generator | — | N/A | 0.0417 | 0.0034 |
| Markov Chain | — | N/A | N/A | N/A |
| Task 1: LSTM AE | 0.0211 (MSE) | N/A | 0.18 | 0.45 |
| Task 2: VAE | 458.34 (ELBO) | N/A | 0.83* | 0.007* |
| Task 3: Transformer | 1.4627 (NLL) | **4.3176** | 0.0402 | 0.3693 |

\* After β-annealing fix

**Test set (Task 3):** NLL = 1.4627 · Perplexity = 4.3176 · Best val loss = 1.4591 · Epochs = 11

---

## Dependencies

```
torch>=2.0.0
torchvision
torchaudio
numpy
matplotlib
pretty_midi
miditok>=2.1.0
miditoolkit
```

Install all at once:
```bash
pip install torch torchvision torchaudio numpy matplotlib pretty_midi miditok miditoolkit
```

Python version: **3.11** (`.pyc` files confirm `cpython-311`)
