# Sentiment Analysis: A Dual-System Framework for Robust Sentiment Classification

> A comparative study and implementation of transformer-based and adaptive lexicon-based sentiment analysis with online learning capabilities.

## Abstract

This repository documents the development and evaluation of two distinct sentiment analysis paradigms: (1) a deep-learning approach utilizing a fine-tuned **XLM-RoBERTa-base** cross-lingual transformer, and (2) a rule-based **adaptive lexicon engine** with online learning capabilities. The transformer model, while achieving 98% accuracy on its fine-tuning validation split, exhibited severe positive-class bias on out-of-distribution inputs. This limitation motivated the development of the adaptive lexicon system, which provides interpretable, controllable predictions with dynamic vocabulary expansion through user feedback. This work demonstrates the fundamental trade-off between the generalization potential of large pre-trained models and the transparency and controllability of rule-based systems in production NLP pipelines.

---

## Architecture Overview

### System 1: Fine-tuned Transformer *(Archived — see `archived_files/`)*

A 3-class sequence classifier built on `xlm-roberta-base` (278M parameters), fine-tuned with a custom **Focal Loss** function and **Adaptive Sample Reweighting** to handle class imbalance.

- **Architecture:** `XLMRobertaForSequenceClassification` — 12 hidden layers, 12 attention heads, hidden size 768, vocabulary 250,002 tokens
- **Status:** Archived due to severe positive-class bias on real-world inputs
- **Location:** Model weights in `./sentiment_model/`, training scripts restored to root (`train.py`, `test.py`) and also preserved in `archived_files/`

### System 2: Adaptive Lexicon Engine *(Active)*

A context-aware rule-based sentiment analyzer with online learning.

| Component | Details |
|-----------|---------|
| Positive lexicon | 47 seed words |
| Negative lexicon | 37 seed words |
| Neutral lexicon | 18 seed words |
| Intensifiers | 25 terms (multiplier: 1.5×) |
| Negation terms | 21 terms (window: *i*-1) |
| Double negation | Resolved to positive |
| Classification thresholds | score > 0.05 → Positive, score < −0.05 → Negative, else Neutral |
| Confidence | \|score\| × max(pos, neg) / (pos + neg) |
| Learning | Online vocabulary expansion from corrections; persisted to `sentiment_knowledge.pkl` |

---

## Quick Start

### Installation (Reproducible Environment)

```bash
# Create isolated environment
conda create -n sentiment-env python=3.9
conda activate sentiment-env

# Install exact frozen dependencies
pip install -r requirements_frozen.txt
```

Alternatively, use the Conda environment file:
```bash
conda env create -f environment.yml
conda activate sentiment-analysis
```

> **Note:** `requirements.txt` contains version *ranges* for flexibility. For exact reproduction of results, always use `requirements_frozen.txt` with pinned versions.

### Running the System

| Entry Point | Description |
|-------------|-------------|
| `python start_system.py` | Interactive system launcher/router (recommended first run) |
| `python fixed_main.py` | Standard interface — manual input + dataset evaluation |
| `python intelligent_main.py` | Full pipeline with NL query analysis + analytics dashboard |
| `python lightweight_intelligent_main.py` | Headless deployment (no matplotlib/plotly dependencies) |
| `python enhanced_fixed_main.py` | Extended interface with 6 operational modes |

---

## Training Reproduction

### Phase 1: Transformer Fine-tuning

The training script `train.py` (restored from `archived_files/`) contains the complete fine-tuning pipeline. To reproduce:

```bash
python train.py
```

#### Hyperparameter Configuration

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Base model | `xlm-roberta-base` | `config.py` |
| Learning rate | 2×10⁻⁵ | `config.py: LEARNING_RATE` |
| Per-device batch size (train) | 16 | `config.py: BATCH_SIZE` |
| Per-device batch size (eval) | 16 | `config.py: BATCH_SIZE` |
| Training epochs | 5 (best checkpoint at epoch 3) | `config.py: NUM_EPOCHS` |
| Weight decay | 0.01 | `config.py: WEIGHT_DECAY` |
| Warmup steps | 500 | `config.py: WARMUP_STEPS` |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=10⁻⁸) | HuggingFace Trainer default |
| LR scheduler | Linear warmup + linear decay | HuggingFace Trainer default |
| Loss function | Focal Loss (α=1, γ=2) | `train.py: FocalLoss` |
| Max sequence length | 128 tokens | `config.py: MAX_LENGTH` |
| Data split | 90% train / 10% eval, stratified | `random_state=42` |
| Class balancing | Minority upsampling via `sklearn.utils.resample` | `train.py` |
| Mixed precision | FP16 on CUDA, FP32 on CPU | `train.py` |
| Eval strategy | Per-epoch | `eval_strategy="epoch"` |
| Best model selection | By F1 score | `metric_for_best_model="f1"` |
| Random seed | 42 | `seed=42` |

#### Training History (from `trainer_state.json`)

| Epoch | Eval Accuracy | Eval Loss | Training Loss (at step) |
|-------|--------------|-----------|------------------------|
| 1 | 0.68 | 0.582 | 0.776 (step 50) |
| 2 | 0.94 | 0.205 | — |
| 3 | 0.98 | 0.129 | — |

> **⚠️ Critical Note:** These metrics were evaluated on a validation split of only ~10 samples from a ~100-sample fine-tuning dataset. The 98% accuracy is not indicative of real-world generalization. See [Limitations](#limitations).

### Phase 2: Adaptive Lexicon Training

The lexicon system trains via statistical frequency discovery from labeled datasets:

```bash
python auto_trainer.py        # Automated word/bigram discovery from CSV/Excel
python dataset_trainer.py     # Train from configured datasets
python train_menu.py          # Interactive training menu
```

Key training parameters:
- Minimum word occurrences to adopt: 3
- Sentiment ratio threshold: > 0.7 to classify a word
- Minimum text length: 3 characters
- No train/test split (learns from entire input sequentially)

---

## Evaluation

### Running the Evaluation Suite

```bash
python evaluate_model.py
```

This script evaluates **both** systems on the combined sample dataset and reports:
- Per-class **Precision**, **Recall**, and **F1-score** (Positive, Negative, Neutral)
- **Macro-F1** and **Weighted-F1** scores
- Overall **accuracy**
- **Confusion matrix**
- Test set **class distribution** (to assess potential imbalance)

Results are saved to `evaluation_results.txt` for archival.

---

## Dataset Provenance

### Training/Evaluation Data

| File | Domain | Text Column | Label Column | Samples | Pos | Neg | Neu |
|------|--------|------------|-------------|---------|-----|-----|-----|
| `sample_data.csv` | General | `text` | `sentiment` | 10 | 4 (40%) | 3 (30%) | 3 (30%) |
| `sample_movie_reviews.csv` | Film | `review_text` | `sentiment_label` | 20 | 7 (35%) | 7 (35%) | 6 (30%) |
| `sample_product_reviews.csv` | E-commerce | `text` | `sentiment` | 20 | 7 (35%) | 7 (35%) | 6 (30%) |
| `sample_social_media.csv` | Social media | `post` | `emotion` | 20 | 7 (35%) | 7 (35%) | 6 (30%) |
| `my_reviews.csv` | E-commerce | `text` | `sentiment` | 20 | 7 (35%) | 7 (35%) | 6 (30%) |
| `fix_bias_data.csv` | Corrective | `text` | `sentiment` | 20 | 5 (25%) | **10 (50%)** | 5 (25%) |
| **Combined** | **Mixed** | — | — | **110** | **37 (34%)** | **41 (37%)** | **32 (29%)** |

> **Data Quality Notes:**
> - Column schemas are heterogeneous across files — the evaluation script normalizes these automatically
> - `my_reviews.csv` contains partial duplicates of `sample_product_reviews.csv`
> - `fix_bias_data.csv` is intentionally skewed toward negative class (50%) for corrective training
> - All balanced files follow artificial alternating class sequences (pos → neg → neu) and should be shuffled before use
> - Total dataset size (~110 samples) is insufficient for robust neural model training

---

## Model Choice Justification

**XLM-RoBERTa** (`xlm-roberta-base`, 278M parameters) was selected for its strong cross-lingual transfer learning capabilities, supporting 100+ languages. However, this architectural choice carries significant trade-offs for English-only deployment:

| Consideration | XLM-RoBERTa | RoBERTa-base | DeBERTa-v3-base |
|--------------|-------------|-------------|-----------------|
| Parameters | 278M | 125M | 86M |
| Languages | 100+ | English | English |
| Vocab size | 250K | 50K | 128K |
| Inference speed | Slower | Faster | Fastest |
| English-only accuracy | Comparable | Higher | Highest |

**Recommendation:** For future iterations targeting English-only sentiment analysis, `DeBERTa-v3-base` or `RoBERTa-base` are recommended as more parameter-efficient alternatives with stronger English-specific performance.

---

## Limitations

1. **Sarcasm and Irony:** Neither system reliably detects implicit sarcasm (e.g., *"Oh great, another meeting"*). The lexicon system treats individual words at face value; the transformer lacks sufficient training examples of sarcastic text.

2. **Sequence Length Truncation:** The transformer truncates inputs exceeding **128 tokens** (~2-3 sentences). Documents longer than this are silently truncated without windowing or segmentation. This behavior should be made explicit to end users.

3. **Training Data Scale:** The transformer was fine-tuned on approximately **100 samples** — orders of magnitude below the thousands typically required for robust fine-tuning. The reported 98% accuracy reflects memorization of this small set, not generalization.

4. **Lexicon Vocabulary Coverage:** The Phase 2 system initializes with ~123 seed words. Domain-specific vocabulary (e.g., financial jargon, medical terminology) may yield uncertain predictions until corrective feedback expands the lexicon.

5. **Domain Transfer:** A model fine-tuned on product/movie reviews will not reliably classify financial news, clinical notes, or legal text. Cross-domain evaluation has not been conducted.

6. **No Train/Test Separation in Lexicon Training:** The adaptive lexicon system (`auto_trainer.py`, `dataset_trainer.py`) learns from the entire input dataset sequentially without holdout, risking memorization-based overfitting of word frequencies.

7. **Class Label Granularity:** The system outputs only 3 coarse classes (Positive/Negative/Neutral). Fine-grained emotion detection (joy, anger, fear, surprise, disgust) is not supported.

8. **Multilingual Claims vs. Reality:** While XLM-RoBERTa supports 100+ languages, the system has only been tested on English text. Cross-lingual performance is unvalidated.

---

## Project Structure

```
sentiment-analysis/
├── README.md                          # This document
├── evaluate_model.py                  # Scientific evaluation suite (F1, confusion matrix)
│
├── # ─── Entry Points ───
├── start_system.py                    # System launcher / router
├── fixed_main.py                      # Standard interface (manual + dataset)
├── intelligent_main.py                # Full NL analysis pipeline
├── lightweight_intelligent_main.py    # Headless deployment version
├── enhanced_fixed_main.py             # Extended 6-mode interface
│
├── # ─── Training Scripts ───
├── train.py                           # Transformer fine-tuning (restored from archive)
├── test.py                            # Transformer evaluation (restored from archive)
├── train_sentiment.py                 # Baseline transformer trainer
├── auto_trainer.py                    # Lexicon word/bigram discovery
├── dataset_trainer.py                 # Lexicon training from datasets
├── quick_train.py                     # Quick CLI training utility
├── windows_safe_train.py              # Windows-compatible training
├── train_menu.py                      # Interactive training menu
│
├── # ─── Core Modules ───
├── adaptive_sentiment.py              # AdaptiveSentimentAnalyzer (lexicon engine)
├── enhanced_adaptive_learning.py      # Confidence scoring + active learning
├── ensemble_adaptive_learning.py      # Multi-model ensemble voting
├── online_learning_system.py          # Background online learning daemon
├── critical_analysis_system.py        # NL query analysis + reasoning
├── learning_analytics_dashboard.py    # Plotly/Matplotlib analytics
├── robust_dataset_loader.py           # Multi-format data loader
├── config.py                          # Central configuration
├── dataset_config.py                  # Dataset management CLI
├── utils.py                           # Shared utilities
│
├── # ─── Data ───
├── sample_data.csv                    # 10 general samples
├── sample_movie_reviews.csv           # 20 movie review samples
├── sample_product_reviews.csv         # 20 product review samples
├── sample_social_media.csv            # 20 social media samples
├── my_reviews.csv                     # 20 personal review samples
├── fix_bias_data.csv                  # 20 bias correction samples
├── dataset_paths.json                 # Dataset registry
│
├── # ─── Environment ───
├── requirements.txt                   # Version ranges (flexible)
├── requirements_frozen.txt            # Exact pinned versions (reproducible)
├── requirements_minimal.txt           # Minimal dependencies (no torch)
├── environment.yml                    # Conda environment specification
│
├── # ─── Documentation ───
├── docs/                              # Supplementary guides
│   ├── ADAPTIVE_LEARNING_README.md
│   ├── DATASET_ADDRESSES_GUIDE.md
│   ├── PROJECT_STRUCTURE.md
│   ├── QUICK_START_GUIDE.md
│   ├── TRAINING_GUIDE.md
│   └── UPDATED_PROJECT_STRUCTURE.md
├── logs/                              # Development logs
│   ├── BUG_FIX_SUMMARY.md
│   └── MERGE_COMPLETE_SUMMARY.md
│
├── # ─── Model Artifacts (not tracked in git) ───
├── sentiment_model/                   # XLM-RoBERTa weights + checkpoints
├── results/                           # Training run outputs
├── archived_files/                    # All Phase 1 legacy scripts
└── sentiment_knowledge.pkl            # Lexicon system persistent state
```

---

## Environment Reproducibility

### Frozen Dependencies (Recommended)

```bash
pip install -r requirements_frozen.txt
```

This installs exact versions tested with this codebase. Key versions:
- Python 3.9+
- PyTorch 2.4.0
- Transformers 4.56.1
- scikit-learn 1.5.2
- pandas 2.2.3

### Conda Environment

```bash
conda env create -f environment.yml
conda activate sentiment-analysis
```

### System Requirements
- **CPU:** Any modern x86_64 processor
- **GPU:** Optional; CUDA-compatible GPU for transformer training/inference
- **RAM:** ≥8 GB (16 GB recommended for transformer training)
- **Disk:** ~2 GB (including model weights)

---

## License

MIT License. If you utilize this framework in your research, please cite this repository.
