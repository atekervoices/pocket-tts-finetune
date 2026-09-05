# Pocket-TTS Multilingual & Multi-Speaker Fine-Tuning Guide

This repository contains the complete pipeline for fine-tuning **Pocket-TTS** (Kyutai's CALM continuous flow matching architecture: 24-layer FlowLM + Mimi neural audio codec) on custom African language speech datasets with individual speaker profiles.

---

## 🎯 Supported Datasets & Languages

The pipeline automatically loads and trains on:
- **Waxal**: `crestai/waxal_tts`
- **Kinyarwanda**: `crestai/kin_tts`
- **Salt / Luganda**: `crestai/salt_tts`
- **Nigerian Pidgin**: `crestai/pidjin_tts`
- **Wolof**: `crestai/wolof_tts`
- **Twi**: `crestai/twi_akosua_female_speaker_tts`
- **Vo / Ewe**: `crestai/vo_tts`

---

## 🚀 Quickstart (GPU Machine)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/atekervoices/pocket-tts-finetune.git
cd pocket-tts-finetune

# Install required dependencies
pip install -r requirements.txt
```

*(Or using `uv`):*
```bash
uv sync
```

---

### 2. Configure Your Hugging Face Token

To automatically download datasets and push your fine-tuned model to Hugging Face:

#### Option A: Terminal Environment Variable (Recommended)
```bash
export HF_TOKEN="hf_your_actual_token_here"
```
*(On Windows PowerShell):*
```powershell
$env:HF_TOKEN="hf_your_actual_token_here"
```

#### Option B: In `pocket_tts_finetune.py`
Open `pocket_tts_finetune.py` and set:
```python
HF_TOKEN = "hf_your_actual_token_here"
HF_REPO_NAME = "crestai/pocket_tts_multilingual_finetuned"
```

---

### 3. Launch Full Fine-Tuning Pipeline

Run the automated script to execute data preparation, tokenization, Mimi latent caching, training, and HF upload:

```bash
python pocket_tts_finetune.py --push-to-hub
```

#### For Multi-GPU Machines (e.g. 2x, 4x, or 8x GPUs):
```bash
torchrun --nproc-per-node 4 pocket_tts_finetune.py --push-to-hub
```

---

## 📓 Interactive Training via Jupyter / Colab

If you prefer interactive execution, open:
[`pocket_tts_train_notebook.ipynb`](pocket_tts_train_notebook.ipynb)

The notebook includes step-by-step cells for:
1. GPU verification
2. Dataset downloading and audio processing
3. Multilingual SentencePiece tokenizer training (vocab size 4000)
4. Mimi codec latent precomputation
5. Training loop with live step and loss monitoring
6. Voice-cloning and test sentence generation across all languages
7. Hugging Face hub publishing

---

## ⚙️ Configuration & Hyperparameters

The training configuration is defined in [`training/configs/finetune_custom_languages.yaml`](training/configs/finetune_custom_languages.yaml):

| Parameter | Default Value | Description |
|---|---|---|
| `model_config` | `english_2026-04_24l.yaml` | 24-layer teacher backbone |
| `reset_text_embedding` | `true` | Resets lookup table for new SentencePiece tokenizer |
| `batch_size` | `32` | Per-device batch size |
| `grad_accum_steps` | `2` | Effective batch size = 64 |
| `lr` | `2e-4` | Learning rate with constant scheduler |
| `warmup_steps` | `1000` | Warmup steps |
| `max_steps` | `50000` | Max training iterations |
| `ema_decay` | `0.999` | Exponential moving average of weights |
| `ckpt_freq` | `2500` | Checkpoint saving frequency |

---

## 🎙️ Testing Generated Speech & Voice Cloning

Once trained, test your checkpoint directly using the Python API:

```python
from pocket_tts.models.tts_model import TTSModel

# Load model from fine-tuned checkpoint
model = TTSModel.load_model(
    config="training/configs/finetune_custom_languages.yaml",
    checkpoint="runs/finetune_custom_languages/model.safetensors",
)

# Set voice reference prompt from a speaker WAV
state = model.get_state_for_audio_prompt("data/custom_dataset/audio/sample_speaker.wav")

# Generate speech
audio = model.generate_audio(state, "Jërëjëf ci sa liggéey bu rafet bi.")
```

---

## 📦 Published Model Structure on Hugging Face

After training completes with `--push-to-hub`, your repository will contain:
- `model.safetensors`: Trained FlowLM continuous flow weights (EMA averaged)
- `tokenizer.model`: Custom multilingual SentencePiece tokenizer
- `args.yaml`: Complete reproduction configuration
