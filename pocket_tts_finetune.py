"""
Pocket-TTS Multilingual & Multi-Speaker Fine-Tuning Pipeline
============================================================
Fine-tunes the Kyutai Pocket-TTS (CALM architecture: 24-layer FlowLM + Mimi codec)
on custom African multilingual speech datasets with individual speakers:
  - crestai/waxal_tts
  - crestai/kin_tts
  - crestai/salt_tts
  - crestai/pidjin_tts
  - crestai/wolof_tts
  - crestai/twi_akosua_female_speaker_tts
  - crestai/vo_tts

Pipeline Overview:
  1. Download and preprocess audio to 24kHz WAVs + build JSONL manifests
  2. Train custom multilingual SentencePiece tokenizer (vocab size 4000)
  3. Precompute Mimi neural codec latents for fast GPU throughput
  4. Fine-tune Pocket-TTS teacher model with reset text embeddings
  5. Validate voice-cloning and TTS generation per speaker/language
  6. Export model weights (.safetensors), config (.yaml), and tokenizer (.model)
  7. Push fine-tuned artifacts to Hugging Face Hub
"""

import os
import sys
import gc
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import soundfile as sf
import sentencepiece as spm
from datasets import load_dataset, concatenate_datasets, Audio
from huggingface_hub import login, HfApi

# Ensure pocket-tts repo is in sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
POCKET_TTS_DIR = SCRIPT_DIR if (SCRIPT_DIR / "pocket_tts").exists() else (SCRIPT_DIR / "pocket-tts")
if str(POCKET_TTS_DIR) not in sys.path:
    sys.path.insert(0, str(POCKET_TTS_DIR))

from pocket_tts.models.mimi import build_mimi, MimiModel
from training.args import TrainArgs, dump_args, load_args, save_args
from training.checkpointing import EMA, latest_checkpoint, load_checkpoint, save_checkpoint
from training.dataloader import SubprocessDataLoader, DataLoader
from training.distributed import get_rank, get_world_size, init_distributed, is_torchrun
from training.modules.builders import build_models
from training.train_utils import (
    ProgressLog,
    _compile_models,
    add_file_logging,
    ensure_train_latents,
    git_commit,
    lr_at,
    setup_logging,
    write_samples,
)
from pocket_tts.models.tts_model import TTSModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pocket_tts_finetune")

# ==============================================================================
# 1. Configuration
# ==============================================================================
TARGET_SR = 24000  # Mimi neural audio codec standard sample rate
VOCAB_SIZE = 4000  # Matches Pocket-TTS lookup table n_bins
DEFAULT_DATA_DIR = Path("data/custom_dataset")
DEFAULT_AUDIO_DIR = Path("data/custom_dataset/audio")
DEFAULT_OUTPUT_DIR = Path("runs/finetune_custom_languages")
DEFAULT_CONFIG_PATH = POCKET_TTS_DIR / "training/configs/finetune_custom_languages.yaml"

DATASETS_TO_LOAD = [
    {"repo": "crestai/waxal_tts", "lang": "waxal"},
    {"repo": "crestai/kin_tts", "lang": "kinyarwanda"},
    {"repo": "crestai/salt_tts", "lang": "salt"},
    {"repo": "crestai/pidjin_tts", "lang": "pidgin"},
    {"repo": "crestai/wolof_tts", "lang": "wolof"},
    {"repo": "crestai/twi_akosua_female_speaker_tts", "lang": "twi"},
    {"repo": "crestai/vo_tts", "lang": "vo"},
]

# Hugging Face deployment configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_NAME = "crestai/pocket_tts_multilingual_finetuned"


# ==============================================================================
# 2. Dataset Preparation & Audio Extraction
# ==============================================================================
def prepare_custom_datasets(
    data_dir: Path = DEFAULT_DATA_DIR,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    val_split_ratio: float = 0.02,
    max_duration_sec: float = 30.0,
    min_duration_sec: float = 1.0,
    streaming: bool = True,
    max_samples_per_dataset: Optional[int] = None,
) -> tuple[Path, Path]:
    """
    Streams and downloads HF datasets on the fly, normalizes audio to 24kHz mono WAV,
    and creates train.jsonl, valid.jsonl manifests, and speaker reference catalog.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl_path = data_dir / "train_aligned.jsonl"
    valid_jsonl_path = data_dir / "valid_aligned.jsonl"

    if train_jsonl_path.exists() and valid_jsonl_path.exists():
        logger.info(f"Manifests already exist at {train_jsonl_path} and {valid_jsonl_path}. Skipping extraction.")
        return train_jsonl_path, valid_jsonl_path

    logger.info(f"Starting dataset preprocessing (streaming={streaming})...")
    all_records = []

    for d_info in DATASETS_TO_LOAD:
        repo_name = d_info["repo"]
        lang = d_info["lang"]
        logger.info(f"Streaming dataset: {repo_name} ({lang})")
        try:
            ds = load_dataset(repo_name, streaming=streaming)
            split = ds["train"] if "train" in ds else list(ds.values())[0]
            # Resample audio column to 24 kHz on the fly
            split = split.cast_column("audio", Audio(sampling_rate=TARGET_SR))

            count = 0
            for idx, item in enumerate(split):
                if max_samples_per_dataset and count >= max_samples_per_dataset:
                    break
                audio_dict = item.get("audio")
                if not audio_dict:
                    continue

                audio_array = audio_dict["array"]
                sr = audio_dict["sampling_rate"]
                text = item.get("text", "").strip()
                speaker_id = item.get("speaker_id", f"{lang}_speaker")

                if not text:
                    continue

                duration = len(audio_array) / sr
                if duration < min_duration_sec or duration > max_duration_sec:
                    continue

                clean_lang = lang.replace(" ", "_")
                clean_speaker = str(speaker_id).replace(" ", "_").replace("/", "_")
                filename = f"{clean_lang}_{clean_speaker}_{idx:06d}.wav"
                file_path = audio_dir / filename

                if not file_path.exists():
                    sf.write(str(file_path), audio_array, sr, subtype="PCM_16")

                record = {
                    "path": str(file_path.resolve()),
                    "start": 0.0,
                    "duration": float(duration),
                    "transcript": text,
                    "speaker": str(speaker_id),
                    "language": lang,
                }
                all_records.append(record)
                count += 1

                if count % 1000 == 0:
                    logger.info(f"[{lang}] Streamed {count} utterances...")

        except Exception as e:
            logger.error(f"Error loading {repo_name}: {e}")

    logger.info(f"Total processed utterances: {len(all_records)}")
    if not all_records:
        raise RuntimeError("No valid audio records extracted from datasets.")

    # Save Speaker Reference Samples Catalog for Testing
    speaker_samples_dir = data_dir / "speaker_samples"
    speaker_samples_dir.mkdir(parents=True, exist_ok=True)
    speaker_catalog = {}

    for r in all_records:
        key = f"{r['language']}_{r['speaker']}"
        if key not in speaker_catalog and 3.0 <= r["duration"] <= 12.0:
            dest_wav = speaker_samples_dir / f"{key}_ref.wav"
            import shutil
            shutil.copyfile(r["path"], dest_wav)
            speaker_catalog[key] = {
                "language": r["language"],
                "speaker_id": r["speaker"],
                "reference_audio_path": str(dest_wav.resolve()),
                "sample_transcript": r["transcript"],
            }

    catalog_path = data_dir / "speakers_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(speaker_catalog, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved speaker reference catalog ({len(speaker_catalog)} speakers) to {catalog_path}")

    # Train / Validation Split
    import random
    random.seed(42)
    random.shuffle(all_records)

    n_val = max(1, int(len(all_records) * val_split_ratio))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(valid_jsonl_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(train_records)} train records to {train_jsonl_path}")
    logger.info(f"Wrote {len(val_records)} validation records to {valid_jsonl_path}")

    return train_jsonl_path, valid_jsonl_path


# ==============================================================================
# 3. Train Multilingual SentencePiece Tokenizer
# ==============================================================================
def train_multilingual_tokenizer(
    manifest_paths: List[Path],
    output_prefix: Path = DEFAULT_DATA_DIR / "tokenizer",
    vocab_size: int = VOCAB_SIZE,
) -> Path:
    """
    Trains a SentencePiece BPE tokenizer on all transcripts from custom datasets.
    """
    model_path = output_prefix.with_suffix(".model")
    if model_path.exists():
        logger.info(f"SentencePiece tokenizer already exists at {model_path}. Skipping.")
        return model_path

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    temp_corpus = output_prefix.parent / "corpus.txt"

    total_lines = 0
    with open(temp_corpus, "w", encoding="utf-8") as f_out:
        for path in manifest_paths:
            with open(path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    data = json.loads(line)
                    transcript = data.get("transcript", "").strip()
                    if transcript:
                        f_out.write(transcript + "\n")
                        total_lines += 1

    logger.info(f"Training SentencePiece tokenizer on {total_lines} lines...")
    spm.SentencePieceTrainer.train(
        input=str(temp_corpus),
        model_prefix=str(output_prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=-1,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    temp_corpus.unlink(missing_ok=True)
    logger.info(f"Trained SentencePiece model successfully: {model_path}")
    return model_path


# ==============================================================================
# 4. Precompute Mimi Latents
# ==============================================================================
def precompute_latents_for_dataset(
    manifest_path: Path,
    device: torch.device,
    batch_size: int = 16,
):
    """
    Precomputes Mimi codec representations (.safetensors) for high-speed training.
    """
    meta_path = manifest_path.with_suffix(".meta.json")
    if meta_path.exists():
        logger.info(f"Latents metadata {meta_path} already exists. Skipping precomputation.")
        return

    logger.info(f"Precomputing Mimi latents for {manifest_path}...")
    from training.scripts.precompute_latents import run_precompute
    run_precompute(
        manifest=manifest_path,
        device=str(device),
        batch_size=batch_size,
    )
    logger.info("Latents precomputation complete.")


# ==============================================================================
# 5. Training Loop
# ==============================================================================
def run_training(config_path: str):
    """
    Loads configuration and launches the Pocket-TTS continuous flow training loop.
    """
    setup_logging()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)

    args = load_args(config_path)
    device = init_distributed()
    rank, world_size = get_rank(), get_world_size()
    torch.manual_seed(args.seed + rank)
    run_dir = args.run_dir
    log_path = add_file_logging(run_dir, rank)
    progress = ProgressLog(run_dir / "progress.jsonl", enabled=rank == 0)

    if rank == 0:
        logger.info(f"Starting Pocket-TTS training on device: {device} (World Size: {world_size})")
        logger.info(f"Run directory: {run_dir.resolve()}")
        save_args(args, run_dir / "args.yaml")

    # Build trainable models (FlowLM + Mimi)
    model, mimi, _config = build_models(args)
    model.to(device)
    mimi.to(device)

    # Ensure latents are computed
    ensure_train_latents(args, mimi, device, rank, world_size)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"FlowLM trainable parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=args.optim.betas,
        eps=args.optim.eps,
        weight_decay=args.optim.weight_decay,
        fused=(device.type == "cuda"),
    )
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    start_step = 0
    ckpt = latest_checkpoint(run_dir)
    if ckpt is not None:
        start_step = load_checkpoint(ckpt, model, optimizer, ema)
        logger.info(f"Resumed checkpoint from step {start_step}")

    wrapped: torch.nn.Module = model
    if is_torchrun():
        from torch.nn.parallel import DistributedDataParallel as DDP
        wrapped = DDP(model, device_ids=[device.index], find_unused_parameters=not args.compile)

    sentence_piece = model.flow_lm.conditioner.tokenizer.sp
    train_loader = iter(
        SubprocessDataLoader(
            args.data.train_jsonl,
            sentence_piece,
            args.batch_size,
            mimi.sample_rate,
            mimi.frame_rate,
            args.data.max_duration_sec,
            args.data.max_voice_prompt_sec,
            rank,
            world_size,
            seed=args.seed + start_step,
            shuffle=args.data.shuffle,
            num_procs=args.data.loader_procs,
        )
    )

    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    )
    model.train()

    logger.info(f"Beginning training loop up to step {args.max_steps}...")
    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args.optim)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            batch = next(train_loader)
            with autocast:
                loss, metrics = wrapped(
                    audio=batch.audio.to(device, non_blocking=True),
                    num_audio_frames=batch.num_audio_frames.to(device, non_blocking=True),
                    text_tokens=[t.to(device, non_blocking=True) for t in batch.text_tokens],
                    voice_audio=batch.voice_audio.to(device, non_blocking=True),
                    num_voice_prompt_frames=batch.num_voice_prompt_frames.to(device, non_blocking=True),
                    tail_latents=batch.tail_latents.to(device, non_blocking=True) if batch.tail_latents is not None else None,
                    prompt_latents=batch.prompt_latents.to(device, non_blocking=True) if batch.prompt_latents is not None else None,
                )
                loss_scaled = loss / args.grad_accum_steps

            loss_scaled.backward()
            accum_loss += float(loss.detach())

        if args.optim.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.max_norm)

        optimizer.step()
        if ema is not None:
            ema.step()

        if rank == 0 and (step % args.log_freq == 0 or step < 10):
            logger.info(f"Step {step:06d}/{args.max_steps:06d} | Loss: {accum_loss:.4f} | LR: {lr:.2e}")

        # Checkpointing
        if rank == 0 and (step + 1) % args.ckpt_freq == 0:
            save_checkpoint(
                run_dir,
                step + 1,
                model,
                optimizer,
                ema,
                args.num_ckpt_keep,
            )
            logger.info(f"Saved checkpoint at step {step + 1}")

    if rank == 0:
        # Final save
        save_checkpoint(run_dir, args.max_steps, model, optimizer, ema, args.num_ckpt_keep)
        logger.info(f"Training completed successfully. Weights saved in {run_dir.resolve()}")


# ==============================================================================
# 6. Push to Hugging Face Hub
# ==============================================================================
def push_to_huggingface(
    run_dir: Path,
    repo_name: str,
    hf_token: Optional[str] = None,
):
    """
    Pushes fine-tuned model weights (safetensors), config yaml, and tokenizer to Hugging Face Hub.
    """
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        logger.warning("No Hugging Face token provided. Skipping push to Hub.")
        return

    login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_name, exist_ok=True)

    files_to_upload = [
        run_dir / "model.safetensors",
        run_dir / "args.yaml",
        DEFAULT_DATA_DIR / "tokenizer.model",
        DEFAULT_DATA_DIR / "tokenizer.vocab",
    ]

    for file_p in files_to_upload:
        if file_p.exists():
            logger.info(f"Uploading {file_p.name} to {repo_name}...")
            api.upload_file(
                path_or_fileobj=str(file_p),
                path_in_repo=file_p.name,
                repo_id=repo_name,
            )
    logger.info(f"Model successfully published to https://huggingface.co/{repo_name}")


# ==============================================================================
# 7. Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Pocket-TTS Multilingual Fine-Tuning")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip dataset extraction")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (download full dataset archives locally)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset (for rapid testing)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training step")
    parser.add_argument("--push-to-hub", action="store_true", help="Push trained model to Hugging Face")
    parser.add_argument("--hf-repo", type=str, default=HF_REPO_NAME, help="HF repository name")
    args = parser.parse_args()

    # Step 1: Prepare Datasets
    if not args.skip_data_prep:
        train_jsonl, valid_jsonl = prepare_custom_datasets(
            streaming=(not args.no_stream),
            max_samples_per_dataset=args.max_samples,
        )
        # Step 2: Fit Tokenizer
        tokenizer_model = train_multilingual_tokenizer([train_jsonl, valid_jsonl])
        logger.info(f"Datasets and tokenizer ready: {tokenizer_model}")

    # Step 3: Run Training
    if not args.skip_train:
        run_training(args.config)

    # Step 4: Push to Hub
    if args.push_to_hub:
        push_to_huggingface(DEFAULT_OUTPUT_DIR, args.hf_repo, HF_TOKEN)


if __name__ == "__main__":
    main()
