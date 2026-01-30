#!/usr/bin/env python3
"""
Multi-Speaker TTS Training Script for LangCoach

This script fine-tunes the Orpheus-3B TTS model on multiple speaker datasets
from HuggingFace (Jinsaryko series). It supports:
- Multiple speakers with configurable dataset sources
- LoRA-based efficient fine-tuning via Unsloth
- SNAC audio tokenization
- Configurable training parameters

Usage:
    python train_multi_speaker.py --speakers Ceylia Tifa --max_samples 100 --max_steps 60
    python train_multi_speaker.py --config config.yaml
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torchaudio.transforms as T
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments, Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for multi-speaker TTS training."""
    # Model settings
    base_model: str = "unsloth/orpheus-3b-0.1-ft"
    max_seq_length: int = 2048
    load_in_4bit: bool = False

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training settings
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    num_train_epochs: Optional[int] = None
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    logging_steps: int = 1
    seed: int = 3407

    # Data settings
    speakers: List[str] = field(default_factory=lambda: ["Ceylia", "Tifa"])
    dataset_prefix: str = "Jinsaryko"
    max_samples_per_speaker: Optional[int] = None

    # Output settings
    output_dir: str = "outputs"
    model_name_template: str = "orpheus-tts-{speakers}"
    save_merged_16bit: bool = True
    push_to_hub: bool = False
    hub_token: Optional[str] = None


# Token constants for Orpheus model
TOKENIZER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7
AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10


class SNACAudioTokenizer:
    """SNAC-based audio tokenizer for TTS training."""

    def __init__(self, device: str = "cuda"):
        from snac import SNAC
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to(device)
        self.device = device

    def tokenize(self, waveform, sample_rate: int) -> List[int]:
        """Convert audio waveform to SNAC tokens."""
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)

        # Resample to 24kHz if needed
        if sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)

        waveform = waveform.unsqueeze(0).to(self.device)

        # Generate codes from SNAC
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        # Interleave codes from all layers
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + 128266)
            all_codes.append(codes[1][0][2*i].item() + 128266 + 4096)
            all_codes.append(codes[2][0][4*i].item() + 128266 + (2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item() + 128266 + (3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item() + 128266 + (4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item() + 128266 + (5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item() + 128266 + (6*4096))

        return all_codes

    def to_cpu(self):
        """Move SNAC model to CPU to free GPU memory."""
        self.snac_model = self.snac_model.to("cpu")
        self.device = "cpu"

    def to_cuda(self):
        """Move SNAC model to CUDA."""
        self.snac_model = self.snac_model.to("cuda")
        self.device = "cuda"


class MultiSpeakerDatasetBuilder:
    """Builds combined multi-speaker dataset for TTS training."""

    def __init__(self, audio_tokenizer: SNACAudioTokenizer, text_tokenizer):
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer

    def load_speaker_dataset(
        self,
        speaker_name: str,
        dataset_prefix: str = "Jinsaryko",
        max_samples: Optional[int] = None
    ) -> Dataset:
        """Load dataset for a single speaker."""
        dataset_name = f"{dataset_prefix}/{speaker_name}"
        logger.info(f"Loading dataset: {dataset_name}")

        dataset = load_dataset(dataset_name, split="train")

        if max_samples is not None and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")

        # Add source field for multi-speaker support
        def add_source(example):
            example["source"] = speaker_name
            return example

        dataset = dataset.map(add_source)
        logger.info(f"Loaded {len(dataset)} samples for speaker: {speaker_name}")

        return dataset

    def build_combined_dataset(
        self,
        speakers: List[str],
        dataset_prefix: str = "Jinsaryko",
        max_samples_per_speaker: Optional[int] = None
    ) -> Dataset:
        """Build combined dataset from multiple speakers.

        Process each dataset separately to handle different audio formats,
        then concatenate the processed results.
        """
        all_processed = []

        for speaker in speakers:
            ds = self.load_speaker_dataset(
                speaker,
                dataset_prefix,
                max_samples_per_speaker
            )
            # Process each speaker's dataset separately
            processed = self._process_single_speaker_dataset(ds, speaker)
            all_processed.append(processed)

        combined = concatenate_datasets(all_processed)
        logger.info(f"Combined dataset size: {len(combined)} samples")

        return combined

    def _process_single_speaker_dataset(self, dataset: Dataset, speaker: str) -> Dataset:
        """Process a single speaker's dataset: tokenize audio and create input_ids."""
        # Get sample rate from first example
        sample_rate = dataset[0]["audio"]["sampling_rate"]
        logger.info(f"Processing {speaker} - Audio sample rate: {sample_rate}")

        # Tokenize audio
        def add_audio_codes(example):
            codes_list = None
            try:
                audio = example.get("audio")
                if audio and "array" in audio:
                    codes_list = self.audio_tokenizer.tokenize(
                        audio["array"],
                        audio.get("sampling_rate", sample_rate)
                    )
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")
            example["codes_list"] = codes_list
            return example

        logger.info(f"Tokenizing audio for {speaker}...")
        dataset = dataset.map(add_audio_codes, remove_columns=["audio"])

        # Filter out failed tokenizations
        dataset = dataset.filter(lambda x: x["codes_list"] is not None)
        dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)
        logger.info(f"After filtering {speaker}: {len(dataset)} samples")

        # Remove duplicate frames
        def remove_duplicate_frames(example):
            vals = example["codes_list"]
            if len(vals) % 7 != 0:
                vals = vals[:len(vals) // 7 * 7]

            if len(vals) == 0:
                example["codes_list"] = vals
                return example

            result = vals[:7]
            for i in range(7, len(vals), 7):
                current_first = vals[i]
                previous_first = result[-7]
                if current_first != previous_first:
                    result.extend(vals[i:i+7])

            example["codes_list"] = result
            return example

        dataset = dataset.map(remove_duplicate_frames)

        # Create input_ids for training
        def create_input_ids(example):
            text_prompt = f"{example['source']}: {example['text']}"
            text_ids = self.text_tokenizer.encode(text_prompt, add_special_tokens=True)
            text_ids.append(END_OF_TEXT)

            input_ids = (
                [START_OF_HUMAN]
                + text_ids
                + [END_OF_HUMAN]
                + [START_OF_AI]
                + [START_OF_SPEECH]
                + example["codes_list"]
                + [END_OF_SPEECH]
                + [END_OF_AI]
            )
            example["input_ids"] = input_ids
            example["labels"] = input_ids
            example["attention_mask"] = [1] * len(input_ids)
            return example

        logger.info(f"Creating input_ids for {speaker}...")
        dataset = dataset.map(
            create_input_ids,
            remove_columns=["text", "codes_list"]
        )

        # Keep only required columns
        columns_to_keep = ["input_ids", "labels", "attention_mask"]
        columns_to_remove = [
            col for col in dataset.column_names
            if col not in columns_to_keep
        ]
        dataset = dataset.remove_columns(columns_to_remove)

        return dataset

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """Process dataset: tokenize audio and create input_ids.

        Note: This method is kept for backward compatibility but
        build_combined_dataset now handles processing internally.
        """
        # Get sample rate from first example
        sample_rate = dataset[0]["audio"]["sampling_rate"]
        logger.info(f"Audio sample rate: {sample_rate}")

        # Tokenize audio
        def add_audio_codes(example):
            codes_list = None
            try:
                audio = example.get("audio")
                if audio and "array" in audio:
                    codes_list = self.audio_tokenizer.tokenize(
                        audio["array"],
                        audio.get("sampling_rate", sample_rate)
                    )
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")
            example["codes_list"] = codes_list
            return example

        logger.info("Tokenizing audio...")
        dataset = dataset.map(add_audio_codes, remove_columns=["audio"])

        # Filter out failed tokenizations
        dataset = dataset.filter(lambda x: x["codes_list"] is not None)
        dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)
        logger.info(f"After filtering: {len(dataset)} samples")

        # Remove duplicate frames
        def remove_duplicate_frames(example):
            vals = example["codes_list"]
            if len(vals) % 7 != 0:
                # Truncate to valid length
                vals = vals[:len(vals) // 7 * 7]

            if len(vals) == 0:
                example["codes_list"] = vals
                return example

            result = vals[:7]
            for i in range(7, len(vals), 7):
                current_first = vals[i]
                previous_first = result[-7]
                if current_first != previous_first:
                    result.extend(vals[i:i+7])

            example["codes_list"] = result
            return example

        dataset = dataset.map(remove_duplicate_frames)

        # Create input_ids for training
        def create_input_ids(example):
            # Multi-speaker format: "{source}: {text}"
            text_prompt = f"{example['source']}: {example['text']}"

            text_ids = self.text_tokenizer.encode(text_prompt, add_special_tokens=True)
            text_ids.append(END_OF_TEXT)

            example["text_tokens"] = text_ids
            input_ids = (
                [START_OF_HUMAN]
                + example["text_tokens"]
                + [END_OF_HUMAN]
                + [START_OF_AI]
                + [START_OF_SPEECH]
                + example["codes_list"]
                + [END_OF_SPEECH]
                + [END_OF_AI]
            )
            example["input_ids"] = input_ids
            example["labels"] = input_ids
            example["attention_mask"] = [1] * len(input_ids)

            return example

        logger.info("Creating input_ids...")
        dataset = dataset.map(
            create_input_ids,
            remove_columns=["text", "codes_list", "text_tokens"]
        )

        # Keep only required columns
        columns_to_keep = ["input_ids", "labels", "attention_mask"]
        columns_to_remove = [
            col for col in dataset.column_names
            if col not in columns_to_keep
        ]
        dataset = dataset.remove_columns(columns_to_remove)

        return dataset


def load_model_and_tokenizer(config: TrainingConfig):
    """Load base model and tokenizer with LoRA configuration."""
    from unsloth import FastLanguageModel

    logger.info(f"Loading model: {config.base_model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=config.load_in_4bit,
    )

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def train(config: TrainingConfig):
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Multi-Speaker TTS Training")
    logger.info("=" * 60)
    logger.info(f"Speakers: {config.speakers}")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Max steps: {config.max_steps}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Initialize audio tokenizer
    audio_tokenizer = SNACAudioTokenizer(device="cuda")

    # Build dataset (build_combined_dataset now handles processing internally)
    dataset_builder = MultiSpeakerDatasetBuilder(audio_tokenizer, tokenizer)
    dataset = dataset_builder.build_combined_dataset(
        speakers=config.speakers,
        dataset_prefix=config.dataset_prefix,
        max_samples_per_speaker=config.max_samples_per_speaker
    )

    logger.info(f"Final dataset size: {len(dataset)} samples")

    # Free GPU memory from SNAC model
    audio_tokenizer.to_cpu()
    torch.cuda.empty_cache()

    # Setup training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps if config.num_train_epochs is None else -1,
        num_train_epochs=config.num_train_epochs if config.num_train_epochs else 1,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=config.output_dir,
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU: {gpu_stats.name}, Max memory: {max_memory} GB")
    logger.info(f"Reserved memory before training: {start_gpu_memory} GB")

    # Train
    logger.info("Starting training...")
    trainer_stats = trainer.train()

    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    logger.info(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds")
    logger.info(f"Peak reserved memory: {used_memory} GB")

    # Generate model name with speaker names
    speakers_str = "-".join(config.speakers)
    model_name = config.model_name_template.format(speakers=speakers_str)

    # Save model
    save_path = os.path.join(config.output_dir, model_name)

    # Save LoRA adapters
    lora_path = os.path.join(save_path, "lora")
    logger.info(f"Saving LoRA adapters to: {lora_path}")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # Save merged 16bit model if requested
    if config.save_merged_16bit:
        merged_path = os.path.join(save_path, "merged_16bit")
        logger.info(f"Saving merged 16bit model to: {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    # Save training config
    config_path = os.path.join(save_path, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "speakers": config.speakers,
            "base_model": config.base_model,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "max_steps": config.max_steps,
            "learning_rate": config.learning_rate,
            "final_loss": trainer_stats.metrics.get("train_loss"),
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Model saved to: {save_path}")
    logger.info(f"Supported speakers: {config.speakers}")
    logger.info("=" * 60)

    return model, tokenizer, save_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Speaker TTS Training for LangCoach"
    )

    # Speaker configuration
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=["Ceylia", "Tifa"],
        help="List of speaker names (from Jinsaryko datasets)"
    )
    parser.add_argument(
        "--dataset_prefix",
        default="Jinsaryko",
        help="HuggingFace dataset prefix"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per speaker (for testing)"
    )

    # Model configuration
    parser.add_argument(
        "--base_model",
        default="unsloth/orpheus-3b-0.1-ft",
        help="Base model name"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit quantization"
    )

    # LoRA configuration
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha"
    )

    # Training configuration
    parser.add_argument(
        "--max_steps",
        type=int,
        default=60,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides max_steps)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--no_save_merged",
        action="store_true",
        help="Don't save merged 16bit model"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = TrainingConfig(
        speakers=args.speakers,
        dataset_prefix=args.dataset_prefix,
        max_samples_per_speaker=args.max_samples,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.output_dir,
        save_merged_16bit=not args.no_save_merged,
    )

    train(config)


if __name__ == "__main__":
    main()
