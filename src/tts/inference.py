#!/usr/bin/env python3
"""
Multi-Speaker TTS Inference Script for LangCoach

This script provides inference capabilities for the fine-tuned multi-speaker
Orpheus TTS model. It supports:
- Multiple speaker selection at inference time
- Batch inference
- Audio output in various formats

Usage:
    # Basic usage
    python inference.py --model_path outputs/orpheus-tts-Ceylia-Tifa --speaker Ceylia --text "Hello world"
    python inference.py --model_path outputs/orpheus-tts-Ceylia-Tifa --speaker Tifa --text "How are you?"
    
    # With 4bit quantization (saves ~75%% memory, 2-3x faster)
    python inference.py --model_path outputs/orpheus-tts-Ceylia-Tifa --speaker Ceylia --text "Hello world" --load_in_4bit
    
    # With 8bit quantization (saves ~50%% memory, 1.5-2x faster)
    python inference.py --model_path outputs/orpheus-tts-Ceylia-Tifa --speaker Tifa --text "How are you?" --load_in_8bit
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for TTS inference."""
    model_path: str
    use_lora: bool = True  # Use LoRA adapters or merged model
    device: str = "cuda"

    # Quantization settings
    load_in_4bit: bool = False  # Use 4bit quantization for faster inference and less memory
    load_in_8bit: bool = False  # Use 8bit quantization (balance between speed and quality)

    # Generation parameters - optimized for speed
    max_new_tokens: int = 600  # Reduced from 1200 for faster generation (~10s audio max)
    temperature: float = 0.4  # Lower temp for faster, more deterministic generation
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Output settings
    sample_rate: int = 24000


class MultiSpeakerTTSInference:
    """Multi-speaker TTS inference engine."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        self.supported_speakers = []
        self.training_config = None  # Store training config for base model info

        # Load training config first to get base model info if needed
        self._load_speaker_info()
        self._load_model()
        self._load_snac()

    def _load_model(self):
        """Load the fine-tuned model with dynamic LoRA merge and quantization."""
        from unsloth import FastLanguageModel
        from peft import PeftModel

        model_path = self.config.model_path

        # Check if we should use LoRA with dynamic merge
        if self.config.use_lora:
            lora_path = os.path.join(model_path, "lora")
            if os.path.exists(lora_path):
                # Dynamic merge: Load base model + LoRA adapter
                # Get base model from training config
                if self.training_config:
                    base_model = self.training_config.get("base_model", "unsloth/orpheus-3b-0.1-ft")
                else:
                    # Try to read from adapter_config.json
                    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
                    if os.path.exists(adapter_config_path):
                        with open(adapter_config_path, "r") as f:
                            adapter_config = json.load(f)
                            base_model = adapter_config.get("base_model_name_or_path", "unsloth/orpheus-3b-0.1-ft")
                    else:
                        base_model = "unsloth/orpheus-3b-0.1-ft"

                logger.info(f"Loading base model: {base_model}")
                logger.info(f"Loading LoRA adapter from: {lora_path}")
                logger.info(f"Quantization: 4bit={self.config.load_in_4bit}, 8bit={self.config.load_in_8bit}")

                # Load base model with quantization
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                )

                # Load LoRA adapter (dynamic merge happens automatically)
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                
            else:
                logger.warning(f"LoRA path not found: {lora_path}, trying merged model or direct path")
                # Fallback to merged model or direct path
                merged_path = os.path.join(model_path, "merged_16bit")
                if os.path.exists(merged_path):
                    model_path = merged_path
                
                logger.info(f"Loading model from: {model_path}")
                logger.info(f"Quantization: 4bit={self.config.load_in_4bit}, 8bit={self.config.load_in_8bit}")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=self.config.load_in_4bit,
                    load_in_8bit=self.config.load_in_8bit,
                )
        else:
            # Use merged model
            merged_path = os.path.join(model_path, "merged_16bit")
            if os.path.exists(merged_path):
                model_path = merged_path
            
            logger.info(f"Loading merged model from: {model_path}")
            logger.info(f"Quantization: 4bit={self.config.load_in_4bit}, 8bit={self.config.load_in_8bit}")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
            )

        FastLanguageModel.for_inference(self.model)
        quant_info = f"4bit={self.config.load_in_4bit}, 8bit={self.config.load_in_8bit}"
        logger.info(f"Model loaded and set to inference mode (Quantization: {quant_info})")

    def _load_snac(self):
        """Load SNAC model for audio decoding."""
        from snac import SNAC

        logger.info("Loading SNAC model...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        # Move SNAC to GPU for faster decoding
        self.snac_model = self.snac_model.to(self.config.device)
        self.snac_model.eval()
        logger.info(f"SNAC model loaded on {self.config.device}")

    def _load_speaker_info(self):
        """Load supported speaker information from training config."""
        config_path = os.path.join(self.config.model_path, "training_config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.training_config = json.load(f)
                self.supported_speakers = self.training_config.get("speakers", [])
                logger.info(f"Supported speakers: {self.supported_speakers}")
        else:
            logger.warning("Training config not found, speaker list unknown")
            self.training_config = None

    def get_supported_speakers(self) -> List[str]:
        """Return list of supported speakers."""
        return self.supported_speakers

    def synthesize(
        self,
        text: str,
        speaker: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> np.ndarray:
        """
        Synthesize speech for given text and speaker.

        Args:
            text: Text to synthesize
            speaker: Speaker name (must be one of supported speakers)
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            top_p: Override default top_p

        Returns:
            Audio waveform as numpy array (24kHz)
        """
        if self.supported_speakers and speaker not in self.supported_speakers:
            logger.warning(
                f"Speaker '{speaker}' not in supported list: {self.supported_speakers}"
            )

        # Format prompt with speaker prefix
        prompt = f"{speaker}: {text}"
        logger.info(f"Generating speech for: {prompt[:50]}...")

        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Add special tokens
        start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # EOT, EOH

        modified_input_ids = torch.cat(
            [start_token, input_ids, end_tokens], dim=1
        )

        # Prepare attention mask
        attention_mask = torch.ones_like(modified_input_ids)

        # Move to device
        input_ids = modified_input_ids.to(self.config.device)
        attention_mask = attention_mask.to(self.config.device)
        
        # Dynamic max_new_tokens based on text length
        # Roughly 7 tokens per character for audio, with minimum of 200
        text_length = len(text)
        estimated_tokens = max(200, min(text_length * 7, max_new_tokens or self.config.max_new_tokens))

        # Generate with optimized settings
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=estimated_tokens,
                do_sample=True,
                temperature=temperature or self.config.temperature,
                top_p=top_p or self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Process generated tokens
        audio = self._decode_audio(generated_ids)

        return audio

    def _decode_audio(self, generated_ids: torch.Tensor) -> np.ndarray:
        """Decode generated token IDs to audio waveform."""
        token_to_find = 128257  # Start of speech marker
        token_to_remove = 128258  # End of speech marker

        # Find start of speech tokens
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
        else:
            cropped_tensor = generated_ids

        # Remove end tokens
        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        # Convert to code lists
        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t.item() - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        # Decode with SNAC
        audio_samples = []
        for code_list in code_lists:
            if len(code_list) == 0:
                continue
            audio = self._redistribute_codes(code_list)
            audio_samples.append(audio)

        if audio_samples:
            return audio_samples[0].detach().squeeze().cpu().numpy()
        else:
            return np.zeros(24000)  # Return 1 second of silence if failed

    def _redistribute_codes(self, code_list: List[int]) -> torch.Tensor:
        """Redistribute interleaved codes to SNAC layers and decode."""
        layer_1 = []
        layer_2 = []
        layer_3 = []

        for i in range((len(code_list) + 1) // 7):
            layer_1.append(code_list[7 * i])
            layer_2.append(code_list[7 * i + 1] - 4096)
            layer_3.append(code_list[7 * i + 2] - (2 * 4096))
            layer_3.append(code_list[7 * i + 3] - (3 * 4096))
            layer_2.append(code_list[7 * i + 4] - (4 * 4096))
            layer_3.append(code_list[7 * i + 5] - (5 * 4096))
            layer_3.append(code_list[7 * i + 6] - (6 * 4096))

        codes = [
            torch.tensor(layer_1, device=self.config.device).unsqueeze(0),
            torch.tensor(layer_2, device=self.config.device).unsqueeze(0),
            torch.tensor(layer_3, device=self.config.device).unsqueeze(0),
        ]

        with torch.no_grad():
            audio_hat = self.snac_model.decode(codes)
        return audio_hat

    def synthesize_batch(
        self,
        texts: List[str],
        speaker: str,
    ) -> List[np.ndarray]:
        """
        Synthesize speech for multiple texts.

        Args:
            texts: List of texts to synthesize
            speaker: Speaker name

        Returns:
            List of audio waveforms
        """
        results = []
        for text in texts:
            audio = self.synthesize(text, speaker)
            results.append(audio)
        return results

    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None
    ):
        """Save audio to file."""
        import soundfile as sf

        sr = sample_rate or self.config.sample_rate
        sf.write(output_path, audio, sr)
        logger.info(f"Audio saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Speaker TTS Inference for LangCoach"
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--speaker",
        required=True,
        help="Speaker name for synthesis"
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--use_merged",
        action="store_true",
        help="Use merged model instead of LoRA"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model with 4bit quantization (saves ~75%% memory, 2-3x faster)"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model with 8bit quantization (saves ~50%% memory, 1.5-2x faster)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Generation temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--list_speakers",
        action="store_true",
        help="List supported speakers and exit"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate quantization settings
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Cannot use both 4bit and 8bit quantization. Choose one.")

    config = InferenceConfig(
        model_path=args.model_path,
        use_lora=not args.use_merged,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    tts = MultiSpeakerTTSInference(config)

    if args.list_speakers:
        print("Supported speakers:")
        for speaker in tts.get_supported_speakers():
            print(f"  - {speaker}")
        return

    # Synthesize
    audio = tts.synthesize(args.text, args.speaker)

    # Save output
    tts.save_audio(audio, args.output)

    print(f"Generated audio for speaker '{args.speaker}'")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
