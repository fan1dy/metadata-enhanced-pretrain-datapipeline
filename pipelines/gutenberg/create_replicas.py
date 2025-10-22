"""
Replicate Gutenberg Excerpts with Varying Repetition Patterns

This script takes the excerpts created by create_excerpt.py and replicates them according
to a specified repetition schedule to create training data for studying memorization in
language models.

Process:
1. Loads the token.jsonl and text.jsonl files created by create_excerpt.py
2. Divides the dataset into buckets (default: 500 samples per bucket)
3. Replicates each bucket a different number of times according to the REPETITIONS array:
   - Bucket 1: 1x repetition (samples 0-499)
   - Bucket 2: 2x repetitions (samples 500-999)
   - Bucket 3: 3x repetitions (samples 1000-1499)
   - ... up to 128x repetitions
4. Saves replicated files as rep_1_text.jsonl, rep_2_text.jsonl, etc.
5. Optionally combines all replicated data into one large file

Configuration:
    - REPETITIONS: Array defining how many times each bucket is replicated
    - BUCKET_SIZE: Number of samples per bucket (default: 500)
    - SEQ_LENGTH: Length of each sequence in tokens (default: 8190)

Usage:
    Modify the input_path and output_path in main() function, then run:
    python create_replicas.py
"""
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from tqdm import tqdm
import gc
import logging
import os
import numpy as np

# Constants
FILE_NAMES = {"TOKEN": "token.jsonl", "TEXT": "text.jsonl"}
REPETITIONS = np.array([1,2,3,4,8,16,24,32,48,64,96,128])
BUCKET_SIZE = 500
SEQ_LENGTH = 8191

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('replicate_gutenberg_excerpt.log'),
        logging.StreamHandler()
    ]
)


def load_and_validate_data(data_path: Path, repetitions: np.ndarray, bucket_size: int):
    dataset = load_dataset('json', data_files=str(data_path), split='train')
    assert len(repetitions) * bucket_size <= len(dataset), (
        f"Required {len(repetitions) * bucket_size} samples, "
        f"but dataset only contains {len(dataset)}"
    )
    return dataset


def save_replicated_data(text, token, repetitions: np.ndarray, bucket_size: int, output_path: Path):
    """Save replicated datasets using HuggingFace's efficient methods."""

    # Check for existing files
    completed_reps = set()
    for path in output_path.glob("rep_*_text.jsonl"):
        rep = int(path.stem.split('_')[1])
        completed_reps.add(rep)

    for idx, rep in enumerate(tqdm(repetitions, desc="Processing buckets")):
        if rep in completed_reps:
            logging.info(f"Skipping repetition {rep} - already processed")
            continue

        # Get current slice
        start_idx = idx * bucket_size
        current_slice_text = text.select(range(start_idx, start_idx + bucket_size))
        current_slide_token = token.select(range(start_idx, start_idx + bucket_size)) # token does not require replication
        
        # Create replicated version efficiently
        if rep > 1:
            replicated_slices = [current_slice_text] * rep
            current_slice_text = concatenate_datasets(replicated_slices)
        
        # Save using dataset's built-in method
        output_text_file = str(output_path / f"rep_{rep}_{FILE_NAMES['TEXT']}")
        current_slice_text.to_json(output_text_file)
        current_slide_token.to_json(str(output_path / f"rep_{rep}_{FILE_NAMES['TOKEN']}"))
        
        # logging.info(f"Saved {rep} repetitions ({len(current_slice_text)} samples) to {output_text_file}")

        # Cleanup
        current_slice_text = None
        current_slide_token = None
        gc.collect()


def save_replicated_text_in_one(text, repetitions: np.ndarray, bucket_size: int, seq_length: int, output_path: Path):
    """Save all replicated text data in a single JSON file with each sequence trimmed to 8190 tokens."""

    logging.info("Creating combined replicated dataset...")

    all_replicated_text = []

    for idx, rep in enumerate(tqdm(repetitions, desc="Combining text data")):
        # Get current slice
        start_idx = idx * bucket_size
        current_slice_text = text.select(range(start_idx, start_idx + bucket_size))
        
        # Create replicated version efficiently
        if rep > 1:
            replicated_slices = [current_slice_text] * rep
            current_slice_text = concatenate_datasets(replicated_slices)
        
        # Append to combined dataset
        all_replicated_text.append(current_slice_text)
        
        # Cleanup
        current_slice_text = None
        gc.collect()
    
    # Combine all slices into one dataset
    combined_dataset = concatenate_datasets(all_replicated_text)

    # Save combined dataset
    output_file = str(output_path / f"combined_gutenberg_text_{seq_length * bucket_size * np.sum(repetitions)}.jsonl")
    combined_dataset.to_json(output_file)
    
    logging.info(f"Saved combined replicated text dataset ({len(combined_dataset)} samples) to {output_file}")
    
    # Final cleanup
    all_replicated_text = None
    combined_dataset = None
    gc.collect()


def main():
    input_path = Path('/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_en_8k_mixtral')
    output_path = Path("/iopsstor/scratch/cscs/xyixuan/dataset/gutenberg_apertus_buk167")
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info("Loading dataset...")
    token_seq = load_and_validate_data(input_path / FILE_NAMES['TOKEN'], REPETITIONS, BUCKET_SIZE)
    text_seq = load_and_validate_data(input_path / FILE_NAMES['TEXT'], REPETITIONS, BUCKET_SIZE)

    logging.info("Saving replicated datasets...")
    save_replicated_data(
        text=text_seq,
        token=token_seq,
        repetitions=REPETITIONS,
        bucket_size=BUCKET_SIZE,
        output_path=output_path
    )

    logging.info("Process completed successfully")

if __name__ == "__main__":
    main()