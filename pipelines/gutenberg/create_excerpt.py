"""
Create Gutenberg Excerpts Dataset

This script creates the initial Gutenberg dataset by extracting and tokenizing excerpts
from Project Gutenberg books.

Process:
1. Loads the full Project Gutenberg dataset (manu/project_gutenberg)
2. Filters books by removing duplicates and keeping only books with sufficient length
3. Extracts text excerpts from each book between specified character positions
4. Tokenizes the excerpts using a specified tokenizer
5. Randomly selects a fixed number of tokens from each excerpt with a random offset
6. Validates that each excerpt has exactly the expected number of tokens
7. Saves two output files:
   - token.jsonl: contains the tokenized sequences (input_ids)
   - text.jsonl: contains the detokenized text versions

Usage:
    python create_excerpt.py --tokenizer <tokenizer_name> --num-articles 10000 \\
                            --num-tokens 8190 --output-dir <output_path>
"""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import random
from collections import Counter
import os
import pathlib
from functools import partial
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLUMN_NAMES = {
    "TEXT": "text",
    "ID": "id",
    "INPUT_IDS": "input_ids",
    "SELECTED_TOKENS": "selected_tokens",
    "DETOKENIZED_TEXTS": "detokenized_texts",
    "SEQ_LENGTH": "seq_length",
}

FILE_NAMES = {"TOKEN": "token.jsonl", "TEXT": "text.jsonl"}

def create_tokenize_fn(_tokenizer):
    """Create a partial tokenizer function with fixed parameters."""
    return partial(
        _tokenizer, truncation=False, padding=False, add_special_tokens=False
    )


def batch_tokenize_gutenberg(batch, tokenize_fn, char_pos_start, char_pos_end):
    """
    Tokenize sequences from a batch of articles between specified character positions.

    Args:
        batch (dict): Batch of data containing the 'text' field.
        tokenize_fn (function): Tokenization function.
        char_pos_start (int, optional): Starting character position. Defaults to 10_000.
        char_pos_end (int, optional): Ending character position. Defaults to 70_000.

    Returns:
        dict: Dictionary containing input_ids and sequence lengths.
    """
    input_ids_list = []
    seq_length_list = []

    for sequence in batch[COLUMN_NAMES["TEXT"]]:
        input_ids = tokenize_fn(text=sequence[char_pos_start:char_pos_end]).input_ids
        input_ids_list.append(input_ids)
        seq_length_list.append(len(input_ids))

    return {
        COLUMN_NAMES["INPUT_IDS"]: input_ids_list,
        COLUMN_NAMES["SEQ_LENGTH"]: seq_length_list,
    }


def select_tokens_from_random_offset(batch, _tokenizer, num_tokens, seed=42):
    """
    Select a sequence of tokens with random offset from each batch, detokenize them, and return the results.

    Args:
        batch (dict): Batch of data containing 'input_ids'.
        _tokenizer (AutoTokenizer): The tokenizer used for detokenization.
        num_tokens (int): Number of tokens to extract.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary containing selected tokens and detokenized texts.
    """
    # Create a separate random number generator for reproducibility
    rng = random.Random(seed)

    selected_tokens = []
    detokenized_texts = []

    for input_ids in batch[COLUMN_NAMES["INPUT_IDS"]]:
        offset = rng.randint(0, len(input_ids) - num_tokens)
        selected_ids = input_ids[offset : offset + num_tokens]
        selected_tokens.append(selected_ids)
        detokenized_texts.append(_tokenizer.decode(selected_ids))

    return {
        COLUMN_NAMES["SELECTED_TOKENS"]: selected_tokens,
        COLUMN_NAMES["DETOKENIZED_TEXTS"]: detokenized_texts,
    }


def remove_duplicate_ids(example, seen_ids):
    """Helper function to filter out duplicates based on ID."""
    if example[COLUMN_NAMES["ID"]] in seen_ids:
        return False
    seen_ids.add(example[COLUMN_NAMES["ID"]])
    return True


def verify_num_token(example, tokenize_fn, expected_num_tokens):
    """Helper function to only keep examples with the expected number of tokens."""
    # Selected_tokens is a list of tokens, not the length

    return len(tokenize_fn(example[COLUMN_NAMES["DETOKENIZED_TEXTS"]]).input_ids) == expected_num_tokens


def save_tokenized_datasets(dataset, output_dir: str):
    """Save both tokenized and detokenized versions of a dataset to JSONL files."""
    # Validate inputs
    if COLUMN_NAMES["SELECTED_TOKENS"] not in dataset.column_names:
        raise ValueError(
            f"Token column '{COLUMN_NAMES['SELECTED_TOKENS']}' not found in dataset"
        )
    if COLUMN_NAMES["DETOKENIZED_TEXTS"] not in dataset.column_names:
        raise ValueError(
            f"Text column '{COLUMN_NAMES['DETOKENIZED_TEXTS']}' not found in dataset"
        )

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the tokenized version
    dataset.select_columns(COLUMN_NAMES["SELECTED_TOKENS"]).rename_column(
        COLUMN_NAMES["SELECTED_TOKENS"], COLUMN_NAMES["INPUT_IDS"]
    ).to_json(output_dir / FILE_NAMES["TOKEN"])

    # Save the detokenized version
    dataset.select_columns(COLUMN_NAMES["DETOKENIZED_TEXTS"]).rename_column(
        COLUMN_NAMES["DETOKENIZED_TEXTS"], COLUMN_NAMES["TEXT"]
    ).to_json(output_dir / FILE_NAMES["TEXT"])


def main(args):
    # Use all available cpu processors
    num_proc = os.cpu_count()

    # Load the dataset
    ds = load_dataset("manu/project_gutenberg", split="en", cache_dir="/capstor/scratch/cscs/xyixuan/gutenberg")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = 200_000
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create a partial tokenizer function with fixed parameters
    tokenize_fn = create_tokenize_fn(tokenizer)

    # Filter and select the subset of articles
    seen_ids = set()
    subset = (
        ds.filter(
            lambda x: remove_duplicate_ids(x, seen_ids),
            desc="De-duplicating by IDs"
        )
        .filter(
            lambda example: len(example[COLUMN_NAMES["TEXT"]]) >= args.char_pos_end,
            num_proc=num_proc,
            desc="Filtering by text length"
        )
    )

    # Verify no duplicates
    id_counts = Counter(subset["id"])
    assert len(id_counts) == len(subset), "There are duplicate IDs in the dataset"

    # Tokenize the selected subset of articles
    gutenberg = subset.map(
        batch_tokenize_gutenberg,
        batched=True,
        desc="Tokenizing Gutenberg English articles",
        num_proc=num_proc,
        fn_kwargs={
            "tokenize_fn": tokenize_fn,
            "char_pos_start": args.char_pos_start,
            "char_pos_end": args.char_pos_end,
        },
    )

    # Apply random token selection and detokenization
    gutenberg_with_tokens = gutenberg.map(
        select_tokens_from_random_offset,
        batched=True,
        desc="Selecting tokens from random offset and detokenizing",
        num_proc=num_proc,
        fn_kwargs={
            "_tokenizer": tokenizer,
            "num_tokens": args.num_tokens,
            "seed": args.seed,
        },
    )

    # Validate the processed gutenberg has exactly the number of tokens each article
    gutenberg_with_tokens = gutenberg_with_tokens.filter(
        lambda x: verify_num_token(x, tokenize_fn, args.num_tokens),
        num_proc=num_proc,
        desc="Verifying token counts"
    )

    print(f"After filtering: {len(gutenberg_with_tokens)}")

    gutenberg_final = gutenberg_with_tokens.select(range(args.num_articles))

    save_tokenized_datasets(gutenberg_final, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Gutenberg dataset for token selection"
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default="alehc/swissai-tokenizer", 
        help="Path to your tokenizer / Huggingface tokenizer ID"
    )
    parser.add_argument(
        "--num-articles", type=int, default=10_000, help="Number of articles to process"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=8190, 
        help="Specifies the number of tokens to extract per article. BOS and EOS token are added during actual tokenization, so 8192 - 2 = 8190.",
    )
    parser.add_argument(
        "--char-pos-start",
        type=int,
        default=10_000,
        help="Starting character position for tokenization",
    )
    parser.add_argument(
        "--char-pos-end",
        type=int,
        default=80_000,
        help="Ending character position for tokenization",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output file path",
    )

    args = parser.parse_args()
    main(args)
