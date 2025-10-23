"""
To process HuggingFace Datasets:
    python3 examples/tokenize_megatron/preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/emotion --n-tasks 16 hf --dataset dair-ai/emotion
To process Jsonl files:
    python3 examples/tokenize_megatron/preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/c4-es --n-tasks 16 jsonl --dataset raw_datasets/c4-es-json-files
"""

import argparse
from data_pipeline_pretrain.utils import list_files
from data_pipeline_pretrain.executor import SlurmPipelineNodeExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from data_pipeline_pretrain.pipeline.tokens import MegatronDocumentTokenizer
from datatrove.pipeline.readers import ParquetReader


DUMPS = [
    "CC-MAIN-2022-21",
        "CC-MAIN-2022-27",
         "CC-MAIN-2022-33",
         "CC-MAIN-2022-40",
         "CC-MAIN-2023-40",
         "CC-MAIN-2023-50",
        "CC-MAIN-2024-10"]

def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS token to add after each document. Default: <|endoftext|>",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder to store the tokenized documents",
    )
    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Path to a folder for storing the logs of the preprocessing step. Default: None",
    )
    group.add_argument(
        "--n-tasks",
        type=int,
        default=8,
        help="Total number of tasks to run the preprocessing step. Default: 8",
    )
    # Subparsers for processing either Hugging Face datasets or jsonl files
    sp = parser.add_subparsers(
        dest="readers",
        required=True,
        description="Type of dataset to process. It can be either a Hugging Face Dataset loaded with datasets.load_data ('hf') or a .jsonl dataset ('jsonl')",
    )

    p1 = sp.add_parser(name="hf")
    p1.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to local stored dataset or repository on the Hugging Face hub that can be loaded with datasets.load_dataset",
    )
    p1.add_argument(
        "--column",
        type=str,
        default="text",
        help="Column to preprocess from the Dataset. Default: text",
    )
    p1.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split of the data to process. Default: train",
    )

    p2 = sp.add_parser(name="parquet")
    p2.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a .jsonl file or a folder containing multiple .jsonl files",
    )
    p2.add_argument(
        "--column",
        type=str,
        default="text",
        help="Column to preprocess from the Dataset. Default: text",
    )
    p2.add_argument(
        "--glob-pattern",
        type=str,
        default=None,
        help="A glob pattern to filter files to read. Default: None",
    )

    # add meta data related
    group = parser.add_argument_group(title="Meta Data")
    group.add_argument(
        "--add-metadata",
        action="store_true",
        help="Add metadata to the tokenized documents. Default: False",
    )
    group.add_argument(
        "--meta-data-ratio",
        type=float,
        default=0.9,
        help="How often do we add metedata to the tokenized documents. Default: 0.9",
    )
    group.add_argument(
        "--metadata-position",
        type=str,
        default="left",
        # choices=["left", "right"],
    )
    group.add_argument(
        "--add-metadata-per-sequence",
        action="store_true",
        help="Add metadata per sequence. Default: add to per doc",
    )
    group.add_argument(
        "--metadata-builder",
        type=str,
        choices=["url_domain", "url_suffix", "url", "WO", "QS"],
        default="url",
        help="Select the metadata builder function. Default: url_suffix",
    )

    args = parser.parse_args()

    return args

def make_url_domain_metadata_builder(p_add_meta, seed):
    from numpy.random import default_rng
    from urllib.parse import urlparse
    uniform = default_rng(seed).uniform
    def fn(doc):
        url_domain = urlparse(doc.metadata["url"]).netloc
        return url_domain if uniform() < p_add_meta else ''
    return fn


def make_url_suffix_metadata_builder(p_add_meta, seed):
    from numpy.random import default_rng
    from urllib.parse import urlparse
    uniform = default_rng(seed).uniform
    def fn(doc):
        parsed = urlparse(doc.metadata["url"])
        url_suffix = parsed.path + ("?" + parsed.query if parsed.query else "") + ("#" + parsed.fragment if parsed.fragment else "")
        return url_suffix if uniform() < p_add_meta else ''
    return fn
    
def make_url_metadata_builder(p_add_meta, seed):
    from numpy.random import default_rng
    uniform = default_rng(seed).uniform
    def fn(doc):
        return doc.metadata["url"] if uniform() < p_add_meta else ''
    return fn

def make_WO_metadata_builder(p_add_meta, seed):
    from numpy.random import default_rng
    uniform = default_rng(seed).uniform
    def fn(doc):
        return f'{doc.metadata["weborganizer_topic"]}, {doc.metadata["weborganizer_format"]}' if uniform() < p_add_meta else ''
    return fn

def make_QS_metadata_builder(p_add_meta, seed):
    from numpy.random import default_rng
    uniform = default_rng(seed).uniform
    def fn(doc):
        return str(doc.metadata["int_score"]) if uniform() < p_add_meta else ''
    return fn

def main(args):
    # Build datatrove reader
    metadata_builder_map = {
        "url_domain": make_url_domain_metadata_builder,
        "url_suffix": make_url_suffix_metadata_builder,
        "url": make_url_metadata_builder,
        "WO": make_WO_metadata_builder,
        "QS": make_QS_metadata_builder,
    }
    selected_metadata_builder = metadata_builder_map[args.metadata_builder]

    for dump in DUMPS:
        input_dir = f"{args.dataset}/{dump}" 
        if args.readers == "hf":
            datatrove_reader = HuggingFaceDatasetReader(
                dataset=input_dir,
                text_key=args.column,
                dataset_options={"split": args.split},
            )
        else:
            datatrove_reader = ParquetReader(
                    data_folder=input_dir,
                )

        logs_dir = f"{args.logging_dir}/{dump}/"
            
        print("add meta data:", args.add_metadata)
        pipeline=[
            datatrove_reader,
            MegatronDocumentTokenizer(
                add_metadata=args.add_metadata,
                add_metadata_per_sequence=args.add_metadata_per_sequence,
                metadata_builder=selected_metadata_builder(args.meta_data_ratio, 42),
                mask_metadata=True,
                output_folder=f"{args.output_folder}/{dump}",   
                metadata_position=args.metadata_position,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                sequence_length=4096,
                eos_token=None,
        )
            ]

        SlurmPipelineNodeExecutor(
                pipeline=pipeline,
                job_name="tokenization",
                time="02:00:00",
                partition="normal",
                cpus_per_task=12,
                tasks=len(list_files(input_dir)),
                workers=4,
                logging_dir=logs_dir,
                srun_args={
                    "environment": <ADD the path to a toml file>,
                },
            ).run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)
