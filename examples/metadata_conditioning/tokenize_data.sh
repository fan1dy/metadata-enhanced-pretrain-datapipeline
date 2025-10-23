python3 tokenize_data_with_metainfo_megatron.py \
--add-metadata \
--add-metadata-per-sequence \
--metadata-position "left" \
--metadata-builder "url" \
--output-folder "<your-self-defined-output-folder>" \
--logging-dir "<your-self-defined-logging-folder>" \
--tokenizer-name-or-path dyfan/swissai-tokenizer-wcontext parquet \
--dataset "<dataset-path, e.g. a local copy of FineWeb-Edu dataset>"