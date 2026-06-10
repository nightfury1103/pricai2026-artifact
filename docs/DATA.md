# Data Notes

This anonymous artifact excludes full datasets and generated rationale CSV files to keep the repository small and avoid redistributing data outside its original licenses.

## Expected Directories

Training data should be placed under `datasets/`.

```text
datasets/
  cqa/
    cqa_train.json
    cqa_test.json
    llm/
      train_CoT_0.json
      train_CoT_1.json
      ...
      test_CoT_0.json
  esnli/
    esnli_train.json
    esnli_valid.json
    esnli_test.json
    llm/
      train_CoT_0.json
      ...
      valid_CoT_0.json
      test_CoT_0.json
```

Judge-derived candidate rationale CSV files are expected in local experiment directories with the same filenames used by the construction scripts. For example, ESNLI scripts read candidate files such as causal, contrastive, comparative, historical, if_else, neutral, and consensus rationale sources.

The default local directories are:

```text
[API] ESNLI/
[API] CQA/
```

They can be overridden without editing code:

```bash
export ESNLI_API_DIR=/path/to/esnli/rationale/csvs
export ESNLI_DATASET_DIR=/path/to/esnli/json
export CQA_API_DIR=/path/to/cqa/rationale/csvs
```

`run.py` also accepts explicit directory flags:

```bash
python run.py --dataset cqa --cqa_api_dir /path/to/cqa/rationale/csvs ...
python run.py --dataset esnli --esnli_api_dir /path/to/esnli/rationale/csvs ...
```

## Rebuilding JSON Dataset Files

The dataset loaders in `data_utils.py` can load public datasets through Hugging Face Datasets and write the normalized JSON files expected by `run.py`.

The artifact intentionally documents the expected layout instead of shipping the full local dataset cache.
