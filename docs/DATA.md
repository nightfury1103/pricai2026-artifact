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

## Rebuilding JSON Dataset Files

The dataset loaders in `data_utils.py` can load public datasets through Hugging Face Datasets and write the normalized JSON files expected by `run.py`.

The artifact intentionally documents the expected layout instead of shipping the full local dataset cache.

