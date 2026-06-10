# PRICAI 2026 Anonymous Code Artifact

This repository contains an anonymized implementation for the PRICAI 2026 submission. It provides the training, evaluation, rationale-pack construction, and ablation scripts used for the experiments.

The code builds on the public Distilling Step-by-Step implementation and extends it with rationale selection, boundary-focused rationale packs, CQA/ESNLI judge pipelines, and score-component ablations.

## Repository Contents

```text
run.py                                  Training entry point
evaluate.py                             Saved-model evaluation script
data_utils.py                           Dataset loading and preprocessing
model_utils.py                          Trainer/data-collator extensions
train_utils.py                          Training orchestration
metrics.py                              Text/equation metrics
build_judge_esnli.py                    ESNLI rationale-pack construction
build_judge_cqa_outstanding_pack.py     CQA rationale-pack construction
build_boundary_focus_pack.py            ESNLI boundary-family rationale selection
build_shortcut_focus_pack.py            Shortcut-focused rationale selection
build_hybrid_focus_pack.py              Hybrid rationale-pack construction
build_boundary_score_ablation.py        ESNLI score-component ablation
build_cqa_boundary_score_ablation.py    CQA score-component ablation
tests/                                  Unit tests for selection and ablation logic
results/                                Small summary files for ablation reports
docs/                                   Data and anonymization notes
```

Large datasets, checkpoints, logs, and generated intermediate CSV files are intentionally not included in this anonymous review artifact.

## Environment

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches the local CUDA driver before installing the remaining requirements. The original experiments used T5-family models through Hugging Face Transformers.

## Data Layout

Prepare data under `datasets/`:

```text
datasets/
  cqa/
    cqa_train.json
    cqa_test.json
    llm/
      train_CoT_0.json
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

The source datasets are public datasets such as CQA/Cos-E and ESNLI. Generated rationale files should follow the naming convention shown above.

## Training Examples

Standard fine-tuning:

```bash
python run.py \
  --from_pretrained google/t5-v1_1-base \
  --dataset cqa \
  --model_type standard \
  --label_type gt \
  --llm none \
  --batch_size 64
```

Distilling step-by-step with ground-truth labels and LLM rationales:

```bash
python run.py \
  --from_pretrained google/t5-v1_1-base \
  --dataset cqa \
  --model_type task_prefix \
  --label_type gt \
  --llm palm \
  --alpha 0.5 \
  --batch_size 64
```

Training writes the final saved model to `model_path/<type_rationale>` by default. Override this with `--save_model_dir`.

Standard distillation with LLM labels:

```bash
python run.py \
  --from_pretrained google/t5-v1_1-base \
  --dataset cqa \
  --model_type standard \
  --label_type llm \
  --llm palm \
  --batch_size 64
```

## Rationale-Pack Construction

Build ESNLI judge-derived rationale packs:

```bash
export ESNLI_API_DIR="[API] ESNLI"
export ESNLI_DATASET_DIR="datasets/esnli"
python build_judge_esnli.py
python build_boundary_focus_pack.py
python build_shortcut_focus_pack.py
python build_hybrid_focus_pack.py
```

Build CQA judge-derived rationale packs:

```bash
export CQA_API_DIR="[API] CQA"
python build_judge_cqa_outstanding_pack.py
```

Run score-component ablations:

```bash
python build_boundary_score_ablation.py
python build_cqa_boundary_score_ablation.py
```

These scripts expect the generated candidate rationale CSV files to be available in the local data directories described in `docs/DATA.md`.

## Evaluation

Evaluate a saved model:

```bash
python evaluate.py \
  --model-path model_path/<type_rationale> \
  --test-data datasets/cqa/cqa_test.json \
  --output-path result/cqa_predictions.csv \
  --batch-size 8
```

## Tests

Run the lightweight unit tests:

```bash
python -m unittest discover -s tests
```

The tests cover the score-component decomposition and CQA boundary-specialist selection behavior. They do not require the full datasets.

## Anonymous Review Notes

This artifact has no Git history from the development repository. Before sharing through an anonymous repository mirror, verify the checklist in `docs/ANONYMIZATION.md`.

During double-anonymous review, cite only the anonymous mirror URL. Replace it with the final public GitHub URL after acceptance.

## License and Attribution

This code is distributed under the Apache License 2.0. Portions of the training pipeline derive from the public Distilling Step-by-Step codebase:

```bibtex
@article{hsieh2023distilling,
  title={Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes},
  author={Hsieh, Cheng-Yu and Li, Chun-Liang and Yeh, Chih-Kuan and Nakhost, Hootan and Fujii, Yasuhisa and Ratner, Alexander and Krishna, Ranjay and Lee, Chen-Yu and Pfister, Tomas},
  journal={arXiv preprint arXiv:2305.02301},
  year={2023}
}
```
