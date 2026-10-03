# ViT Failure Diagnosis on Stanford Cars

Explaining and interpreting the mistakes of a Vision Transformer, fine-tuned on
the Stanford Cars dataset. The notebook builds a full pipeline from model
fine-tuning through attribution-based failure diagnosis to an automated
evaluation report.

## Pipeline

| Step | What | Section |
|---|---|---|
| 1 | Load ViT, build labelled Stanford-Cars split (labels come from the devkit metadata) | 1-4 |
| 2-3 | Evaluate on held-out test split, build failure set **F** | 5 |
| 4 | Chefer / Gradient / Attention-Rollout maps, normalisation, agreement, top-K regions | 6-7 |
| 5 | Perturbation validation | 8 |
| 6 | Counterfactual analysis | 8 |
| 7-9 | Failure indicators, rule-based failure mode, stored diagnosis | 9-10 |
| 10-11 | Aggregation and comparisons (correct vs wrong, per class, per method) | 11 |
| 12 | Evaluation of the framework (deletion/insertion, localisation, faithfulness, human validation, consistency) | 12 |
| 13 | Final report | 13 |

**Design notes**
- Every image is resized (squashed) to 224x224 with no crop, so the bounding
  box from the metadata maps exactly onto the model input and doubles as
  ground-truth object localisation.
- The ViT (`vit_base_patch16_224.augreg_in21k_ft_in1k`, via `timm`) is
  fine-tuned from an ImageNet checkpoint.
- Three attribution methods are compared throughout: Chefer, raw gradient,
  and attention rollout, plus their consensus.
- Failure modes identified: `BACKGROUND_RELIANCE`, `LOCALIZATION_FAILURE`,
  `TEXTURE_BIAS`, `CLASS_CONFUSION`, `PARTIAL_OBJECT_FAILURE`,
  `OTHER_AMBIGUOUS`.

## Repo layout

```
.
├── vit_failure_diagnosis.ipynb   # the full pipeline (sections 0-13)
├── outputs/                      # checkpoints, diagnosis artifacts, figures (generated, gitignored)
├── requirements.txt
└── LICENSE
```

## Setup

This notebook was developed on Google Colab (GPU runtime) with the dataset
and checkpoints cached on Google Drive. To run it elsewhere:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Dataset: [Stanford Cars](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)
   (via Kaggle CLI, as set up in Section 1). Update `CFG.local_root` to point
   at your local copy.
3. Outputs/checkpoint path: update `CFG.out_dir` (defaults to a Google Drive
   path) to wherever you want checkpoints and diagnosis artifacts saved.
4. If not running on Colab, remove/skip the `google.colab.drive.mount` cell
   in Section 1.
5. Run the notebook top to bottom. `CFG.max_failures` can be set to a small
   number for a fast dry run before committing to the full failure set.

## Status

Independent research project — see the notebook's "Notes and limitations"
section (end of Section 13) for current caveats.
