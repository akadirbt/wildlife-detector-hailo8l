# Models and Training Artifacts

## Why this folder exists

The repository is not only about deployment code. A major part of the work also lives in:

- model training outputs
- export artifacts
- converted deployment models
- validation and comparison reports

These files are part of the engineering story, so they should be visible in the repo rather than staying only on a local machine.

## Artifact layout

```text
artifacts/
  models/
    best_h8l_balanced.hef

  training/
    exp9/
      weights/
        best.pt
        last.pt
        best.onnx
      results.csv
      results.png
      confusion_matrix.png
      F1_curve.png
      P_curve.png
      R_curve.png
      PR_curve.png
      labels.jpg
      labels_correlogram.jpg
      train_batch*.jpg
      val_batch*_labels.jpg
      val_batch*_pred.jpg
      hyp.yaml
      opt.yaml

  validation/
    raw_tensor_compare/
      report.md
      report.json
      head0_heatmap_compare.png
      head1_heatmap_compare.png
      head2_heatmap_compare.png
```

## What is included

## Deployment model

- `artifacts/models/best_h8l_balanced.hef`

This is the Hailo deployment artifact used on the detector side.

It represents the model in the form most relevant to Pi/Hailo deployment rather than only training-time experimentation.

## Training run outputs

- `artifacts/training/exp9/weights/best.pt`
- `artifacts/training/exp9/weights/last.pt`
- `artifacts/training/exp9/weights/best.onnx`

These show:

- the best checkpoint
- the last checkpoint
- the exported ONNX form

That makes the model-development path visible instead of only the final detector binary.

## Training metrics and plots

The `exp9/` folder also includes the usual results and visual diagnostics:

- confusion matrix
- F1 / precision / recall / PR curves
- aggregate results image
- CSV metrics output
- label statistics images

These are important because they show the model was evaluated and iterated on, not simply dropped into the project without evidence.

## Batch samples

The training folder includes example batch and validation images.

These give visual evidence of:

- what the model was trained/validated against
- how predictions looked during the training pipeline

## Validation and tensor comparison

The `artifacts/validation/raw_tensor_compare/` folder contains validation-side comparison material.

This part of the repo shows extra care around export fidelity and model behavior by preserving:

- validation reports
- comparison images
- supporting output summaries

It helps document that deployment artifacts were not treated as a black box.

## Why Git LFS is used

Some of the model files are large binary artifacts.

They are tracked with Git LFS so that:

- the repo remains healthier
- model binaries are still versioned
- GitHub does not become awkward to use for normal source browsing

## What this part of the repo shows

This section captures a major part of the real work behind the project:

- training effort
- export effort
- validation effort
- deployment preparation

Without these artifacts, the repo would underrepresent how much model-side work actually went into the system.
