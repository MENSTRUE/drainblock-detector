# AGENTS.md

# DrainBlock Detector — AI Coding Agent Guidelines

This file defines the rules, constraints, conventions, and project context that every AI coding agent must follow when working on this repository.

The purpose of this file is to prevent incorrect assumptions, unsupported claims, accidental architecture changes, inconsistent preprocessing, and misleading evaluation.

---

# 1. Project Identity

Project name:

```text
DrainBlock Detector
```

Public learning progress label:

```text
6%
```

Repository:

```text
https://github.com/MENSTRUE/drainblock-detector
```

Current project status:

```text
MVP / Experimental Baseline
```

Current task:

```text
Weakly supervised visible-debris screening
```

Current classes:

```text
clean_context
debris
```

The current project is **not yet a validated physical drain blockage detector**.

---

# 2. Core Project Principle

The most important rule:

> Do not overclaim what the model can do.

The current model estimates whether an image is visually closer to:

```text
clean_context
```

or:

```text
debris
```

The model does NOT currently estimate:

- physical drain blockage percentage,
- blockage severity,
- drainage capacity,
- flood probability,
- water-flow obstruction,
- maintenance urgency,
- physical debris volume,
- infrastructure safety.

Never describe softmax confidence as physical blockage percentage.

Incorrect:

```text
The drain is 83% blocked.
```

Correct:

```text
The model assigns 83% confidence to the debris class.
```

---

# 3. Current Model

The current production/MVP checkpoint is:

```text
models/drainblock_mobilenetv3_FULL_v28.pth
```

Model architecture:

```text
MobileNetV3-Small
```

Framework:

```text
PyTorch
```

Classifier output:

```text
2 classes
```

Class order:

```python
[
    "clean_context",
    "debris",
]
```

Any inference implementation must preserve this class order unless the checkpoint metadata explicitly states otherwise.

---

# 4. Dataset

Dataset source:

```text
Outerview/global-trash-and-debris-index
```

Platform:

```text
Hugging Face
```

License:

```text
CC BY 4.0
```

Dataset type:

```text
General outdoor trash/debris imagery
```

Important:

The dataset is NOT a dedicated drainage dataset.

It does not contain validated labels for:

```text
blocked_drain
clean_drain
partial_blockage
severe_blockage
flood_risk
```

---

# 5. Dataset Audit

Final dataset audit:

```text
Metadata rows       : 25,000
Unique filenames    : 25,000
Images on disk      : 23,297
Missing images      : 1,703
Extra images        : 0
Usable images       : 23,297
```

The final full-data experiment used:

```text
23,297 / 23,297 usable images
```

No usable images were intentionally discarded.

---

# 6. Weak Supervision

The source dataset did not contain the exact labels needed for this MVP.

Pseudo-labels were generated using:

```text
openai/clip-vit-base-patch32
```

Concepts used:

```text
a street or outdoor scene with visible trash, litter,
garbage or debris on the ground
```

and:

```text
a clean street, sidewalk or outdoor scene
without visible trash or litter
```

CLIP is used only as a weak-label generator.

CLIP pseudo-labels are NOT human ground truth.

---

# 7. Pseudo-Label Distribution

Final pseudo-label distribution:

```text
debris           : 17,998
clean_context    : 5,299
total            : 23,297
```

This dataset is imbalanced.

The final pipeline keeps all samples.

Class imbalance is handled using:

```text
class-weighted CrossEntropyLoss
```

Do not add random undersampling unless explicitly requested.

Do not silently discard majority-class data.

---

# 8. Final Dataset Split

Final stratified split:

```text
Train       : 16,307
Validation  : 3,495
Pseudo-Test : 3,495
```

Total:

```text
23,297
```

All three sets are derived from CLIP pseudo-labels.

Therefore:

> test metrics measure agreement with pseudo-labels, not human-verified drain blockage accuracy.

---

# 9. Final Metrics

Final experiment results:

```text
Best Validation Macro F1 : 67.53%

Pseudo-Test Accuracy      : 72.90%
Pseudo-Test Precision     : 65.34%
Pseudo-Test Recall        : 69.24%
Pseudo-Test Macro F1      : 66.23%
```

These values must always be described as:

```text
pseudo-label evaluation metrics
```

or:

```text
agreement with CLIP-derived pseudo labels
```

Never write:

```text
Drain blockage detection accuracy = 72.90%
```

Preferred wording:

```text
The MobileNetV3 baseline achieved 72.90% pseudo-test
accuracy and 66.23% Macro F1 against CLIP-derived
pseudo labels.
```

---

# 10. Confusion Matrix Interpretation

Final pseudo-test confusion matrix:

```text
True clean_context predicted clean_context : 497
True clean_context predicted debris        : 298

True debris predicted clean_context        : 649
True debris predicted debris               : 2051
```

This means the model still produces meaningful:

```text
false positives
false negatives
```

Do not hide these errors.

They are useful evidence of current MVP limitations.

---

# 11. Real-World Domain Gap

Real-world drain images show clear domain shift.

Examples include:

- leaves blocking drains,
- trash inside concrete channels,
- roadside storm drains,
- wet surfaces,
- muddy drains,
- dark drainage channels,
- mixed organic and plastic debris.

The model may incorrectly classify visually obvious drain debris as:

```text
clean_context
```

This is expected because the original training corpus is not drain-specific.

Any future agent must preserve this explanation.

---

# 12. Current Application

The current application uses:

```text
Streamlit
```

Entry point:

```text
app.py
```

Expected user flow:

```text
Upload Image
     ↓
Preprocess
     ↓
Model Inference
     ↓
Prediction
     ↓
Confidence
     ↓
Class Probabilities
     ↓
Annotated Result
     ↓
Save Result
```

---

# 13. Saved Inference Results

Inference results should be stored in:

```text
assets/results/
```

Expected outputs:

```text
result_YYYYMMDD_HHMMSS.jpg
result_YYYYMMDD_HHMMSS.json
```

Do not save temporary outputs into:

```text
models/
notebooks/
reports/
```

unless explicitly required.

---

# 14. Inference Preprocessing

Inference preprocessing MUST match training preprocessing.

Input size:

```text
224 × 224
```

Normalization:

```python
mean = [
    0.485,
    0.456,
    0.406,
]

std = [
    0.229,
    0.224,
    0.225,
]
```

Inference must use:

```text
Resize
ToTensor
ImageNet normalization
```

Do NOT add random augmentation during inference.

Do NOT change preprocessing without also updating training or documenting the difference.

---

# 15. Checkpoint Loading

The checkpoint contains:

```text
model_state_dict
class_names
img_size
full_usable_images
clip_images_processed
all_images_retained
class_weights
best_validation_f1
test_metrics
```

Prefer reading metadata from the checkpoint rather than hardcoding values.

Fallback values may be used only when metadata is unavailable.

---

# 16. Repository Structure

Expected structure:

```text
drainblock-detector/
│
├── assets/
│   └── results/
│
├── models/
│   └── drainblock_mobilenetv3_FULL_v28.pth
│
├── notebooks/
│   └── drainblock.ipynb
│
├── reports/
│   ├── classification_report_FULL_pseudo_test.csv
│   ├── clip_weak_labels_FULL_23297.csv
│   ├── dataset_audit.json
│   ├── full_pseudo_labels_ALL_IMAGES.csv
│   ├── test_FULL_pseudo.csv
│   ├── train_FULL_pseudo.csv
│   ├── training_history_FULL_DATA.csv
│   └── val_FULL_pseudo.csv
│
├── src/
│
├── AGENTS.md
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```

Do not reorganize the repository without explicit instruction.

---

# 17. README Is Source of Truth

The repository README is the primary public documentation.

Agents must keep:

```text
README.md
```

consistent with:

```text
final_summary.json
reports/
checkpoint metadata
```

Never invent metrics that are not supported by experiment output.

Never round values differently across files unless necessary.

Preferred public rounding:

```text
Accuracy       : 72.90%
Macro Precision: 65.34%
Macro Recall   : 69.24%
Macro F1       : 66.23%
Best Val F1    : 67.53%
```

---

# 18. Training Notebook

Main experiment notebook:

```text
notebooks/drainblock.ipynb
```

The notebook should remain reproducible and should include:

- dataset audit,
- GPU detection,
- dependency checks,
- heartbeat logging,
- CLIP pseudo-labeling,
- full-data verification,
- class imbalance handling,
- split generation,
- training,
- validation,
- pseudo-test evaluation,
- confusion matrix,
- checkpoint saving,
- artifact packaging.

Do not remove heartbeat logs.

They were intentionally added because Kaggle jobs occasionally appeared to run without actually executing notebook cells.

---

# 19. Logging

Long-running notebook stages should print progress.

Examples:

```text
[BOOT] ...
[1/12] ...
[2/12] ...
[6/12][CLIP] batch 10/183
...
[DONE] ...
```

Any long loop should periodically print:

- current batch,
- total batch,
- elapsed time,
- ETA if practical.

Avoid silent long-running processes.

---

# 20. Kaggle Runtime

The training environment used during the final run included:

```text
NVIDIA Tesla T4
```

The notebook may detect multiple GPUs.

Do not assume both GPUs are always used.

Single-GPU execution is acceptable when it improves notebook stability.

---

# 21. App UX Rules

The Streamlit MVP should remain simple.

Avoid turning the app into an overly complex dashboard.

Main UI should focus on:

```text
Upload
Analyze
Prediction
Confidence
Probabilities
Saved Result
```

The user should be able to understand the result without reading technical documentation.

---

# 22. Result Language

Preferred result labels:

For debris:

```text
Potential debris detected
```

For clean context:

```text
No dominant debris signal detected
```

Avoid:

```text
Drain blocked
Drain safe
Drain clear
High flood risk
Low flood risk
```

because the current model does not support these claims.

---

# 23. Confidence Language

Always refer to model output as:

```text
model confidence
```

or:

```text
class probability
```

Never call it:

```text
blockage percentage
severity percentage
trash percentage
flood probability
```

unless a future model explicitly supports and validates those quantities.

---

# 24. Error Handling

Any new code should:

- fail clearly when the model file is missing,
- validate uploaded image types,
- handle corrupted images,
- avoid crashing on unsupported files,
- display readable error messages,
- avoid exposing raw stack traces to normal app users.

---

# 25. Dependency Policy

Keep dependencies minimal.

Core expected dependencies:

```text
streamlit
torch
torchvision
pillow
numpy
pandas
scikit-learn
matplotlib
transformers
```

Do not add heavy libraries without a clear reason.

Avoid installing multiple frameworks for the same task.

---

# 26. Coding Style

Prefer:

- clear Python,
- explicit function names,
- small reusable functions,
- type hints where useful,
- pathlib instead of raw string paths,
- comments for non-obvious ML logic,
- readable error messages.

Avoid:

- unnecessary abstraction,
- deeply nested helper layers,
- magic numbers without explanation,
- giant monolithic functions,
- silent exception handling.

---

# 27. Path Handling

Always use:

```python
from pathlib import Path
```

Prefer repository-relative paths.

Example:

```python
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "..."
```

Do not hardcode local developer paths such as:

```text
D:\porto\...
```

inside production code.

Kaggle-specific paths may appear only inside training notebooks.

---

# 28. Do Not Commit Secrets

Never commit:

- API keys,
- Hugging Face tokens,
- Kaggle credentials,
- passwords,
- access tokens,
- environment secrets.

Use:

```text
.env
```

or environment variables when credentials are required.

Ensure secret files remain ignored by Git.

---

# 29. Large File Awareness

Model files and large report files may be too large for normal Git workflows.

Before adding large artifacts:

- check file size,
- confirm whether Git LFS is required,
- avoid committing raw dataset images,
- avoid committing large temporary caches.

Never commit the original full dataset into this repository.

---

# 30. Git Rules

Every meaningful project change should have a descriptive commit.

Examples:

```text
feat: add Streamlit inference app
```

```text
feat: add full-data DrainBlock model
```

```text
docs: add complete project documentation
```

```text
fix: align inference preprocessing with training
```

```text
refactor: simplify result saving pipeline
```

Avoid vague commits such as:

```text
update
fix
code
structure
test
```

unless there is a strong reason.

---

# 31. Public Progress Branding

Public posts should use:

```text
6%
```

Do NOT use:

```text
Day 6/100
100 Days Challenge
Day 6
```

The percentage represents personal learning/growth progress, not a literal project count or challenge day.

---

# 32. Future Development Direction

Current system:

```text
General outdoor image
     ↓
Debris vs Clean Context
```

Future target:

```text
Drain Image
     ↓
Drain Detection
     ↓
Drain ROI
     ↓
Debris Detection
     ↓
Segmentation
     ↓
Spatial Occupancy
     ↓
Temporal Monitoring
     ↓
Risk Prioritization
     ↓
Human Review
```

---

# 33. Planned Future Improvements

Potential development stages:

## A. Drain-Specific Dataset

Collect images of:

```text
clean drains
minor debris
partial obstruction
heavy obstruction
overflow
```

---

## B. Human Annotation

Introduce manually verified labels.

Pseudo labels may remain useful for bootstrapping but should no longer be the sole ground truth.

---

## C. Drain Detection

Use object detection to locate drainage infrastructure.

Candidate models:

```text
YOLO11
RT-DETR
Faster R-CNN
```

---

## D. Debris Localization

Detect trash or obstruction inside the drain ROI.

---

## E. Segmentation

Estimate visible occupied area.

Candidate architectures:

```text
YOLO Seg
Mask R-CNN
SegFormer
DeepLabV3+
```

---

## F. Severity Modeling

Only introduce severity labels once human ground truth exists.

Potential labels:

```text
low
medium
high
```

Do not derive severity directly from softmax confidence.

---

## G. Temporal Monitoring

Potential future CCTV pipeline:

```text
Frame t
Frame t+1
Frame t+2
     ↓
Accumulation Trend
```

---

## H. Human-in-the-Loop

Preferred future operational workflow:

```text
AI Screening
     ↓
Priority Recommendation
     ↓
Human Verification
     ↓
Maintenance Decision
```

AI should support decisions, not replace human inspection in safety-critical scenarios.

---

# 34. MVP Philosophy

This repository values:

```text
honest evaluation
reproducibility
clear limitations
real artifacts
working software
defensible claims
```

over:

```text
inflated metrics
unsupported AI claims
fake production readiness
```

A lower but defensible metric is preferred over a high metric with weak methodology.

---

# 35. Agent Decision Rule

Before making any project change, ask:

```text
Does this change make the system more correct,
more reproducible,
more understandable,
or more useful?
```

If not, avoid unnecessary complexity.

---

# 36. Mandatory Validation After Code Changes

After modifying inference code:

Verify:

```text
1. checkpoint loads
2. class order is correct
3. preprocessing matches training
4. image upload works
5. inference returns two probabilities
6. probabilities sum approximately to 1
7. result image is saved
8. JSON result is saved
9. app does not claim physical blockage severity
```

After modifying model code:

Verify compatibility with:

```text
models/drainblock_mobilenetv3_FULL_v28.pth
```

---

# 37. Never Fabricate Results

AI agents must NEVER invent:

- accuracy,
- F1,
- confusion matrix values,
- dataset size,
- runtime,
- inference speed,
- benchmark comparisons,
- model improvements.

If a metric has not been measured:

write:

```text
Not measured yet.
```

Do not estimate it as if it were an experiment result.

---

# 38. Current Verified Final State

Current verified final experiment:

```text
Project              : DrainBlock Detector
Progress             : 6%
Status               : MVP
Task                 : Weakly supervised debris screening
Model                : MobileNetV3-Small
Usable Dataset       : 23,297 images
Pseudo Labeler       : CLIP ViT-B/32
Train                : 16,307
Validation           : 3,495
Pseudo-Test          : 3,495
Accuracy             : 72.90%
Macro Precision      : 65.34%
Macro Recall         : 69.24%
Macro F1             : 66.23%
Best Val Macro F1    : 67.53%
```

The final notebook completed successfully.

Final model:

```text
models/drainblock_mobilenetv3_FULL_v28.pth
```

---

# 39. Final Reminder

DrainBlock Detector is currently:

> A working weakly supervised debris-screening MVP and a foundation for future drain-specific computer vision research.

It is NOT currently:

> A production-ready drain blockage monitoring system.

Every future contribution should preserve this distinction until proper drain-specific ground truth and validation exist.