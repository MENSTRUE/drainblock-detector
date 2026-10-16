# DrainBlock Detector

> **6% — AI/Computer Vision Learning Progress**

A weakly supervised computer vision MVP for screening visible trash and debris in outdoor scenes as an early building block toward drain blockage monitoring.

<p align="center">
  <b>Image → Debris Screening → Confidence → Saved Result</b>
</p>

---

## Overview

**DrainBlock Detector** is a computer vision prototype developed to explore how AI can assist in identifying visible debris in outdoor environments.

The long-term idea is to support early screening of drainage areas that may require human inspection.

The current version is intentionally positioned as an **MVP and weakly supervised debris-screening baseline**.

It is **not yet a validated drain blockage detector**.

The current model classifies an image into:

- `debris`
- `clean_context`

The model can be accessed through a simple Streamlit application where users upload an image and receive:

- predicted class,
- model confidence,
- class probabilities,
- annotated result image,
- automatically saved inference output.

---

# Problem Background

Trash and debris can accumulate around roads, drainage channels, sidewalks, and urban environments.

In a future operational system, computer vision could potentially assist human operators by screening large numbers of images and highlighting scenes that deserve closer inspection.

However, building a real drain blockage detector requires:

- drain-specific imagery,
- reliable human annotations,
- blockage severity labels,
- localization or segmentation,
- external-domain evaluation,
- and operational validation.

Those requirements are not yet available in this MVP.

Because of that, this project starts from a smaller and more defensible problem:

> **Can a model learn to distinguish outdoor scenes containing visible debris from relatively clean outdoor contexts using weak supervision?**

---

# Project Status

**Current status: MVP / Experimental Baseline**

The current system supports:

- full-dataset preprocessing,
- CLIP-based pseudo-label generation,
- MobileNetV3-Small classification,
- class imbalance handling,
- train/validation/test splitting,
- model evaluation,
- local image inference,
- Streamlit upload interface,
- automatic result saving.

The system does **not yet** estimate:

- physical blockage percentage,
- drain capacity,
- water-flow obstruction,
- flood probability,
- blockage severity,
- maintenance urgency,
- or actual drainage condition.

---

# Dataset

## Source

The source dataset used in this project is:

**Global Trash and Debris Index**

Hugging Face dataset:

```text
Outerview/global-trash-and-debris-index
```

Original source used during development:

```text
https://huggingface.co/datasets/Outerview/global-trash-and-debris-index
```

License:

```text
CC BY 4.0
```

The dataset contains general street-view / outdoor trash and debris imagery.

---

## Important Dataset Limitation

This dataset is **not a dedicated drainage dataset**.

It does not provide ground-truth labels for:

- clogged drains,
- drainage blockage severity,
- water obstruction,
- drainage condition,
- blockage percentage,
- flood risk.

It also does not provide bounding-box annotations specifically for trash blocking drains.

Therefore, the model developed in this repository must be interpreted as a:

> **general visible-debris screening baseline**

rather than a production-ready drain blockage detector.

---

# Dataset Audit

The original metadata contained:

| Item | Count |
|---|---:|
| Metadata rows | 25,000 |
| Unique filenames | 25,000 |
| Images available on disk | 23,297 |
| Missing images | 1,703 |
| Extra images | 0 |
| Usable images | **23,297** |

Only images physically available in the cleaned dataset were included in model development.

---

# Full-Data Training

Earlier experiments used smaller subsets to validate the pipeline and reduce runtime.

The final MVP experiment was then rerun using **all 23,297 usable images**.

```text
Usable images       : 23,297
CLIP processed       : 23,297
Images retained      : 23,297
Images discarded     : 0
```

No usable images were removed during final pseudo-label preparation.

---

# Why Weak Supervision?

The dataset did not contain the binary labels needed by this MVP:

```text
clean_context
debris
```

Instead of manually labeling more than twenty thousand images, the project uses **CLIP** as a pseudo-label generator.

The workflow is:

```text
Raw Dataset
     ↓
Dataset Audit
     ↓
23,297 Usable Images
     ↓
CLIP Zero-Shot Screening
     ↓
Pseudo Labels
     ↓
clean_context / debris
     ↓
Stratified Split
     ↓
MobileNetV3-Small
     ↓
Evaluation
     ↓
Streamlit Inference App
```

This approach allows the entire dataset to participate in the experiment while keeping the project feasible as a portfolio-scale MVP.

---

# CLIP Pseudo-Labeling

The project uses:

```text
openai/clip-vit-base-patch32
```

Two textual concepts are compared against every image.

Concept 1:

```text
a street or outdoor scene with visible trash, litter,
garbage or debris on the ground
```

Concept 2:

```text
a clean street, sidewalk or outdoor scene
without visible trash or litter
```

CLIP embeddings are generated for both text and images.

The image is assigned to whichever concept has the larger similarity probability.

---

# Pseudo-Label Distribution

After processing all usable images:

| Pseudo Class | Images |
|---|---:|
| `debris` | 17,998 |
| `clean_context` | 5,299 |
| **Total** | **23,297** |

The pseudo-label distribution is clearly imbalanced.

Instead of discarding majority-class samples, the final experiment keeps **all available images**.

---

# Handling Class Imbalance

Because `debris` appears much more frequently than `clean_context`, the project uses **class-weighted CrossEntropyLoss**.

Training class weights:

```text
clean_context ≈ 2.1983
debris        ≈ 0.6472
```

This allows all samples to remain in the training pipeline while increasing the penalty for mistakes involving the minority class.

---

# Dataset Split

The complete pseudo-labeled dataset is divided using a stratified split:

```text
70% Training
15% Validation
15% Pseudo-Test
```

Final split:

| Split | Images |
|---|---:|
| Training | 16,307 |
| Validation | 3,495 |
| Pseudo-Test | 3,495 |
| **Total** | **23,297** |

Class distribution is preserved through stratification.

---

# Model

The final lightweight classifier uses:

```text
MobileNetV3-Small
```

Why MobileNetV3-Small?

- lightweight,
- fast inference,
- suitable for MVP deployment,
- relatively small checkpoint,
- practical for CPU inference,
- potential future mobile/edge deployment,
- pretrained ImageNet features.

---

## Architecture Strategy

The model starts from ImageNet pretrained weights.

The original classifier head is replaced with:

```text
MobileNetV3-Small
        ↓
Feature Extractor
        ↓
Classifier
        ↓
2 Classes
        ↓
clean_context / debris
```

---

# Training Strategy

Training is performed in two stages.

## Stage 1 — Classifier Training

The MobileNetV3 feature extractor is frozen.

Only the classifier head is trained.

```text
Epochs: 2
Learning Rate: 3e-4
Optimizer: AdamW
```

---

## Stage 2 — Partial Fine-Tuning

The final feature blocks are unfrozen.

The model is then fine-tuned using a smaller learning rate.

```text
Epochs: 2
Learning Rate: 8e-5
Optimizer: AdamW
```

Total:

```text
4 epochs
```

Mixed precision is used when CUDA is available.

---

# Image Preprocessing

Images are resized to:

```text
224 × 224
```

Training augmentation includes:

- resize,
- random horizontal flip,
- small brightness variation,
- small contrast variation,
- tensor conversion,
- ImageNet normalization.

Validation and test preprocessing use deterministic resize and normalization.

---

# Final Results

## Validation

Best validation Macro F1:

```text
67.53%
```

---

## Pseudo-Test Results

| Metric | Score |
|---|---:|
| Accuracy | **72.90%** |
| Macro Precision | **65.34%** |
| Macro Recall | **69.24%** |
| Macro F1 | **66.23%** |

These values describe agreement with the **CLIP-generated pseudo labels**.

They must **not** be interpreted as human-ground-truth drain blockage performance.

---

# Interpreting the Metrics

This distinction is important.

A statement such as:

> "DrainBlock detects clogged drains with 72.9% accuracy"

would be incorrect.

A more accurate description is:

> **The MobileNetV3 baseline achieved 72.9% accuracy and 66.2% Macro F1 when reproducing CLIP-derived debris versus clean-context pseudo labels on the held-out pseudo-test split.**

The source dataset is not drain-specific and the labels were not manually verified as blockage ground truth.

---

# Inference Application

The repository includes a Streamlit interface:

```text
app.py
```

The application allows users to:

1. upload an image,
2. run model inference,
3. view the prediction,
4. view model confidence,
5. inspect probabilities,
6. view the generated result image,
7. automatically save inference results.

---

# Application Flow

```text
Upload Image
     ↓
Image Preprocessing
     ↓
MobileNetV3-Small
     ↓
Softmax Probabilities
     ↓
Prediction
     ↓
debris / clean_context
     ↓
Display Result
     ↓
Save Image + JSON
```

---

# Running the Application

## 1. Clone Repository

```bash
git clone https://github.com/MENSTRUE/drainblock-detector.git
cd drainblock-detector
```

---

## 2. Create Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Streamlit

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# Model Checkpoint

Final model:

```text
models/
└── drainblock_mobilenetv3_FULL_v28.pth
```

The checkpoint stores:

- model state dictionary,
- class names,
- image size,
- dataset metadata,
- class weighting information,
- validation result,
- pseudo-test metrics.

---

# Output Classes

## `debris`

The image contains visual patterns that the model associates with:

- trash,
- litter,
- garbage,
- scattered debris,
- cluttered outdoor ground surfaces.

Example interpretation:

```text
Potential debris detected
```

---

## `clean_context`

The image is visually closer to:

- relatively clean streets,
- clean sidewalks,
- clean outdoor surfaces,
- scenes without dominant debris patterns.

Example interpretation:

```text
No dominant debris signal detected
```

---

# Confidence Is Not Severity

Suppose the app produces:

```text
Prediction : debris
Confidence : 87.4%
```

This means:

> The classifier assigns approximately 87.4% softmax probability to the `debris` class.

It does **not** mean:

```text
Drain is 87.4% blocked
```

and it does not represent:

- physical fill percentage,
- debris volume,
- blockage severity,
- drainage capacity,
- flood risk.

---

# Saved Inference Results

Inference results are automatically stored under:

```text
assets/results/
```

Example:

```text
assets/
└── results/
    ├── result_20261016_194500.jpg
    └── result_20261016_194500.json
```

The `.jpg` contains the displayed result.

The `.json` stores the prediction information.

Example:

```json
{
  "prediction": "debris",
  "confidence": 0.8342,
  "probabilities": {
    "clean_context": 0.1658,
    "debris": 0.8342
  }
}
```

---

# Repository Structure

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

---

# Reports

The project retains experiment artifacts to make the model-development process auditable.

### `dataset_audit.json`

Stores dataset integrity statistics.

### `clip_weak_labels_FULL_23297.csv`

Contains CLIP pseudo-label predictions for all usable images.

### `full_pseudo_labels_ALL_IMAGES.csv`

Contains the final pseudo-labeled dataset used for splitting.

### `train_FULL_pseudo.csv`

Training split.

### `val_FULL_pseudo.csv`

Validation split.

### `test_FULL_pseudo.csv`

Held-out pseudo-test split.

### `training_history_FULL_DATA.csv`

Training history.

### `classification_report_FULL_pseudo_test.csv`

Detailed pseudo-test classification metrics.

---

# Development Journey

The project was developed iteratively rather than assuming the first design would work.

## Phase 1 — Problem Definition

Initial idea:

```text
Drain image
     ↓
AI
     ↓
Blocked / Not Blocked
```

After inspecting the available dataset, this framing was considered too strong because the dataset did not contain true drain blockage labels.

The project was therefore reframed as:

```text
Outdoor image
     ↓
Visible debris screening
     ↓
Potential inspection signal
```

---

## Phase 2 — Dataset Audit

The downloaded dataset contained metadata for 25,000 images.

Actual usable files:

```text
23,297
```

A total of:

```text
1,703
```

metadata references did not have corresponding image files in the cleaned dataset.

---

## Phase 3 — Weak Label Generation

Since human labels were unavailable, CLIP was introduced to generate weak labels.

Early experiments used smaller subsets to verify:

- dataset loading,
- CLIP runtime,
- label distribution,
- training stability,
- Kaggle GPU execution.

---

## Phase 4 — Runtime Debugging

Several Kaggle runs appeared as:

```text
Running
```

while producing no logs.

Heartbeat logging was added throughout the notebook so execution could be distinguished from stalled Kaggle sessions.

Example:

```text
[BOOT] DrainBlock v2.8 FULL DATA started
[1/12] ...
[2/12] ...
...
[DONE] DRAINBLOCK FULL DATA PIPELINE FINISHED
```

---

## Phase 5 — Full Dataset Run

Once the pipeline was stable, the experiment was rerun using every usable image.

```text
23,297 / 23,297 images
```

were processed.

---

## Phase 6 — MVP Inference

The trained checkpoint was integrated into a Streamlit interface.

The MVP now supports:

```text
Upload
→ Predict
→ Visualize
→ Save
```

---

# Current Limitations

## 1. Weak Labels

The labels come from CLIP rather than manual human annotation.

Any systematic CLIP error can therefore propagate into the MobileNet classifier.

---

## 2. Source Domain

The training corpus represents general outdoor trash/debris scenes.

It is not specifically designed around:

- Indonesian drainage channels,
- storm drains,
- culverts,
- roadside gutters,
- sewer entrances.

---

## 3. Classification Instead of Localization

The current model predicts one label for the entire image.

It cannot show exactly **where** the debris is located.

---

## 4. No Drain Detection

The current system does not explicitly identify the drain itself.

A future system should first identify a drainage region before evaluating debris around it.

---

## 5. No Physical Blockage Measurement

The model cannot determine whether water flow is actually obstructed.

---

## 6. No Severity Ground Truth

There are currently no validated labels such as:

```text
Low blockage
Medium blockage
High blockage
```

Therefore the MVP intentionally avoids presenting model confidence as physical severity.

---

## 7. Domain Shift

Images from:

- CCTV,
- smartphones,
- drones,
- rainy environments,
- nighttime scenes,
- Indonesian residential areas,

may behave differently from the original training corpus.

---

# Future Development

DrainBlock is intended to evolve beyond the current MVP.

## Phase A — Drain-Specific Dataset

Collect or curate images containing actual drainage infrastructure.

Target examples:

```text
clean drain
minor debris
partial obstruction
heavy obstruction
overflow
```

---

## Phase B — Human Annotation

Replace or supplement CLIP pseudo labels with human-reviewed labels.

This will enable evaluation against actual ground truth.

---

## Phase C — Object Detection

Move from image-level classification toward object detection.

Potential pipeline:

```text
Image
   ↓
Drain Detection
   ↓
Debris Detection
```

Possible architectures:

```text
YOLO11
RT-DETR
Faster R-CNN
```

---

## Phase D — Segmentation

Segmentation could estimate how much of a drainage region is occupied by visible debris.

Potential architectures:

```text
YOLO Segmentation
Mask R-CNN
SegFormer
DeepLabV3+
```

---

## Phase E — Drain ROI

Introduce a dedicated drain region of interest.

Concept:

```text
Full Image
     ↓
Detect Drain
     ↓
Crop Drain ROI
     ↓
Analyze Debris
```

This would reduce false positives from trash located far away from drainage infrastructure.

---

## Phase F — Temporal Monitoring

For CCTV or fixed-camera deployments:

```text
Frame t
Frame t+1
Frame t+2
      ↓
Temporal Change
      ↓
Debris Accumulation Trend
```

This could help distinguish temporary objects from gradually accumulating debris.

---

## Phase G — Human-in-the-Loop

A practical deployment should not automatically declare infrastructure unsafe.

A more responsible workflow would be:

```text
AI Screening
     ↓
Priority Recommendation
     ↓
Human Review
     ↓
Inspection / Maintenance Decision
```

---

# Possible Future Architecture

```text
Camera / Uploaded Image
          ↓
     Drain Detector
          ↓
      Drain Region
          ↓
     Debris Detector
          ↓
     Segmentation
          ↓
 Spatial Occupancy Features
          ↓
 Temporal / Context Features
          ↓
     Risk Prioritization
          ↓
       Human Review
```

---

# MVP vs Future System

| Capability | Current MVP | Future Target |
|---|:---:|:---:|
| Debris screening | ✅ | ✅ |
| Image upload | ✅ | ✅ |
| Confidence output | ✅ | ✅ |
| Saved results | ✅ | ✅ |
| Full-dataset training | ✅ | ✅ |
| Drain detection | ❌ | ✅ |
| Debris localization | ❌ | ✅ |
| Segmentation | ❌ | ✅ |
| Human ground truth | ❌ | ✅ |
| Severity estimation | ❌ | ✅ |
| Temporal monitoring | ❌ | ✅ |
| Maintenance prioritization | ❌ | ✅ |
| Real-world validation | ❌ | ✅ |

---

# Reproducibility

The notebook uses a fixed random seed:

```text
42
```

The final experiment records:

- dataset audit,
- generated pseudo labels,
- dataset splits,
- training history,
- classification report,
- model checkpoint,
- final experiment summary.

This makes the development process more transparent and easier to reproduce.

---

# Technology Stack

### Machine Learning

- Python
- PyTorch
- Torchvision
- MobileNetV3-Small

### Weak Supervision

- OpenAI CLIP
- Hugging Face Transformers

### Data

- Pandas
- NumPy
- Pillow
- Scikit-learn

### Interface

- Streamlit

### Development / Training

- Kaggle
- NVIDIA Tesla T4
- Git
- GitHub

---

# Key Learning

This project demonstrates an important machine-learning lesson:

> A model architecture is only one part of an AI system. Dataset quality, label validity, domain alignment, evaluation design, and honest interpretation of results are equally important.

The first idea was to build a direct drain blockage detector.

The available evidence did not support that claim.

Instead of forcing the result, the project was reframed into a smaller but defensible MVP:

> **weakly supervised visible-debris screening**

This provides a working technical baseline while clearly defining what must be improved before the system can become a real drainage-monitoring solution.

---

# Disclaimer

DrainBlock Detector is an experimental portfolio project.

The current model should not be used as the sole basis for:

- infrastructure maintenance decisions,
- flood warnings,
- municipal operations,
- safety-critical decisions,
- environmental enforcement.

Predictions should be interpreted as AI-assisted screening signals only.

---

# License

Project code is released under the license included in:

```text
LICENSE
```

The source dataset has its own license and attribution requirements.

Dataset:

```text
Outerview/global-trash-and-debris-index
```

License:

```text
CC BY 4.0
```

Please review the original dataset license before redistribution or commercial use.

---

# Author

**Wafa Bila Syaefurokhman**

Focus:

```text
Android Development
Artificial Intelligence
Machine Learning
Computer Vision
```

GitHub:

```text
https://github.com/MENSTRUE
```

---

## Closing Note

DrainBlock currently answers a relatively small question:

> **Does this outdoor image visually resemble a debris-heavy or relatively clean scene?**

The long-term goal is much larger:

> **Can computer vision help prioritize drainage infrastructure that deserves human inspection before problems become more serious?**

This repository documents the first working step toward that direction.