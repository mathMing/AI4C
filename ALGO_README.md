# AI4C - Algo (Model Training & Inference)

This directory contains the model training and inference code for **AI4C** (Diabetic Retinopathy grading from fundus images).

## Directory layout

`
algo/
├── configs/                # YAML configs for models
│   ├── retfound_b16.yaml   # RETFound ViT-B/16 (default)
│   └── dr_vit_b16.yaml     # Generic ViT-B/16
├── datasets/               # PyTorch dataset wrappers
│   └── fundus_dataset.py
├── models/                 # Model definitions
│   ├── retfound_classifier.py
│   └── vit_classifier.py
├── transforms/             # Data augmentations
│   └── fundus_augment.py
├── utils/                  # Logger & metrics
│   ├── logger.py
│   └── metrics.py
├── data/                   # Datasets (gitignored, very large)
├── runs/                   # Training outputs (checkpoints)
│   ├── final_classifier.pt       # Full fine-tuned RETFound (~1.2 GB)
│   └── final_classifier.lora.pt  # LoRA adapter (~13 MB)
├── train.py                # Main training entry
├── evaluate.py             # Evaluation / metrics
├── gradcam.py              # Grad-CAM visualization
├── gradcam_5classes.py     # 5-class Grad-CAM
├── local_demo.py           # Local inference demo
├── prepare_data.py         # Generic data prep
├── prepare_ddr.py          # DDR dataset prep
├── prepare_ddr_real.py     # DDR real split
├── prepare_ddr_real_flat.py
├── prepare_ddr_for_automorph.py
├── fix_ddr.py              # Fix DDR label files
├── run_automorph_full.sh   # Pipeline script
├── requirements.txt
└── ALGO_README.md
`

## Trained model weights (tracked in git)

| File | Size | Description |
|------|------|-------------|
| uns/final_classifier.pt       | ~1.2 GB | Full fine-tuned RETFound ViT-B/16, 5-class DR grading |
| uns/final_classifier.lora.pt  | ~13 MB  | LoRA adapter (lighter alternative) |

Both were trained on DDR (DR_grading train split, 5 classes: 0=No DR / 1=Mild / 2=Moderate / 3=Severe / 4=Proliferative).

## Quick start

`ash
# install
pip install -r requirements.txt

# train
python train.py --config configs/retfound_b16.yaml

# evaluate
python evaluate.py --weights runs/final_classifier.pt

# visual explanation
python gradcam_5classes.py --weights runs/final_classifier.pt
`

## Notes

- data/ is git-ignored (datasets are too large).
- uns/ contains the committed model checkpoints.
- Original upstream RETFound repo: see https://github.com/rmaphoh/RETFound_MAE
