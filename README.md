# Gradient Inversion PHI PoC (Federated Learning)

This repository contains a minimal, reproducible proof-of-concept showing that
burned-in text on medical images can be reconstructed from a single federated
learning (FL) client update.

The demo uses a DICOM image with synthetic burned-in text and performs gradient
inversion on the server side to recover the text.

## Quickstart

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Run the PoC (uses MPS if available on Apple silicon)
.venv/bin/python phi_gradient_inversion_poc.py --prefer-mps
```

Outputs are written to `outputs/latest`:
- `target.png` (the ground-truth cropped text)
- `recon_step_*.png` (intermediate reconstructions)
- `recon_best.png` and `recon_final.png`
- `report.txt`

In the default example, `recon_step_800.png` typically shows a readable
reconstruction of the phrase **"SEMIERECT PORTABLE"** (synthetic text).

## What to Run (Reproducible Example)

The defaults are already set to a known DICOM and crop that contains the burned-in
text:

```bash
.venv/bin/python phi_gradient_inversion_poc.py \
  --dicom-path "dataset_cancer_imaging_archive/manifest-1740445452889/MIDI-B-Synthetic-Validation/5864436320/08-17-2011-NA-XR CHEST 1 VIEW AP at DC-30150/1.000000-AP for Daniel Hoover-74781/1-1.dcm" \
  --crop 1627 0 956 559 \
  --image-size 512 \
  --steps 2000 \
  --save-every 400 \
  --tv-weight 1e-6 \
  --prefer-mps
```
