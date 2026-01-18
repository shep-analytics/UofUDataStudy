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

The zip includes a single DICOM at `sample_data/1-1.dcm` with synthetic burned-in
text, so the defaults work as-is. You can override them if needed:

```bash
.venv/bin/python phi_gradient_inversion_poc.py \
  --dicom-path "sample_data/1-1.dcm" \
  --crop 1627 0 956 559 \
  --image-size 512 \
  --steps 2000 \
  --save-every 400 \
  --tv-weight 1e-6 \
  --prefer-mps
```

## Why This Is a Solid PoC

- **Directly models the FL threat:** a server receives a client update and runs
  gradient inversion to reconstruct the client image.
- **Single-step gradient leakage:** this is the highest-risk, well-studied case
  and demonstrates that raw updates can expose visual PHI.
- **Readable text recovered:** the reconstruction produces legible burned-in
  words, which is the concrete privacy failure this PoC is designed to show.

## Notes and Limitations

- This is a *proof-of-concept*, not a guarantee of success in every FL setup.
- Real deployments may use mitigations such as secure aggregation, gradient
  clipping, or differential privacy, which can reduce or prevent leakage.
- The included DICOM is from TCIA's MIDI-B-Synthetic-Validation collection and
  contains **synthetic** burned-in text (DOI: https://doi.org/10.7937/cf2paw56).

If you want this extended to multiple images, FedAvg-style updates, or mitigation
experiments, just ask.
# UofUDataStudy
