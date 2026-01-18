#!/usr/bin/env python3
"""
Proof-of-concept: gradient inversion of burned-in text from FL-style client updates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

DEFAULT_DICOM = Path("sample_data/1-1.dcm")
DEFAULT_CROP = (1627, 0, 956, 559)  # top-right burned-in text region


def resolve_device(prefer_mps: bool) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_float(value: object, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def load_dicom_image(
    path: Path,
    crop: Optional[Tuple[int, int, int, int]],
    image_size: int,
    pctl_low: float,
    pctl_high: float,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"DICOM not found: {path}")

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]

    slope = _as_float(getattr(ds, "RescaleSlope", None), 1.0)
    intercept = _as_float(getattr(ds, "RescaleIntercept", None), 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        arr = arr.max() - arr

    if crop is not None:
        x, y, w, h = crop
        arr = arr[y : y + h, x : x + w]

    low = np.percentile(arr, pctl_low)
    high = np.percentile(arr, pctl_high)
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-8)

    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def save_image(arr: np.ndarray, path: Path) -> None:
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


class SimpleCNN(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            feat = self.features(dummy)
        self.classifier = nn.Linear(feat.numel(), 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def compute_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
) -> Tuple[Tuple[torch.Tensor, ...], float]:
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return grads, float(loss.item())


def grad_diff_loss(
    dummy_grads: Iterable[torch.Tensor],
    target_grads: Iterable[torch.Tensor],
) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(iter(dummy_grads)).device)
    for dg, tg in zip(dummy_grads, target_grads):
        total = total + (dg - tg).pow(2).sum()
    return total


def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
    tv_w = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean()
    return tv_h + tv_w


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FL-style gradient inversion PoC on burned-in text in DICOM images."
    )
    parser.add_argument(
        "--dicom-path",
        type=Path,
        default=DEFAULT_DICOM,
        help="Path to the DICOM with burned-in text.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=DEFAULT_CROP,
        metavar=("X", "Y", "W", "H"),
        help="Crop (x y w h) to isolate the burned-in text region.",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Resize to NxN.")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps.")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate.")
    parser.add_argument("--tv-weight", type=float, default=1e-6, help="TV regularizer weight.")
    parser.add_argument("--save-every", type=int, default=400, help="Save recon every N steps.")
    parser.add_argument("--pctl-low", type=float, default=1.0, help="Low percentile window.")
    parser.add_argument("--pctl-high", type=float, default=99.0, help="High percentile window.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--prefer-mps", action="store_true", help="Use MPS if available.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/latest"),
        help="Directory for images and report.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.prefer_mps)

    target = load_dicom_image(
        args.dicom_path,
        crop=tuple(args.crop) if args.crop else None,
        image_size=args.image_size,
        pctl_low=args.pctl_low,
        pctl_high=args.pctl_high,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(target, output_dir / "target.png")

    model = SimpleCNN(args.image_size).to(device)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device)
    y = torch.tensor([0], device=device, dtype=torch.long)

    exchange_dir = output_dir / "_fl_exchange"
    exchange_dir.mkdir(parents=True, exist_ok=True)

    # Server sends model weights to the client.
    server_weights = exchange_dir / "server_weights_round0.pt"
    torch.save(model.state_dict(), server_weights)

    client_model = SimpleCNN(args.image_size).to(device)
    client_model.load_state_dict(torch.load(server_weights, map_location=device))

    # Client computes gradients on its local image.
    target_grads, client_loss = compute_gradients(client_model, x, y, loss_fn)
    target_grads = tuple(g.detach().clone() for g in target_grads)

    client_update = exchange_dir / "client_update_round0.pt"
    torch.save({"grads": [g.detach().cpu() for g in target_grads]}, client_update)

    # Server receives the update and performs gradient inversion.
    update = torch.load(client_update, map_location=device)
    target_grads = tuple(g.to(device) for g in update["grads"])

    dummy = torch.rand_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([dummy], lr=args.lr)

    best_loss = None
    best_img = None

    progress = tqdm(range(args.steps), desc="Inversion", ncols=100)
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(dummy), y)
        dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        gd_loss = grad_diff_loss(dummy_grads, target_grads)
        tv = total_variation(dummy) * args.tv_weight
        total = gd_loss + tv
        total.backward()
        optimizer.step()

        with torch.no_grad():
            dummy.clamp_(0.0, 1.0)

        if best_loss is None or total.item() < best_loss:
            best_loss = total.item()
            best_img = dummy.detach().cpu().numpy()[0, 0].copy()

        if args.save_every and (step + 1) % args.save_every == 0:
            current = dummy.detach().cpu().numpy()[0, 0]
            save_image(current, output_dir / f"recon_step_{step+1}.png")

        progress.set_postfix(gd_loss=f"{gd_loss.item():.4e}", tv=f"{tv.item():.4e}")

    final = dummy.detach().cpu().numpy()[0, 0]
    save_image(final, output_dir / "recon_final.png")
    if best_img is not None:
        save_image(best_img, output_dir / "recon_best.png")

    mse = float(np.mean((final - target) ** 2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))

    report = [
        f"Target DICOM: {args.dicom_path}",
        f"Crop: {tuple(args.crop) if args.crop else None}",
        f"Device: {device}",
        f"Client loss: {client_loss:.6f}",
        f"Final MSE: {mse:.6e}",
        f"Final PSNR: {psnr:.2f} dB",
        f"Output dir: {output_dir}",
    ]
    (output_dir / "report.txt").write_text("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    main()
