import argparse
import os

import torch
from torchvision.utils import save_image

from flow_matching import euler_sample, heun_sample
from model_unet import UNetFlow
from model_vit import DiTFlow


def build_model(backbone, img_size):
    if backbone == "unet":
        return UNetFlow(img_size=img_size)
    elif backbone == "vit":
        return DiTFlow(img_size=img_size, patch_size=4, dim=512, depth=8, heads=8)
    raise ValueError(backbone)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="unet", choices=["unet", "vit"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--solver", default="euler", choices=["euler", "heun"])
    parser.add_argument("--out", default="./generated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = build_model(args.backbone, args.img_size).to(device)

    cpt = torch.load(args.checkpoint, map_location=device)
    weights = cpt.get("ema", cpt.get("model", cpt))
    model.load_state_dict(weights)
    model.eval()

    x0 = torch.randn(args.n_samples, 3, args.img_size, args.img_size, device=device)
    sampler = euler_sample if args.solver == "euler" else heun_sample

    print(f"Sampling {args.n_samples} images ({args.solver}, {args.steps} steps)")
    samples = sampler(model, x0, steps=args.steps)
    samples = (samples.clamp(-1, 1) + 1) / 2

    nrow = int(args.n_samples ** 0.5)
    out_path = os.path.join(args.out, f"{args.backbone}_{args.solver}_{args.steps}steps.png")
    save_image(samples, out_path, nrow=nrow, padding=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
