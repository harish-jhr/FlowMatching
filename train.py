import argparse
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from flow_matching import cfm_loss, euler_sample
from model_unet import UNetFlow
from model_vit import DiTFlow


class ImageDataset(Dataset):
    exts = {".jpg", ".jpeg", ".png"}

    def __init__(self, root, transform=None):
        self.paths = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in self.exts)
        self.transform = transform
        print(f"Found {len(self.paths)} images in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.lerp_(p.data, 1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def get_loader(data_root, img_size, batch_size, num_workers):
    tf = transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = ImageDataset(data_root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0))


def build_model(backbone, img_size):
    if backbone == "unet":
        return UNetFlow(img_size=img_size)
    elif backbone == "vit":
        # patch=8 gives 64 patches at 64px — same count as DiT-S/8
        return DiTFlow(img_size=img_size, patch_size=4, dim=512, depth=8, heads=8)
    raise ValueError(backbone)


def log_samples(model, fixed_noise, epoch, step, img_size, sample_steps, device):
    model.eval()
    samples = euler_sample(model, fixed_noise, steps=sample_steps)
    samples = (samples.clamp(-1, 1) + 1) / 2

    grid = make_grid(samples, nrow=8)
    # convert to H W C uint8 for wandb
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    wandb.log({"samples": wandb.Image(grid_np, caption=f"epoch {epoch}")}, step=step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="unet", choices=["unet", "vit"])
    parser.add_argument("--data_root", default="/dir/DiffuseSeg/data/celebahq256_imgs/train") # using CelebA-HQ train split 27 thousabd instances
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma_min", type=float, default=0.001)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb_project", default="flow-matching")
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.backbone}_{args.img_size}px",
        config=vars(args),
    )

    model = build_model(args.backbone, args.img_size).to(device)
    print(f"{args.backbone} | {model.num_params() / 1e6:.1f}M params")

    ema = EMA(model, decay=args.ema_decay)
    ema.shadow.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # cosine decay 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    global_step = 0

    if args.resume:
        cpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(cpt["model"])
        ema.shadow.load_state_dict(cpt["ema"])
        optimizer.load_state_dict(cpt["optimizer"])
        if "scheduler" in cpt:
            scheduler.load_state_dict(cpt["scheduler"])
        start_epoch = cpt["epoch"] + 1
        global_step = cpt.get("step", 0)
        print(f"Resuing from epoch {start_epoch - 1}")

    loader = get_loader(args.data_root, args.img_size, args.batch_size, args.num_workers)

    # fixed noise for visual comparison across epochs on wandb
    fixed_noise = torch.randn(64, 3, args.img_size, args.img_size, device=device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")

        for x1 in pbar:
            x1 = x1.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = cfm_loss(model, x1, sigma_min=args.sigma_min)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wandb.log({"loss": loss.item(), "epoch": epoch}, step=global_step)

        epoch_loss = total_loss / len(loader)
        print(f"epoch {epoch} | loss {epoch_loss:.4f}")
        scheduler.step()
        wandb.log({
            "epoch_loss": epoch_loss,
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
        }, step=global_step)

        # sample with EMA model after every epoch
        log_samples(ema.shadow, fixed_noise, epoch, global_step,
                    args.img_size, args.sample_steps, device)
        model.train()

        torch.save({
            "epoch": epoch,
            "step": global_step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": epoch_loss,
        }, os.path.join(args.save_dir, f"{args.backbone}_latest.pt"))

    wandb.finish()


if __name__ == "__main__":
    main()