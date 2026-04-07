import argparse
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from flow_matching import euler_sample
from model_unet import UNetFlow
from model_vit import DiTFlow


class ImageFolder(Dataset):
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root, transform=None):
        self.paths = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in self.exts)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def build_model(backbone, img_size, device):
    if backbone == "unet":
        m = UNetFlow(img_size=img_size)
    elif backbone == "vit":
        m = DiTFlow(img_size=img_size, patch_size=4, dim=512, depth=8, heads=8)
    else:
        raise ValueError(backbone)
    ckpt = torch.load(args.checkpoint, map_location=device)
    weights = ckpt.get("ema", ckpt.get("model", ckpt))
    m.load_state_dict(weights)
    return m.to(device).eval()


@torch.no_grad()
def generate_images(model, n, batch_size, img_size, steps, device, cache_path=None):
    # returns uint8 tensor (N, 3, H, W) in [0, 255] for torchmetrics
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached generated images")
        return torch.load(cache_path)

    all_imgs = []
    generated = 0
    pbar = tqdm(total=n, desc="generating")
    while generated < n:
        bs = min(batch_size, n - generated)
        x0 = torch.randn(bs, 3, img_size, img_size, device=device)
        imgs = euler_sample(model, x0, steps=steps)
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        imgs = (imgs * 255).byte()
        all_imgs.append(imgs.cpu())
        generated += bs
        pbar.update(bs)
    pbar.close()

    result = torch.cat(all_imgs, dim=0)
    if cache_path:
        torch.save(result, cache_path)
        print(f"Generated images at {cache_path}")
    return result


def load_real_images(data_root, img_size, batch_size, n_max, device):
    tf = transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    ds = ImageFolder(data_root, transform=tf)
    ds.paths = ds.paths[:n_max]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    return loader


def compute_fid_kid(real_loader, fake_imgs, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=1000, normalize=False).to(device)
    for batch in tqdm(real_loader):
        batch = (batch * 255).byte().to(device)
        fid.update(batch, real=True)
        kid.update(batch, real=True)
    gen_loader = DataLoader(fake_imgs, batch_size=256, shuffle=False)
    for batch in tqdm(gen_loader):
        batch = batch.to(device)
        fid.update(batch, real=False)
        kid.update(batch, real=False)

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    return fid_score, kid_mean.item(), kid_std.item()


def compute_is(fake_imgs, device):
    is_metric = InceptionScore(normalize=False).to(device)
    loader = DataLoader(fake_imgs, batch_size=256, shuffle=False)
    for batch in tqdm(loader, desc="IS"):
        is_metric.update(batch.to(device))
    mean, std = is_metric.compute()
    return mean.item(), std.item()


def compute_lpips_diversity(fake_imgs, n_pairs, device):
    n_pairs = min(n_pairs, len(fake_imgs) // 2)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    idx = torch.randperm(len(fake_imgs))[:n_pairs * 2]
    a = (fake_imgs[idx[:n_pairs]].float().div(255) * 2 - 1).to(device)
    b = (fake_imgs[idx[n_pairs:]].float().div(255) * 2 - 1).to(device)

    scores = []
    bs = 64
    for i in tqdm(range(0, n_pairs, bs), desc="LPIPS diversity"):
        scores.append(lpips(a[i:i+bs], b[i:i+bs]).item())
    return float(np.mean(scores))


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="unet", choices=["unet", "vit"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="/dir/DiffuseSeg/data/celebahq256_imgs/train")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_pairs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.backbone, args.img_size, device)
    print(f"Loaded {args.backbone} from {args.checkpoint}")

    cache_path = f"./eval_cache_{args.backbone}_{args.n_samples}samples_{args.steps}steps.pt"
    print(f"Generating {args.n_samples} images for metric calculation ")
    fake_imgs = generate_images(model, args.n_samples, args.batch_size,
                                args.img_size, args.steps, device, cache_path)

    real_loader = load_real_images(args.data_root, args.img_size,
                                   args.batch_size, args.n_samples, device)

    print("Computing FID adnd KID  :")
    fid, kid_mean, kid_std = compute_fid_kid(real_loader, fake_imgs, device)

    print("Computing Inception Score : ")
    is_mean, is_std = compute_is(fake_imgs, device)

    print("Computing LPIPS diversity : ")
    lpips_div = compute_lpips_diversity(fake_imgs, args.n_pairs, device)

    print(f"  FID              : {fid:.2f}")
    print(f"  KID              : {kid_mean:.4f} ± {kid_std:.4f}")
    print(f"  IS               : {is_mean:.2f} ± {is_std:.2f}")
    print(f"  LPIPS diversity  : {lpips_div:.4f}")


if __name__ == "__main__":
    main()
