# Flow Matching on CelebA-HQ

Unconditional face generation using Conditional Flow Matching (CFM) — trained on CelebA-HQ 256 images resized to 64×64. Two backbones: a diffusers UNet and a DiT-style Vision Transformer.

| UNet (epoch 25) | ViT (epoch 100) |
|:-:|:-:|
| ![UNet samples](results/CFM_UNet_64x64_ep25_grid.png) | ![ViT samples](results/CFM_ViT_64x64_ep100_grid.png) |

---

## Setup

```bash
git clone https://github.com/harish-jhr/FlowMatching
cd FlowMatching
pip install torch torchvision diffusers accelerate tqdm Pillow numpy wandb
```

---

## Data

Point the training script at any flat folder of images. The dataloader resizes automatically. I used [CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):

```
/path/to/data/
├── 00001.jpg
├── 00002.jpg
└── ...
```

---

## Train

```bash
# UNet
python train.py --backbone unet --data_root /path/to/data --batch_size 128 --epochs 300

# ViT
python train.py --backbone vit --data_root /path/to/data --batch_size 64 --epochs 300
```

Checkpoints are saved to `./checkpoints/` after every epoch. Training is logged to WandB
Resume from a checkpoint:
```bash
python train.py --backbone unet --resume ./checkpoints/unet_latest.pt
```

---

## Sample

```bash
python sample.py --backbone unet --checkpoint ./checkpoints/unet_latest.pt
python sample.py --backbone vit  --checkpoint ./checkpoints/vit_latest.pt
```

Saves a grid of 64 images to `./generated/`. Use `--solver heun --steps 50` for faster sampling at similar quality.

---

## Files

| File | What it does |
|------|-------------|
| `flow_matching.py` | OT-CFM loss, Euler and Heun samplers |
| `model_unet.py` | Diffusers UNet2DModel wrapper |
| `model_vit.py` | DiT-style ViT with AdaLN time conditioning |
| `train.py` | Training loop with EMA, AMP, wandb logging |
| `sample.py` | Generate images from a saved checkpoint |

---

## Metrics

Evaluated on 3000 generated images (100 Euler steps) vs 3000 GT CelebA-HQ images. FID/KID use InceptionV3 features to compare real and generated distributions (lower FID indicates the generated images are statistically closer to real faces). LPIPS diversity measures how different generated images are from each other using VGG perceptual features on random pairs (higher means less mode collapse).

| Metric | UNet (ep 25) | ViT (ep 100) | Direction |
|--------|:------------:|:------------:|:---------:|
| FID | 20.93 | 22.64 | ↓ lower = better |
| KID | 0.0113 ± 0.0006 | 0.0116 ± 0.0006 | ↓ lower = better |
| IS | 2.11 ± 0.07 | 2.12 ± 0.06 | ↑ (unreliable for faces) |
| LPIPS diversity | 0.4422 | 0.4541 | ↑ higher = more diverse |

UNet converges faster (better FID at 4× fewer epochs). ViT shows slightly higher sample diversity at convergence.


## Reference

> Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
