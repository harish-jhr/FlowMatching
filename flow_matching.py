import torch


def sample_xt(x0, x1, t, sigma_min=0.001):
    t = t.view(-1, 1, 1, 1)
    return (1 - (1 - sigma_min) * t) * x0 + t * x1


def target_ut(x0, x1, sigma_min=0.001):
    # velocity is constant along the path, no t dependence — that's the point of OT-CFM
    return x1 - (1 - sigma_min) * x0


def cfm_loss(model, x1, sigma_min=0.001):
    t = torch.rand(x1.shape[0], device=x1.device)
    x0 = torch.randn_like(x1)
    xt = sample_xt(x0, x1, t, sigma_min)
    ut = target_ut(x0, x1, sigma_min)
    return (model(xt, t) - ut).pow(2).mean()


@torch.no_grad()
def euler_sample(model, x0, steps=100):
    x = x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        x = x + dt * model(x, t)
    return x


@torch.no_grad()
def heun_sample(model, x0, steps=50):
    x = x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        t_next = torch.full((x.shape[0],), (i + 1) / steps, device=x.device)
        k1 = model(x, t)
        k2 = model(x + dt * k1, t_next)
        x = x + dt * 0.5 * (k1 + k2)
    return x
