"""Proyecto 2 - Diseño generativo de personajes (Deep Learning 2026).

DCGAN 64x64 para el universo "Reinos de Pixelia" usando el dataset TinyHero.
Un solo script configurable para que los experimentos cambien UNA cosa a la vez:

  --loss bce    pérdida no saturante (BCE)              [base]
  --loss hinge  pérdida hinge                            [experimento de pérdida]
  --sn          normalización espectral en D             [experimento de estabilización 1]
  --mbstd       capa minibatch-std en D (anti colapso)   [experimento de estabilización 2]
  --inst_noise  ruido gaussiano en entradas de D         [experimento de estabilización 3]
  --lrD         lr distinta para D (TTUR)                [experimento de estabilización 4]

Todo lo demás (datos, semilla, épocas, lr, ruido de evaluación) es idéntico entre corridas.

Uso:
  python gan.py train --name base --loss bce --epochs 60
  python gan.py train --name hinge --loss hinge --epochs 60
  python gan.py train --name sn --loss bce --sn --epochs 60
  python gan.py train --name mbstd --loss bce --mbstd --epochs 60
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "tinyhero" / "data"
RUNS_DIR = ROOT / "runs"
NZ = 100
SEED = 2026          # semilla global de entrenamiento (igual en todas las corridas)
FIXED_SEED = 7       # semilla del ruido fijo de evaluación (16 vectores z)


def get_device():
    if os.environ.get("GAN_DEVICE"):
        return torch.device(os.environ["GAN_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- datos
def load_rgba_on_white(path):
    """Las imágenes de TinyHero son RGBA con fondo transparente: las componemos sobre blanco."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def list_images():
    return sorted(DATA_DIR.glob("*/*.png"), key=lambda p: (p.parent.name, int(p.stem)))


class TinyHero(Dataset):
    """Carga todo en memoria (3648 imágenes de 64x64) normalizado a [-1, 1].
    Aumento de datos: volteo horizontal aleatorio (p=0.5)."""

    def __init__(self, augment=True):
        self.paths = list_images()
        arr = np.stack([np.asarray(load_rgba_on_white(p), dtype=np.uint8) for p in self.paths])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        self.x = x
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        if self.augment and random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        return img


# ----------------------------------------------------------------------------- modelos
class Generator(nn.Module):
    """DCGAN: z (100) -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64x3, BatchNorm + ReLU, salida Tanh."""

    def __init__(self, nz=NZ, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class MinibatchStd(nn.Module):
    """Minibatch standard deviation (Karras et al., ProGAN 2018): agrega un canal con la desviación estándar
    media entre las muestras del lote. Si G colapsa (todas las muestras iguales), D lo detecta inmediatamente."""

    def forward(self, x):
        s = x.std(0, unbiased=False).mean().view(1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, s], 1)


class Discriminator(nn.Module):
    """DCGAN: 64x64x3 -> 1 logit.
    --sn     reemplaza BatchNorm por normalización espectral en todas las convoluciones.
    --mbstd  agrega una capa minibatch-std antes de la última convolución."""

    def __init__(self, ndf=64, sn=False, mbstd=False):
        super().__init__()
        SN = nn.utils.spectral_norm if sn else (lambda m: m)
        norm = (lambda c: nn.Identity()) if sn else nn.BatchNorm2d
        self.net = nn.Sequential(
            SN(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)), norm(ndf * 2), nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)), norm(ndf * 4), nn.LeakyReLU(0.2, True),
            SN(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)), norm(ndf * 8), nn.LeakyReLU(0.2, True),
            MinibatchStd() if mbstd else nn.Identity(),
            SN(nn.Conv2d(ndf * 8 + int(mbstd), 1, 4, 1, 0, bias=False)),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    name = m.__class__.__name__
    if "Conv" in name:
        w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
        nn.init.normal_(w, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------- pérdidas
def d_loss_fn(kind, real_logits, fake_logits):
    if kind == "bce":
        return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
    if kind == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    raise ValueError(kind)


def g_loss_fn(kind, fake_logits):
    if kind == "bce":  # no saturante: max log D(G(z))
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    if kind == "hinge":
        return -fake_logits.mean()
    raise ValueError(kind)


# ----------------------------------------------------------------------------- utilidades
def fixed_noise(n=16, device="cpu"):
    g = torch.Generator().manual_seed(FIXED_SEED)
    return torch.randn(n, NZ, generator=g).to(device)


def z_from_seed(seed):
    """Cada personaje de la galería se regenera a partir de una semilla entera."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(1, NZ, generator=g)


def to_uint8(x):
    x = (x.clamp(-1, 1) + 1) * 127.5
    return x.round().byte().permute(0, 2, 3, 1).cpu().numpy()


def save_grid(x, path, nrow=8, pad=2):
    imgs = to_uint8(x)
    n, h, w, _ = imgs.shape
    ncol = int(np.ceil(n / nrow))
    grid = np.full((ncol * (h + pad) + pad, nrow * (w + pad) + pad, 3), 255, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        grid[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = im
    Image.fromarray(grid).save(path)


def load_generator(run_dir, device="cpu", ckpt="G_final.pt"):
    G = Generator().to(device)
    G.load_state_dict(torch.load(Path(run_dir) / ckpt, map_location=device))
    G.eval()
    return G


# ----------------------------------------------------------------------------- entrenamiento
def train(args):
    device = get_device()
    set_seed(SEED)
    out = RUNS_DIR / args.name
    (out / "fixed").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    ds = TinyHero(augment=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(SEED))
    G = Generator().to(device)
    D = Discriminator(sn=args.sn, mbstd=args.mbstd).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lrD or args.lr, betas=(0.5, 0.999))
    zfix = fixed_noise(16, device)

    log = {"iter": [], "epoch": [], "lossD": [], "lossG": [], "D_real": [], "D_fake": []}
    start_epoch = 0
    last = out / "ckpt" / "last.pt"
    if last.exists():  # reanudar si la sesión se interrumpió
        st = torch.load(last, map_location=device)
        G.load_state_dict(st["G"]); D.load_state_dict(st["D"])
        optG.load_state_dict(st["optG"]); optD.load_state_dict(st["optD"])
        log, start_epoch = st["log"], st["epoch"] + 1
        print(f"Reanudando desde época {start_epoch}")

    with torch.no_grad():
        G.eval(); save_grid(G(zfix), out / "fixed" / "epoch_000.png", nrow=8); G.train()

    it = len(log["iter"])
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        for real in dl:
            real = real.to(device)
            b = real.size(0)
            # --- D
            z = torch.randn(b, NZ, device=device)
            fake = G(z)
            # ruido de instancia en las entradas de D (decae linealmente a 0)
            sigma = args.inst_noise * max(0.0, 1 - epoch / args.epochs)
            nz_ = (lambda t: t + sigma * torch.randn_like(t)) if sigma > 0 else (lambda t: t)
            rl, fl = D(nz_(real)), D(nz_(fake.detach()))
            lossD = d_loss_fn(args.loss, rl, fl)
            optD.zero_grad(set_to_none=True); lossD.backward(); optD.step()
            # --- G
            fl2 = D(nz_(fake))
            lossG = g_loss_fn(args.loss, fl2)
            optG.zero_grad(set_to_none=True); lossG.backward(); optG.step()

            log["iter"].append(it); log["epoch"].append(epoch)
            log["lossD"].append(lossD.item()); log["lossG"].append(lossG.item())
            log["D_real"].append(torch.sigmoid(rl).mean().item())
            log["D_fake"].append(torch.sigmoid(fl).mean().item())
            it += 1
        ep = epoch + 1
        if ep in args.snap_epochs or ep == args.epochs:
            with torch.no_grad():
                G.eval(); save_grid(G(zfix), out / "fixed" / f"epoch_{ep:03d}.png", nrow=8); G.train()
        if ep % args.ckpt_every == 0 or ep == args.epochs:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(),
                        "optD": optD.state_dict(), "log": log, "epoch": epoch}, last)
            torch.save(G.state_dict(), out / "ckpt" / f"G_epoch_{ep:03d}.pt")
        n = len(dl)
        print(f"[{args.name}] epoch {ep}/{args.epochs} lossD={np.mean(log['lossD'][-n:]):.3f} "
              f"lossG={np.mean(log['lossG'][-n:]):.3f} D(x)={np.mean(log['D_real'][-n:]):.2f} "
              f"D(G(z))={np.mean(log['D_fake'][-n:]):.2f} t={time.time()-t0:.0f}s", flush=True)

    torch.save(G.state_dict(), out / "G_final.pt")
    json.dump(log, open(out / "log.json", "w"))
    print("listo", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--loss", choices=["bce", "hinge"], default="bce")
    t.add_argument("--sn", action="store_true")
    t.add_argument("--mbstd", action="store_true")
    t.add_argument("--inst_noise", type=float, default=0.0, help="sigma inicial del ruido en entradas de D")
    t.add_argument("--lrD", type=float, default=None, help="lr de D (por defecto = --lr)")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--ckpt_every", type=int, default=10)
    t.add_argument("--snap_epochs", type=int, nargs="*", default=[1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100])
    a = p.parse_args()
    if a.cmd == "train":
        train(a)
