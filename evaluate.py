"""Evaluación de los experimentos.

  python evaluate.py figures   # curvas de pérdida, rejillas de ruido fijo, muestras y métricas por corrida

`cnn_features` (features de ResNet18 preentrenada) se puede reutilizar en galeria.py para la prueba de vecinos.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from gan import NZ, ROOT, RUNS_DIR, TinyHero, load_generator, to_uint8

FIG = ROOT / "figuras"
RUNS = ["base", "hinge", "sn", "mbstd", "noise", "ttur", "sn_bias"]   # todas comparadas en la época 10
LABEL = {"base": "Base: DCGAN + BCE no saturante",
         "hinge": "Exp. pérdida: hinge",
         "sn": "Exp. estabilización 1: spectral norm en D",
         "mbstd": "Exp. estabilización 2: minibatch-std en D",
         "noise": "Exp. estabilización 3: ruido en entradas de D",
         "ttur": "Exp. estabilización 4: lr de D = lr de G / 4",
         "sn_bias": "Diagnóstico: spectral norm + sesgos en D",
         "wgangp": "Modelo final: WGAN-GP"}
CMP_CKPT = "ckpt/G_epoch_010.pt"   # punto de comparación común (mismas iteraciones para todos)


# ----------------------------------------------------------------------------- features CNN
_cnn = None


def cnn_features(x, bs=256):
    """x en [-1,1], Nx3x64x64 -> features ResNet18 (ImageNet) de 512-d, normalizadas L2."""
    global _cnn
    if _cnn is None:
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        m.fc = torch.nn.Identity()
        _cnn = m.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            b = (x[i:i + bs].cpu() + 1) / 2
            b = F.interpolate(b, size=128, mode="nearest")
            feats.append(_cnn((b - mean) / std))
    return F.normalize(torch.cat(feats), dim=1)


def frechet(f1, f2):
    f1, f2 = f1.double().numpy(), f2.double().numpy()
    mu1, mu2 = f1.mean(0), f2.mean(0)
    c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    # sqrt(c1 c2) vía eigendecomposición de la matriz simétrica sqrt(c1) c2 sqrt(c1)
    w, v = np.linalg.eigh(c1)
    s1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    w2 = np.linalg.eigvalsh(s1 @ c2 @ s1)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * np.sqrt(np.clip(w2, 0, None)).sum())


def mean_pairwise(f):
    sim = f @ f.T
    n = len(f)
    return float((1 - (sim.sum() - n) / (n * (n - 1))).item())


def background_and_spread(x):
    """Fondo medio (borde de 4 px; en los datos reales es 1.0 = blanco) y desviación estándar por píxel entre muestras."""
    border = torch.ones(64, 64, dtype=torch.bool)
    border[4:-4, 4:-4] = False
    return float(x[:, :, border].mean()), float(x.std(0).mean())


def homogeneity_check(seed=0):
    """D sin BatchNorm y sin sesgos (conv + LeakyReLU) es positivamente homogénea: D(c·x) = c·D(x) para c > 0.
    Se comprueba numéricamente con pesos aleatorios: con sesgos, la relación deja de cumplirse."""
    from gan import Discriminator
    torch.manual_seed(seed)
    x = torch.randn(8, 3, 64, 64) * 0.5
    out = {}
    for bias in (False, True):
        D = Discriminator(sn=True, bias=bias)
        with torch.no_grad():
            if bias:
                for m in D.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.uniform_(m.bias, -0.1, 0.1)
            for _ in range(30):   # converge la iteración de potencia de la normalización espectral
                D(x)
            D.eval()
            d1, d2, d0 = D(x), D(0.25 * x), D(torch.zeros(1, 3, 64, 64))
        out["con_sesgos" if bias else "sin_sesgos"] = {"D(x)": round(float(d1.mean()), 4), "D(0.25x)": round(float(d2.mean()), 4),
                                                        "D(0.25x)/D(x)": round(float((d2 / d1).mean()), 4),
                                                        "D(0)": round(float(d0), 4)}
    return out


def generate(G, n, seed=123):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, NZ, generator=g)
    with torch.no_grad():
        return torch.cat([G(z[i:i + 250]) for i in range(0, n, 250)])


# ----------------------------------------------------------------------------- figuras
def smooth(y, k=25):
    y = np.asarray(y)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def figures(final=None):
    FIG.mkdir(exist_ok=True)
    logs = {r: json.load(open(RUNS_DIR / r / "log.json")) for r in RUNS}

    # 1) curvas de pérdida y salidas de D (una columna por corrida)
    fig, axes = plt.subplots(2, len(RUNS), figsize=(4.0 * len(RUNS), 7), sharex=False)
    for j, r in enumerate(RUNS):
        L = logs[r]
        ax = axes[0, j]
        ax.plot(L["lossD"], alpha=0.15, color="C0"); ax.plot(smooth(L["lossD"]), color="C0", label="pérdida D")
        ax.plot(L["lossG"], alpha=0.15, color="C3"); ax.plot(smooth(L["lossG"]), color="C3", label="pérdida G")
        ax.set_title(LABEL[r], fontsize=9); ax.legend(); ax.grid(alpha=0.3)
        ax = axes[1, j]
        ax.plot(smooth(L["D_real"]), label="D(x) real"); ax.plot(smooth(L["D_fake"]), label="D(G(z)) falso")
        ax.set_ylim(0, 1); ax.set_xlabel("iteración"); ax.legend(); ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("pérdida"); axes[1, 0].set_ylabel("sigmoid(logit) medio")
    fig.tight_layout(); fig.savefig(FIG / "curvas_perdida.png", dpi=130); plt.close(fig)

    # 1b) curvas de los modelos finales WGAN-GP: el crítico no tiene sigmoide, se grafica la distancia de Wasserstein estimada
    for r in [f for f in final if json.load(open(RUNS_DIR / f / "config.json"))["loss"] == "wgangp"]:
        L = json.load(open(RUNS_DIR / r / "log.json"))
        logs[r] = L
        w = np.asarray(L["C_real"]) - np.asarray(L["C_fake"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        ax = axes[0]
        ax.plot(L["lossD"], alpha=0.15, color="C0"); ax.plot(smooth(L["lossD"]), color="C0", label="pérdida del crítico (−W + 10·GP)")
        ax.plot(L["lossG"], alpha=0.15, color="C3"); ax.plot(smooth(L["lossG"]), color="C3", label="pérdida de G (−C(G(z)))")
        ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("paso de G"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"{LABEL[r]}: pérdidas", fontsize=10)
        ax = axes[1]
        ax.plot(w, alpha=0.15, color="C2"); ax.plot(smooth(w), color="C2", label="W estimada = C(x) − C(G(z))")
        ax.set_xlabel("paso de G"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("Distancia de Wasserstein estimada (debe bajar si G se acerca a los datos)", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / f"curvas_{r}.png", dpi=130); plt.close(fig)

    # 2) rejilla de ruido fijo: filas = época, columnas = los 16 mismos z
    for r in RUNS + [f for f in final if f not in RUNS]:
        snaps = sorted((RUNS_DIR / r / "fixed").glob("epoch_*.png"))
        rows = []
        for s in snaps:
            g = np.asarray(Image.open(s))
            # la rejilla guardada es 2x8 con pad 2 -> la convertimos en 1x16
            tiles = [g[2 + rr * 66: 2 + rr * 66 + 64, 2 + cc * 66: 2 + cc * 66 + 64] for rr in range(2) for cc in range(8)]
            rows.append(np.concatenate([np.pad(t, ((1, 1), (1, 1), (0, 0)), constant_values=255) for t in tiles], 1))
        img = np.concatenate(rows, 0)
        fig, ax = plt.subplots(figsize=(16, 1.05 * len(snaps)))
        ax.imshow(img, interpolation="nearest")
        ax.set_yticks([33 + 66 * i for i in range(len(snaps))])
        ax.set_yticklabels([f"época {int(s.stem.split('_')[1])}" for s in snaps])
        ax.set_xticks([33 + 66 * i for i in range(16)]); ax.set_xticklabels([f"z{i}" for i in range(16)], fontsize=7)
        ax.set_title(f"Rejilla de ruido fijo (16 vectores z, semilla 7) — {LABEL[r]}")
        fig.tight_layout(); fig.savefig(FIG / f"ruido_fijo_{r}.png", dpi=110); plt.close(fig)

    # 3) métricas: FID con features de ResNet18 (no comparable con el FID estándar) y diversidad
    real = TinyHero(augment=False).x
    f_real = cnn_features(real)
    sub = torch.randperm(len(f_real), generator=torch.Generator().manual_seed(0))[:1000]
    bg, sp = background_and_spread(real[sub])
    metrics = {"real": {"diversidad": mean_pairwise(f_real[sub]), "fondo_medio": bg, "desv_entre_muestras": sp},
               "homogeneidad_D": homogeneity_check()}
    for r in RUNS + [f + "_final" for f in final]:
        G = load_generator(RUNS_DIR / r.replace("_final", ""), ckpt="G_final.pt" if r.endswith("_final") else CMP_CKPT)
        fake = generate(G, 1000)
        f_fake = cnn_features(fake)
        bg, sp = background_and_spread(fake)
        metrics[r] = {"fid_resnet18": frechet(f_real, f_fake), "diversidad": mean_pairwise(f_fake),
                      "fondo_medio": bg, "desv_entre_muestras": sp}
        # D colapsado a cero durante muchas épocas -> alarma
        L = logs[r.replace("_final", "")]
        metrics[r]["frac_iters_lossD_lt_0.05"] = (None if "wgangp" in r  # en WGAN-GP la pérdida del crítico es negativa por diseño
                                                  else float(np.mean(np.asarray(L["lossD"]) < 0.05)))
        save_samples = to_uint8(fake[:64])
        grid = np.concatenate([np.concatenate(list(save_samples[i * 8:(i + 1) * 8]), 1) for i in range(8)], 0)
        Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(FIG / f"muestras_{r}.png")
    # muestras reales para comparar
    idx = torch.randperm(len(real), generator=torch.Generator().manual_seed(1))[:64]
    rs = to_uint8(real[idx])
    grid = np.concatenate([np.concatenate(list(rs[i * 8:(i + 1) * 8]), 1) for i in range(8)], 0)
    Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(FIG / "muestras_reales.png")
    json.dump(metrics, open(FIG / "metricas.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["figures"])
    p.add_argument("--final", nargs="*", default=[], help="corridas evaluadas también con G_final.pt (p. ej. ttur wgangp)")
    a = p.parse_args()
    figures(a.final)
