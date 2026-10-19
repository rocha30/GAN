"""Galería final + prueba de vecinos más cercanos.

  python galeria.py candidates --run sn                      # 200 candidatos (semillas 0..199) -> figuras/candidatos.png
  python galeria.py gallery --run sn --seeds 3 17 42 ...     # galeria/ + figuras/vecinos_mas_cercanos.png
  python galeria.py gallery --run sn --seeds ... --names nombres.json   # (bonificación) agrega nombre/clase
  python galeria.py verify                                   # regenera desde galeria.json y compara bit a bit

La galería debe poder regenerarse EXACTAMENTE desde runs/<run>/G_final.pt + las semillas de galeria/galeria.json.
Cada personaje se genera por separado (lote de 1) y en modo eval(), así el resultado no depende de qué otras
semillas se generen junto a él.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gan import NZ, ROOT, RUNS_DIR, TinyHero, load_generator, to_uint8, z_from_seed
from evaluate import cnn_features, mean_pairwise   # features ResNet18 normalizadas L2 -> similitud coseno = producto punto

FIG = ROOT / "figuras"
GAL = ROOT / "galeria"
N_CANDIDATES = 200
CRITERIO = ("Revisamos a ojo los 200 candidatos (figuras/candidatos.png). Requisitos: (1) silueta humanoide legible "
            "(cabeza, torso, brazos y piernas) y (2) sin artefactos graves (manchas sueltas, cuerpo deshecho). Entre los que "
            "cumplen, elegimos 10 que (3) no se parezcan entre sí en color de cabello, vista (frente, espalda, perfil) ni ropa. "
            "La sugerencia automática greedy max-min de `galeria.py candidates` se descartó porque favorecía candidatos atípicos "
            "por tener artefactos; la distinción entre los 10 se verificó con el coseno ResNet18 entre pares (diversidad_galeria).")


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def generate_seeds(G, seeds):
    """Un z por semilla y un forward por personaje (lote de 1) -> resultado reproducible exactamente."""
    zs = torch.cat([z_from_seed(s) for s in seeds])
    with torch.no_grad():
        x = torch.cat([G(z.unsqueeze(0)) for z in zs])
    return zs, x


def greedy_maxmin(f, k, first=None):
    """Selección farthest-point: empieza por el candidato más "atípico" y agrega el más lejano (coseno) a los ya elegidos."""
    sim = f @ f.T
    first = int(sim.mean(1).argmin()) if first is None else first
    chosen = [first]
    while len(chosen) < k:
        closest = sim[:, chosen].max(1).values
        closest[chosen] = float("inf")
        chosen.append(int(closest.argmin()))
    return chosen


def candidates(run):
    """Genera N_CANDIDATES personajes (semillas 0..N_CANDIDATES-1), guarda la hoja de contacto figuras/candidatos.png
    con el número de semilla sobre cada imagen e imprime una sugerencia de los 10 más distintos (greedy max-min)."""
    FIG.mkdir(exist_ok=True)
    G = load_generator(RUNS_DIR / run)
    seeds = list(range(N_CANDIDATES))
    _, x = generate_seeds(G, seeds)
    imgs = to_uint8(x)

    nrow, ncol = 10, 20
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.05, nrow * 1.2))
    for ax, s in zip(axes.flat, seeds):
        ax.imshow(imgs[s], interpolation="nearest")
        ax.set_title(str(s), fontsize=8, pad=2)
        ax.axis("off")
    fig.suptitle(f"{N_CANDIDATES} candidatos del modelo '{run}' (semillas 0–{N_CANDIDATES - 1}, z = z_from_seed(semilla))",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "candidatos.png", dpi=110)
    plt.close(fig)

    f = cnn_features(x)
    sug = greedy_maxmin(f, 10)
    print(f"guardado {rel(FIG / 'candidatos.png')}")
    print(f"diversidad de los {N_CANDIDATES} candidatos (1 - coseno medio entre pares): {mean_pairwise(f):.3f}")
    print("sugerencia greedy max-min (10 más distintos en features ResNet18):", " ".join(map(str, sug)))


def _nn_reference(f_all, x_all, n):
    """Para cada imagen REAL i (de las n originales): vecina más cercana entre las demás imágenes reales, incluidas
    las volteadas, excluyendo a i y a su propia versión volteada (índices i y n+i). Devuelve cosenos y L2."""
    sim = f_all[:n] @ f_all.T
    idx = torch.arange(n)
    sim[idx, idx] = -1
    sim[idx, idx + n] = -1
    ref_cos = sim.max(1).values

    ref_l2 = torch.empty(n)
    flat = x_all.flatten(1)
    for i in range(0, n, 512):
        j = torch.arange(i, min(i + 512, n))
        d = torch.cdist(flat[j], flat)
        d[j - i, j] = float("inf")
        d[j - i, j + n] = float("inf")
        ref_l2[j] = d.min(1).values
    return ref_cos, ref_l2


def _stats(t):
    t = t.double()
    return {"media": round(t.mean().item(), 4), "mediana": round(t.median().item(), 4),
            "p5": round(t.quantile(0.05).item(), 4), "p95": round(t.quantile(0.95).item(), 4),
            "min": round(t.min().item(), 4), "max": round(t.max().item(), 4)}


def gallery(run, seeds, names=None, criterio=CRITERIO):
    """Genera la galería de `seeds` con el modelo `run`, la guarda en galeria/ y corre la prueba de vecinos más cercanos
    contra los 3,648 datos de entrenamiento (y sus versiones volteadas, porque se entrenó con volteo horizontal)."""
    assert seeds, "faltan --seeds"
    assert names is None or len(names) == len(seeds), "nombres.json debe tener una entrada por semilla"
    GAL.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    for old in GAL.glob("personaje_*"):   # evita restos de una selección anterior
        old.unlink()

    # 1) generar y guardar
    G = load_generator(RUNS_DIR / run)
    zs, x_gen = generate_seeds(G, seeds)
    imgs = to_uint8(x_gen)
    files = []
    for i, (s, im, z) in enumerate(zip(seeds, imgs, zs), 1):
        stem = GAL / f"personaje_{i:02d}_semilla_{s}"
        png, png4, npy = Path(f"{stem}.png"), Path(f"{stem}_x4.png"), Path(f"{stem}_z.npy")
        Image.fromarray(im).save(png)
        Image.fromarray(im).resize((256, 256), Image.NEAREST).save(png4)
        np.save(npy, z.numpy())
        files.append({"png": rel(png), "png_x4": rel(png4), "z": rel(npy)})

    # figura de la galería (2 x 5) para el notebook y la presentación
    ncol = min(5, len(seeds))
    nrow = -(-len(seeds) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (s, im) in enumerate(zip(seeds, imgs)):
        ax = axes.flat[i]
        ax.imshow(im, interpolation="nearest")
        t = f"#{i + 1} · semilla {s}"
        if names:
            t = f"{names[i]['nombre']}\n{names[i]['clase']}\n{t}"
        ax.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "galeria.png", dpi=130)
    plt.close(fig)

    # 2) vecinos más cercanos: originales + volteadas
    ds = TinyHero(augment=False)
    real = ds.x
    n = len(real)
    x_all = torch.cat([real, torch.flip(real, dims=[3])])
    f_all = cnn_features(x_all)
    f_gen = cnn_features(x_gen)

    sim = f_gen @ f_all.T
    cos, j_cnn = sim.max(1)
    dist = torch.cdist(x_gen.flatten(1), x_all.flatten(1))
    l2, j_pix = dist.min(1)

    ref_cos, ref_l2 = _nn_reference(f_all, x_all, n)

    def src(j):
        j = int(j)
        return rel(ds.paths[j % n]), bool(j >= n)

    # 3) figura estilo Goodfellow et al. (2014), fig. 2: generado / vecino CNN / vecino píxel
    k = len(seeds)
    show = to_uint8(torch.cat([x_gen, x_all[j_cnn], x_all[j_pix]]))
    fig, axes = plt.subplots(3, k, figsize=(1.9 * k, 7.2), squeeze=False)
    for c in range(k):
        label = f"#{c + 1} semilla {seeds[c]}"
        if names:
            label = f"{names[c]['nombre']}\n{label}"
        titles = [label,
                  f"cos = {cos[c]:.3f}" + (" (volt.)" if j_cnn[c] >= n else ""),
                  f"L2 = {l2[c]:.1f}" + (" (volt.)" if j_pix[c] >= n else "")]
        for r in range(3):
            ax = axes[r, c]
            ax.imshow(show[r * k + c], interpolation="nearest")
            ax.set_title(titles[r], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("generado", fontsize=10)
    axes[1, 0].set_ylabel("vecino CNN\n(ResNet18, coseno)", fontsize=10)
    axes[2, 0].set_ylabel("vecino en píxeles\n(L2)", fontsize=10)
    fig.suptitle(f"Prueba de vecinos más cercanos contra los {n} datos de entrenamiento (+ sus volteados). "
                 f"Referencia real→real: coseno medio {ref_cos.mean():.3f}, p95 {ref_cos.quantile(0.95):.3f}; "
                 f"L2 medio {ref_l2.mean():.1f}, p5 {ref_l2.quantile(0.05):.1f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "vecinos_mas_cercanos.png", dpi=130)
    plt.close(fig)

    # 4) galeria.json
    fs = f_gen @ f_gen.T
    off = fs[~torch.eye(k, dtype=torch.bool)]
    personajes = []
    for i in range(k):
        p_cnn, v_cnn = src(j_cnn[i])
        p_pix, v_pix = src(j_pix[i])
        p = {"n": i + 1, "semilla": int(seeds[i])}
        if names:
            p.update(nombre=names[i]["nombre"], clase=names[i]["clase"])
        p.update(archivos=files[i],
                 nn_cnn_ruta=p_cnn, nn_cnn_volteada=v_cnn, nn_cnn_cos=round(cos[i].item(), 4),
                 nn_pixel_ruta=p_pix, nn_pixel_volteada=v_pix, nn_pixel_l2=round(l2[i].item(), 3),
                 frac_reales_con_vecino_mas_cercano=round((ref_cos > cos[i]).float().mean().item(), 4),
                 frac_reales_con_vecino_mas_cercano_l2=round((ref_l2 < l2[i]).float().mean().item(), 4),
                 max_cos_con_otro_personaje=round(torch.cat([fs[i, :i], fs[i, i + 1:]]).max().item(), 4))
        personajes.append(p)

    meta = {
        "modelo": f"runs/{run}/G_final.pt",
        "nz": NZ,
        "generacion": "z = torch.randn(1, 100, generator=torch.Generator().manual_seed(semilla)); G.eval(); un forward por personaje",
        "n_candidatos": N_CANDIDATES,
        "semillas_candidatos": f"0..{N_CANDIDATES - 1}",
        "tasa_seleccion": f"{k} de {N_CANDIDATES} ({100 * k / N_CANDIDATES:.0f} %)",
        "criterio": criterio,
        "metodo_vecinos": (f"Búsqueda en los {n} datos de entrenamiento y sus {n} versiones volteadas horizontalmente "
                           "(el entrenamiento usó volteo). CNN: features de ResNet18 (ImageNet) normalizadas L2, coseno = "
                           "producto punto, vecino = argmax. Píxeles: distancia L2 en [-1, 1] sobre 3x64x64, vecino = argmin. "
                           "Referencia: para cada imagen real, su vecina real más cercana excluyéndose a sí misma y a su volteada."),
        "referencia_real_nn_cos": _stats(ref_cos),
        "referencia_real_nn_l2": _stats(ref_l2),
        "diversidad_galeria": {"1_menos_coseno_medio_entre_pares": round(1 - off.mean().item(), 4),
                               "coseno_max_entre_dos_personajes": round(off.max().item(), 4),
                               "referencia_reales_1_menos_coseno_medio": 0.2743},
        "personajes": personajes,
    }
    with open(GAL / "galeria.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"Referencia real→real: coseno {meta['referencia_real_nn_cos']}  L2 {meta['referencia_real_nn_l2']}")
    for p in personajes:
        print(f"#{p['n']:2d} semilla {p['semilla']:3d}  cos={p['nn_cnn_cos']:.3f} ({p['nn_cnn_ruta']}"
              f"{', volt.' if p['nn_cnn_volteada'] else ''})  L2={p['nn_pixel_l2']:.1f}  "
              f"reales con vecino más parecido: {100 * p['frac_reales_con_vecino_mas_cercano']:.0f} %  "
              f"máx. coseno con otro personaje: {p['max_cos_con_otro_personaje']:.3f}")
    print(f"diversidad de la galería: {meta['diversidad_galeria']}")
    print(f"guardado {rel(GAL / 'galeria.json')} y {rel(FIG / 'vecinos_mas_cercanos.png')}")


def verify():
    """Regenera cada personaje desde el modelo y la semilla de galeria/galeria.json y lo compara bit a bit
    con el PNG y el z guardados."""
    meta = json.load(open(GAL / "galeria.json", encoding="utf-8"))
    G = load_generator(ROOT / Path(meta["modelo"]).parent)
    seeds = [p["semilla"] for p in meta["personajes"]]
    zs, x = generate_seeds(G, seeds)
    imgs = to_uint8(x)
    ok = True
    for p, im, z in zip(meta["personajes"], imgs, zs):
        same_png = np.array_equal(np.asarray(Image.open(ROOT / p["archivos"]["png"])), im)
        same_z = np.array_equal(np.load(ROOT / p["archivos"]["z"]), z.numpy())
        ok &= same_png and same_z
        print(f"#{p['n']:2d} semilla {p['semilla']:3d}  png idéntico: {same_png}  z idéntico: {same_z}")
    print("GALERÍA REGENERADA EXACTAMENTE" if ok else "¡DIFERENCIAS!")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["candidates", "gallery", "verify"])
    p.add_argument("--run", default="sn")
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--names", default=None, help="JSON: lista de {nombre, clase} en el mismo orden que --seeds")
    p.add_argument("--criterio", default=CRITERIO, help="criterio de selección que se guarda en galeria.json")
    a = p.parse_args()
    if a.cmd == "candidates":
        candidates(a.run)
    elif a.cmd == "gallery":
        gallery(a.run, a.seeds, json.load(open(a.names, encoding="utf-8")) if a.names else None, a.criterio)
    else:
        raise SystemExit(0 if verify() else 1)
