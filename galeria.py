"""Galería final + prueba de vecinos más cercanos.   >>> PARTE DE LA PAREJA (ver INSTRUCCIONES_PAREJA.md) <<<

  python galeria.py candidates --run sn                      # 200 candidatos (semillas 0..199) -> figuras/candidatos.png
  python galeria.py gallery --run sn --seeds 3 17 42 ...     # galeria/ + figuras/vecinos_mas_cercanos.png
  python galeria.py gallery --run sn --seeds ... --names nombres.json   # (bonificación) agrega nombre/clase

La galería debe poder regenerarse EXACTAMENTE desde runs/<run>/G_final.pt + las semillas de galeria/galeria.json.
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
from evaluate import cnn_features   # features ResNet18 normalizadas L2 -> similitud coseno = producto punto

FIG = ROOT / "figuras"
GAL = ROOT / "galeria"
N_CANDIDATES = 200


def candidates(run):
    """Genera N_CANDIDATES personajes, uno por semilla 0..N_CANDIDATES-1 (z = z_from_seed(semilla)),
    y guarda una hoja de contacto en figuras/candidatos.png con el número de semilla sobre cada imagen.

    Pistas: G = load_generator(RUNS_DIR / run); usa torch.no_grad(); to_uint8(x) convierte el tensor
    [-1,1] a imágenes uint8 HxWx3; una rejilla de 10x20 con plt.subplots funciona bien (interpolation="nearest").
    """
    # TODO (pareja)
    raise NotImplementedError


def gallery(run, seeds, names=None):
    """1) Generar los personajes de `seeds` con el modelo `run`.
       2) Guardar en galeria/: para cada personaje i
            personaje_{i:02d}_semilla_{s}.png       (64x64, tal cual sale del modelo)
            personaje_{i:02d}_semilla_{s}_x4.png    (256x256, ampliado con Image.NEAREST, para verlo mejor)
            personaje_{i:02d}_semilla_{s}_z.npy     (el vector z de 100 dimensiones)
       3) Prueba de vecinos más cercanos contra los 3,648 datos de entrenamiento (TinyHero(augment=False).x):
            a) con features CNN: f = cnn_features(x); similitud = f_gen @ f_real.T; vecino = argmax
            b) en píxeles: torch.cdist(x_gen.flatten(1), x_real.flatten(1)); vecino = argmin
            c) referencia para interpretar: para cada imagen REAL, similitud con su vecina real más cercana
               (excluyéndose a sí misma) -> media, mediana y p95. Si un generado está MÁS cerca de su vecino que
               lo típico entre reales, es sospechoso de memorización.
       4) Figura figuras/vecinos_mas_cercanos.png: 3 filas x 10 columnas (generado / vecino CNN / vecino píxel),
          con la similitud o distancia en el título de cada imagen.
       5) galeria/galeria.json con: modelo, nz, tasa_seleccion ("10 de 200 ..."), criterio de selección,
          y por personaje: semilla, archivos, vecino CNN (ruta y coseno), vecino píxel (ruta y L2), y nombre/clase si hay `names`.
    """
    # TODO (pareja)
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["candidates", "gallery"])
    p.add_argument("--run", default="sn")
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--names", default=None, help="JSON: lista de {nombre, clase} en el mismo orden que --seeds")
    a = p.parse_args()
    if a.cmd == "candidates":
        candidates(a.run)
    else:
        gallery(a.run, a.seeds, json.load(open(a.names)) if a.names else None)
