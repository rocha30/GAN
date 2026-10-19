# Proyecto 2: Diseño generativo de personajes, *Reinos de Pixelia*

GAN de 64×64 entrenada con el dataset TinyHero (pixel art LPC, con licencia CC-BY-SA 3.0 / GPL 3.0) para generar personajes nuevos de un RPG de fantasía.
Los experimentos de pérdida y estabilización se hicieron sobre una DCGAN; todas sus variantes colapsaron, diagnosticamos la causa y el **modelo final es una WGAN-GP** (`runs/wgangp/`).

## Estructura
| Archivo / carpeta | Contenido |
|---|---|
| `proyecto2_gan.ipynb` | Notebook principal: universo, datos, hipótesis previas, evidencias, galería y análisis |
| `gan.py` | Datos, generador, discriminador, pérdidas (BCE no saturante / hinge / WGAN-GP), normalización espectral, sesgos en D, minibatch-std, ruido de instancia, TTUR y entrenamiento con checkpoints |
| `evaluate.py` | Curvas de pérdida, rejillas de ruido fijo, muestras y métricas (FID con features de ResNet18, diversidad, fondo medio, desviación entre muestras y prueba de homogeneidad de D) |
| `galeria.py` | 200 candidatos, galería final, prueba de vecinos más cercanos y `verify` (comprueba bit a bit que la galería se regenera) |
| `nombres.json` | Nombre y clase de cada personaje (bonificación), en el orden de las semillas |
| `analisis.py` | Textos de interpretación compartidos por el notebook y la presentación |
| `make_notebook.py`, `make_presentacion.py` | Generan `proyecto2_gan.ipynb` y `presentacion.pdf` a partir de los resultados |
| `runs/<corrida>/` | `G_final.pt`, `log.json`, `config.json`, `fixed/` (rejillas de ruido fijo), `ckpt/` (checkpoints) |
| `figuras/` | Figuras usadas en la presentación |
| `galeria/` | Los 10 personajes en PNG (64×64 y ampliados ×4), el vector z de cada uno (`.npy`) y `galeria.json` (semillas, tasa de selección y vecinos) |
| `presentacion.pdf` | Informe (máximo 12 diapositivas + anexos) |
| `INSTRUCCIONES_PAREJA.md` | División del trabajo y resultados de la parte 1 |

## Cómo ejecutar
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy matplotlib pillow nbformat jupyter

# 1) Datos (≈7 MB)
mkdir -p data && curl -L -o data/tinyhero.zip https://github.com/AgaMiko/pixel_character_generator/raw/master/data.zip
unzip -q data/tinyhero.zip -d data/tinyhero          # -> data/tinyhero/data/{0,1,2,3}/*.png

# 2) Entrenamiento (6 corridas; ~15 s/época en un Apple M3 con MPS; también funciona en CUDA)
S="--ckpt_every 5 --snap_epochs 1 2 5 10 15 20"
python gan.py train --name base  --loss bce            --epochs 10 $S
python gan.py train --name hinge --loss hinge          --epochs 10 $S
python gan.py train --name sn    --loss bce --sn       --epochs 10 $S
python gan.py train --name mbstd --loss bce --mbstd    --epochs 20 $S
python gan.py train --name noise --loss bce --inst_noise 0.1 --epochs 15 $S
python gan.py train --name ttur  --loss bce --lrD 5e-5 --epochs 15 $S
python gan.py train --name sn_bias --loss bce --sn --dbias --epochs 10 $S      # diagnóstico (parte 2)
# modelo final: WGAN-GP (~2 h en CPU, 8 hilos)
python gan.py train --name wgangp --loss wgangp --ngf 32 --ndf 32 --lr 1e-4 --beta1 0.0 --beta2 0.9        --epochs 270 --ckpt_every 10 --snap_epochs 1 5 10 25 50 100 150 200 270
# (en zsh usa ${=S} en lugar de $S)

# 3) Figuras y métricas (las corridas DCGAN se comparan en el checkpoint de la época 10)
python evaluate.py figures --final ttur wgangp

# 4) Galería: regenera EXACTAMENTE los 10 personajes desde runs/wgangp/G_final.pt y las semillas de galeria/galeria.json
python galeria.py candidates --run wgangp        # figuras/candidatos.png (semillas 0..199)
python galeria.py gallery --run wgangp --seeds $(python -c "import json;print(*[p['semilla'] for p in json.load(open('galeria/galeria.json',encoding='utf-8'))['personajes']])") --names nombres.json
python galeria.py verify                         # compara bit a bit los PNG y z regenerados con los guardados

# 5) Notebook y presentación
python make_notebook.py && jupyter nbconvert --to notebook --execute --inplace proyecto2_gan.ipynb
python make_presentacion.py                      # presentacion.pdf
```
Si se interrumpe, `gan.py train` reanuda desde `runs/<nombre>/ckpt/last.pt`. Los pesos `runs/*/G_final.pt` (≈14 MB c/u; el modelo final `runs/wgangp/G_final.pt` pesa ≈3.6 MB) y `runs/*/ckpt/G_epoch_010.pt` (usados para comparar) se incluyen en la entrega.
Para regenerar la galería basta con `runs/wgangp/G_final.pt` + `runs/wgangp/config.json` y los datos no son necesarios (solo para la prueba de vecinos).

## Semillas
| Uso | Semilla |
|---|---|
| Entrenamiento (`random`, `numpy`, `torch` y el orden del DataLoader) | 2026 |
| Ruido fijo de evaluación (16 vectores z) | 7 |
| Muestras para métricas (1,000 z) | 123 |
| Candidatos de la galería | 0 a 199 (un z por semilla: `torch.randn(1, 100, generator=torch.Generator().manual_seed(s))`) |
| Personajes finales | ver `galeria/galeria.json` (modelo `runs/wgangp/G_final.pt`) |

Nota: en MPS/GPU algunas operaciones no son deterministas bit a bit, así que volver a entrenar puede dar pesos ligeramente distintos. La galería sí se regenera de forma exacta a partir de `G_final.pt` y las semillas.

## Versiones
- Parte 1 (corridas base, hinge, sn, mbstd, noise, ttur): Python 3.14.7 · torch 2.14.1 · torchvision 0.29.1 · numpy 2.5.4 · matplotlib 3.11.2 · pillow 12.3.0, en un Apple M3 (backend MPS).
- Parte 2 (sn_bias, wgangp, galería, notebook y PDF): Python 3.12.10 · torch 2.13.0+cpu · torchvision 0.28.0 · numpy 2.3.4 · matplotlib 3.10.9 · pillow 12.2.0 · reportlab 5.0.1 · nbformat 5.10.4, en Windows 11 (solo CPU).

## Datos y licencia
TinyHero (https://github.com/AgaMiko/pixel_character_generator) contiene 3,648 sprites de 64×64 generados con el
Universal LPC Spritesheet (https://github.com/makrohn/Universal-LPC-spritesheet). Licencia dual CC-BY-SA 3.0 / GNU GPL 3.0.
No contiene rostros de personas reales.
