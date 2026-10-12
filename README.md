# Proyecto 2: Diseño generativo de personajes, *Reinos de Pixelia*

DCGAN de 64×64 entrenada con el dataset TinyHero (pixel art LPC, con licencia CC-BY-SA 3.0 / GPL 3.0) para generar personajes nuevos de un RPG de fantasía.

## Estructura
| Archivo / carpeta | Contenido |
|---|---|
| `proyecto2_gan.ipynb` | Notebook principal: universo, datos, hipótesis previas, evidencias, galería y análisis |
| `gan.py` | Datos, generador, discriminador, pérdidas (BCE no saturante / hinge), normalización espectral, minibatch-std, ruido de instancia, TTUR y entrenamiento con checkpoints |
| `evaluate.py` | Curvas de pérdida, rejillas de ruido fijo, muestras y métricas (FID con features de ResNet18 y diversidad) |
| `galeria.py` | 200 candidatos, galería final y prueba de vecinos más cercanos |
| `runs/<corrida>/` | `G_final.pt`, `log.json`, `config.json`, `fixed/` (rejillas de ruido fijo), `ckpt/` (checkpoints cada 5 épocas) |
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
# (en zsh usa ${=S} en lugar de $S)

# 3) Figuras y métricas (todas comparadas en el checkpoint de la época 10)
python evaluate.py figures --final ttur

# 4) Galería: regenera los 10 personajes desde runs/<modelo>/G_final.pt y las semillas de galeria/galeria.json
python galeria.py candidates --run <modelo>
python galeria.py gallery --run <modelo> --seeds <semillas de galeria/galeria.json> --names nombres.json
```
Si se interrumpe, `gan.py train` reanuda desde `runs/<nombre>/ckpt/last.pt`. Los pesos `runs/*/G_final.pt` (≈14 MB c/u) y `runs/*/ckpt/G_epoch_010.pt` (usados para comparar) se incluyen en la entrega.

## Semillas
| Uso | Semilla |
|---|---|
| Entrenamiento (`random`, `numpy`, `torch` y el orden del DataLoader) | 2026 |
| Ruido fijo de evaluación (16 vectores z) | 7 |
| Muestras para métricas (1,000 z) | 123 |
| Candidatos de la galería | 0 a 199 (un z por semilla: `torch.randn(1, 100, generator=torch.Generator().manual_seed(s))`) |
| Personajes finales | ver `galeria/galeria.json` |

Nota: en MPS/GPU algunas operaciones no son deterministas bit a bit, así que volver a entrenar puede dar pesos ligeramente distintos. La galería sí se regenera de forma exacta a partir de `G_final.pt` y las semillas.

## Versiones
Python 3.14.7 · torch 2.14.1 · torchvision 0.29.1 · numpy 2.5.4 · matplotlib 3.11.2 · pillow 12.3.0. Entrenado en un Apple M3 (backend MPS).

## Datos y licencia
TinyHero (https://github.com/AgaMiko/pixel_character_generator) contiene 3,648 sprites de 64×64 generados con el
Universal LPC Spritesheet (https://github.com/makrohn/Universal-LPC-spritesheet). Licencia dual CC-BY-SA 3.0 / GNU GPL 3.0.
No contiene rostros de personas reales.

## Declaración de uso de asistentes de IA
Usamos Claude Code (Anthropic) como asistente para: elegir el dataset con licencia abierta, escribir el código de
entrenamiento y evaluación (`gan.py`, `evaluate.py`), estructurar el notebook y redactar el borrador de este README.
`galeria.py`, la selección de personajes, los nombres y clases, y la presentación son trabajo de la pareja.
Las ideas de los experimentos y las conclusiones fueron revisadas por el equipo, que entiende y es responsable del código.
**Ninguna imagen de la galería proviene de un generador externo:** todas las produce `runs/<modelo>/G_final.pt`.
