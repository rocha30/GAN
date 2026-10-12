# Instrucciones para la pareja: Proyecto 2 GAN

**Entrega: hoy, domingo 11 de octubre, 23:59 en Canvas.** Ve en orden y vigila la hora: los pasos 1 a 3 son obligatorios; el 4 es bonificación.

## Qué ya está hecho (parte 1)
- Universo *Reinos de Pixelia* y datos documentados (TinyHero, CC-BY-SA 3.0 / GPL 3.0, 3,648 imágenes): en el notebook y el README.
- `gan.py`: DCGAN de 64×64, pérdidas BCE no saturante y hinge, normalización espectral, minibatch-std, ruido de instancia, TTUR y entrenamiento con checkpoints.
- Seis corridas entrenadas en `runs/` (`base`, `hinge`, `sn`, `mbstd`, `noise`, `ttur`) con la misma semilla, los mismos datos y el mismo ruido fijo, comparadas en la época 10. **Todas presentan colapso de modos** (ver Resultados).
- `evaluate.py figures`: curvas de pérdida, rejillas de ruido fijo, muestras y métricas en `figuras/`.
- `proyecto2_gan.ipynb` con las hipótesis previas y el análisis de los experimentos; `README.md`.
- **Resultados y conclusiones de los experimentos:** sección "Resultados (parte 1)" al final de este archivo.

## Tu parte
### Paso 0: preparar el entorno (3 min)
```bash
git clone <URL_DEL_REPO> GAN && cd GAN
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy matplotlib pillow nbformat jupyter ipykernel
python -m ipykernel install --user --name gan-venv      # kernel con torch para el notebook
# datos (no están en git):
mkdir -p data && curl -L -o data/tinyhero.zip https://github.com/AgaMiko/pixel_character_generator/raw/master/data.zip
unzip -q data/tinyhero.zip -d data/tinyhero
```
Los pesos ya vienen en el repo (`runs/*/G_final.pt`); **no hace falta re-entrenar**.

### Paso 1: programar `galeria.py` (código, obligatorio)
El archivo ya tiene las firmas y la especificación detallada en los docstrings. Hay que implementar dos funciones:

1. **`candidates(run)`**: generar 200 personajes, uno por semilla `0..199` usando `z_from_seed(s)` de `gan.py`,
   y guardar la hoja `figuras/candidatos.png` con el número de semilla sobre cada imagen.
2. **`gallery(run, seeds, names)`**:
   - guardar en `galeria/` cada personaje en PNG (64×64 y ampliado ×4 con `Image.NEAREST`) y su vector z (`.npy`);
   - **prueba de vecinos más cercanos** contra las 3,648 imágenes de entrenamiento (`TinyHero(augment=False).x`):
     - con features de CNN: `cnn_features` (ResNet18 preentrenada, ya importada desde `evaluate.py`), con similitud coseno `f_gen @ f_real.T` y `argmax`;
     - en píxeles: `torch.cdist(...)` y `argmin`;
     - **referencia para interpretar**: la similitud de cada imagen *real* con su vecina real más cercana (pon `-1` en la diagonal para excluirla). Un generado más parecido a su vecino que lo típico entre reales es sospechoso de memorización;
   - figura `figuras/vecinos_mas_cercanos.png`: 3 filas × 10 columnas (generado / vecino CNN / vecino píxel), con el valor en cada título, como en la figura 2 de Goodfellow et al. (2014);
   - `galeria/galeria.json` con el modelo, la **tasa de selección ("10 de 200")**, el criterio y, para cada personaje, la semilla, los archivos, los vecinos y las distancias.

Útiles de `gan.py`: `load_generator(RUNS_DIR / run)`, `z_from_seed(s)`, `to_uint8(x)`, `TinyHero(augment=False)` (`.x` = tensor N×3×64×64 en [-1,1] y `.paths` = rutas).

```bash
python galeria.py candidates --run sn                  # modelo recomendado: sn
open figuras/candidatos.png                            # elige 10 personajes DISTINTOS (vista, ropa, colores)
python galeria.py gallery --run sn --seeds S1 S2 ... S10 --names nombres.json
```
**Criterio de selección** (escríbelo en el JSON y en el PDF): por ejemplo, "silueta humanoide legible, sin artefactos graves, y que no se parezcan entre sí en vista, ropa ni paleta".
No ocultes la tasa: no declararla resta **−10**.

### Paso 2: notebook (5 min)
```bash
python make_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=gan-venv proyecto2_gan.ipynb   # ~2 min
```
Luego abre `proyecto2_gan.ipynb` y escribe la interpretación de la prueba de vecinos y la reflexión final en las celdas marcadas **TODO (pareja)** (al final del notebook).
Para cada personaje: ¿es nuevo o casi idéntico a su vecino? Compara su coseno con la referencia real (media y p95). Sé honesto: si alguno es casi copia, dilo y reemplázalo.

### Paso 3: `presentacion.pdf` (máximo 12 diapositivas + anexos; obligatorio)
Puedes hacerla en Google Slides, Canva o PowerPoint y exportarla a PDF. Todas las figuras están en `figuras/` y `galeria/`.

| # | Diapositiva | Contenido y fuente |
|---|---|---|
| 1 | Universo | Título, integrantes y el texto del universo (notebook, celda 1) + `figuras/muestras_reales.png` |
| 2 | Datos | Origen, licencia, tamaño y preparación (notebook, celda "Datos") |
| 3 | Arquitectura | DCGAN, por qué la elegimos, hiperparámetros y tabla de las 3 corridas (notebook) |
| 4 | Experimento de pérdida | Hipótesis previa (textual) → resultado: `muestras_base.png` vs `muestras_hinge.png`, métricas y conclusión (ver abajo) |
| 5 | Experimentos de estabilización | SN (principal) + tabla de mbstd, noise y ttur; `muestras_sn.png` |
| 6 | Evidencia 1: ruido fijo | `figuras/ruido_fijo_base.png` y `ruido_fijo_sn.png` + interpretación (filas idénticas = colapso) |
| 7 | Evidencia 2: curvas | `figuras/curvas_perdida.png` + interpretación |
| 8 | Problemas y decisiones | Ver abajo (colapso, D dominante, recorte de épocas, etc.) |
| 9 | Galería | Los 10 `_x4.png` con semilla (y nombre/clase si haces el bono) + **tasa de selección y criterio** |
| 10 | Evidencia 3: vecinos | `figuras/vecinos_mas_cercanos.png` + interpretación honesta |
| 11 | Conclusiones, limitaciones y **reflexión final** | Responder: *¿cuál de los 10 costó más defender como "nuevo" frente a su vecino más cercano y qué dice eso sobre lo que aprendió la GAN?* (elige el de mayor coseno) |
| 12 | **Matriz de evidencias** | Tabla con las columnas **evidencia, diapositiva donde aparece, conclusión, limitación**, con una fila por evidencia: ruido fijo, curvas, vecinos, galería y tasa, exp. de pérdida, exp. de estabilización |
| Anexos | opcional | `ruido_fijo_*.png` de las otras corridas, `candidatos.png` |

### Paso 4: bonificación (nombres y clases)
Crea `nombres.json` con 10 entradas en el mismo orden que las semillas, coherentes con Pixelia (fantasía medieval):
```json
[{"nombre": "Aldric el Errante", "clase": "Explorador"}, {"nombre": "...", "clase": "Maga de las Brasas"}]
```
Vuelve a correr el paso 1 con `--names nombres.json` y pon nombre y clase en la diapositiva 9.

### Paso 5: entrega en Canvas
1. Agrega tu nombre y tu uso de IA en la sección "Declaración de uso de asistentes de IA" del `README.md`.
2. Comprime la carpeta **sin** `.venv/` ni `data/` (los datos se descargan con el README):
   ```bash
   cd .. && zip -r Proyecto2_GAN.zip GAN -x "GAN/.venv/*" "GAN/data/*" "GAN/runs/*/ckpt/*" "GAN/runs/*/G_final.pt" && \
   zip Proyecto2_GAN.zip GAN/runs/sn/G_final.pt
   ```
   Así el zip queda pequeño (~25 MB) e incluye `runs/sn/G_final.pt`, que es lo necesario para regenerar la galería.
   Sube la carpeta `runs/` completa (con los pesos y checkpoints de las 6 corridas, ~800 MB) a Google Drive y pega el enlace en el README (sección "Cómo ejecutar").
3. Sube a Canvas `Proyecto2_GAN.zip` + `presentacion.pdf` (también puedes subir la carpeta `galeria/` aparte).

### Lista de verificación final
- [ ] `galeria/` tiene 10 PNG + 10 `.npy` + `galeria.json` con semillas y tasa de selección
- [ ] `python galeria.py gallery ...` regenera exactamente la galería (si no, **−15**)
- [ ] Prueba de vecinos interpretada con honestidad
- [ ] PDF con ≤ 12 diapositivas, matriz de evidencias al final y reflexión final
- [ ] Hipótesis previas ("Creemos que ___ mejorará ___ porque ___. Lo consideraremos útil si ___.") en el PDF
- [ ] README con nombres y uso de IA

---
## Resultados (parte 1)
### Resultados de los experimentos (todas las corridas comparadas en la época 10 = 570 iteraciones, con la misma semilla, los mismos datos y el mismo ruido)
Métricas con 1,000 muestras (semilla 123). **FID-ResNet18** es la distancia de Fréchet entre features de ResNet18 de las muestras reales y las generadas (menor es mejor; *no* es comparable con el FID estándar de Inception).
**Diversidad** es 1 − la similitud coseno media entre pares de muestras (mayor es más diverso). Referencia de los datos reales: **0.274**.

| Corrida | Cambio único vs. base | FID-ResNet18 ↓ | Diversidad ↑ | % iteraciones con pérdida de D < 0.05 |
|---|---|---|---|---|
| base | — | 1.151 | 0.016 | 24 % |
| hinge | pérdida hinge | 1.034 | 0.024 | 38 %* |
| sn | spectral norm en D | **1.027** | 0.023 | **0 %** |
| mbstd | minibatch-std en D | 1.160 | 0.018 | 46 % |
| noise | ruido en entradas de D (σ = 0.1 → 0) | 1.065 | 0.020 | 45 % |
| ttur | lr de D = 5e-5 | 1.174 | 0.015 | 0 % |
| ttur (época 15) | — | 1.105 | 0.012 | 0 % |

\*En hinge, una pérdida de D igual a 0 significa que se cumple el margen en ambos lados, que es el equivalente a que D "gane".

**Hallazgo principal: colapso de modos en TODAS las configuraciones.** Desde la época 2–5, los 16 vectores z fijos producen prácticamente el mismo personaje
(una silueta humanoide promedio con textura ruidosa), y la diversidad generada (0.012–0.024) está un orden de magnitud por debajo de la real (0.274).
Lo verificamos de tres formas: (1) la rejilla de ruido fijo, con filas idénticas; (2) generando en modo `train()` y en modo `eval()` de BatchNorm (la desviación estándar entre muestras es 0.024 en ambos, así que no es un artefacto de las estadísticas de BN);
(3) repitiendo la base en CPU, con las mismas pérdidas (lossG ≈ 8.8 en la época 1), así que no es un bug del backend MPS. Al inicializar, G sí varía con z (desviación estándar entre muestras de 0.13), por lo que el colapso ocurre durante el entrenamiento.

**Experimento de pérdida (hinge vs. BCE):** la hipótesis se cumplió solo en parte. La FID bajó (1.151 → 1.034) y la diversidad subió (0.016 → 0.024),
pero el colapso persiste y D sigue dominando (el margen se satisface en el 38 % de las iteraciones). Conclusión: hinge mejora un poco, pero no basta.

**Experimento de estabilización (spectral norm en D):** fue la mejor configuración. Es la única con curvas realmente "enfrentadas": la pérdida de D está en ~0.8–1.2 sin caer nunca bajo 0.05,
D(x) ≈ 0.8 y D(G(z)) ≈ 0.5. Tuvo la menor FID (1.027) y casi la misma diversidad que hinge. Pero la hipótesis de que la rejilla mostraría filas distintas **no se cumplió**: G colapsó aunque D no lo dominaba.
Esto sugiere que el colapso no se debe solo a un D demasiado fuerte.
**Experimentos adicionales** (minibatch-std, ruido de instancia, TTUR), cada uno con un solo cambio respecto a la base: ninguno evitó el colapso. TTUR equilibró el juego (D(G(z)) ≈ 0.2–0.4) pero no aumentó la diversidad.
minibatch-std y ruido empeoraron el dominio de D (la pérdida de D quedó < 0.05 en ~45 % de las iteraciones, una señal de alarma según lo visto en clase).

**Interpretación de las curvas.** En base, hinge, mbstd y noise, la pérdida de D cae casi a 0 y la de G se dispara (5–20), con D(x) ≈ 0.95 y D(G(z)) < 0.1: D gana y G recibe gradientes poco informativos.
En sn y ttur, las curvas se estabilizan enfrentadas (el patrón "sano" visto en clase) y aun así hay colapso. **Lección: unas curvas sanas no garantizan diversidad**; hace falta mirar la rejilla de ruido fijo y medir la diversidad.

**Problemas y decisiones.** (1) Plan inicial: 30 épocas por corrida en paralelo (~40 s/época en un M3). El colapso apareció en la época 5–10, así que detuvimos todas en la época 10 para compararlas en igualdad de condiciones
y usar el tiempo en 3 técnicas adicionales anti-colapso. (2) mbstd se continuó hasta la época 20 y ttur hasta la 15, sin mejora visible. (3) Limitación de cómputo: cada corrida tuvo solo 570–1,140 iteraciones,
muy pocas para una GAN (DCGAN suele necesitar miles). Es probable que con más iteraciones y combinando técnicas (SN + hinge + TTUR, al estilo de SNGAN/SAGAN) se rompa el colapso, y eso queda como trabajo futuro.
(4) Posible causa del dataset: los 3,648 sprites comparten la misma plantilla corporal y fondo, así que la "silueta promedio" es un mínimo local muy atractivo para G.

**Modelo recomendado para la galería: `sn`** (mejor FID, curvas más sanas y diversidad cercana a la máxima observada).


**Para la galería:** como el modelo colapsó, los 200 candidatos se van a parecer mucho entre sí. Elige los 10 MÁS distintos que encuentres (vista, color) y **dilo con honestidad**: la rúbrica premia el análisis honesto ("Prueba de vecinos más cercanos, interpretada con honestidad" y "Análisis de fallos, colapso de modos cuando aplique"). No uses imágenes de ningún otro generador. Si los 10 son casi iguales entre sí, decláralo en la diapositiva 9 y en las conclusiones.

**Hipótesis previas textuales** para el PDF: están en `proyecto2_gan.ipynb` (sección "Diseño experimental").

