"""Construye proyecto2_gan.ipynb (notebook principal de la entrega)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Proyecto 2 — Diseño generativo de personajes
## Universo: *Reinos de Pixelia*

**Universo visual.** Pixelia es un mundo de fantasía medieval en estilo *pixel art* de 16 bits, como el de los RPG clásicos.
Sus habitantes (guerreros, magos, aldeanos y exploradores) se representan de **cuerpo completo**, en **64×64 px**, sobre fondo neutro,
vistos desde uno de **cuatro ángulos** (frente, espalda, izquierda y derecha). Rasgos que esperamos ver en cada personaje:
- silueta humanoide legible: cabeza, torso, brazos y piernas;
- paleta limitada con contornos oscuros, como en el pixel art;
- vestimenta o equipo variado: túnicas, armaduras, cascos, capuchas, cabello de distintos colores y tonos de piel diversos;
- personajes distintos entre sí, con combinaciones de ropa y colores que no aparezcan tal cual en el conjunto de entrenamiento.

Esta descripción es la vara con la que juzgamos la galería final."""))

C.append(md("""## Datos
- **Origen:** dataset *TinyHero* (https://github.com/AgaMiko/pixel_character_generator), con 3,648 personajes generados con el
  *Universal LPC Spritesheet* (https://github.com/makrohn/Universal-LPC-spritesheet).
- **Licencia/permiso:** el arte LPC tiene licencia dual **CC-BY-SA 3.0 / GNU GPL 3.0**, que permite usarlo y modificarlo citando la fuente. No contiene rostros de personas reales.
- **Tamaño:** 3,648 imágenes de 64×64 RGBA, 912 por cada uno de los 4 ángulos de vista (carpetas 0–3).
- **Preparación:** se compone el canal alfa sobre fondo blanco (RGBA → RGB). No hace falta recortar ni redimensionar porque las imágenes ya miden 64×64.
  Se normalizan a [-1, 1] para la salida `tanh` y se aplica **aumento con volteo horizontal** (p = 0.5), coherente porque las vistas izquierda y derecha son espejo una de otra."""))

C.append(code("""import json, subprocess, sys
from pathlib import Path
from IPython.display import Image, display
import torch, torchvision, numpy, matplotlib
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)
TRAIN = False   # True para re-entrenar las 3 corridas (~7 min c/u en un Apple M3; menos en una GPU de Colab)"""))

C.append(md("""## Arquitectura
DCGAN (Radford et al., 2016) a 64×64, siguiendo la recomendación del enunciado de *empezar simple*.
- **G:** z ∈ R¹⁰⁰ → ConvTranspose 4×4 → 8×8 → 16×16 → 32×32 → 64×64×3, con BatchNorm + ReLU y salida Tanh.
- **D:** Conv stride 2 + LeakyReLU(0.2) y BatchNorm → un logit.
- **Hiperparámetros:** Adam(lr = 2e-4, β = (0.5, 0.999)), batch 64; comparación en la época 10 (570 iteraciones), inicialización N(0, 0.02).
- **Comunes a todas las corridas:** semilla global 2026, ruido fijo de evaluación (16 z, semilla 7), mismos datos y mismas épocas.

Código: `gan.py` (datos, modelos, pérdidas y entrenamiento), `evaluate.py` (figuras y métricas de los experimentos) y `galeria.py` (galería y vecinos más cercanos)."""))

C.append(md("""## Diseño experimental (se cambia una sola cosa a la vez respecto a la base)
| Corrida | Pérdida | D | Qué cambia respecto a la base |
|---|---|---|---|
| `base` | BCE no saturante | BatchNorm | — (control) |
| `hinge` | **hinge** | BatchNorm | solo la pérdida |
| `sn` | BCE no saturante | **normalización espectral** (sin BN) | solo la normalización de D |

### Hipótesis previas (escritas antes de entrenar)
**Experimento de pérdida (hinge):**
> *Creemos que la **pérdida hinge** mejorará **la estabilidad y la diversidad de las muestras** porque solo penaliza los logits que no superan el margen,
> así D deja de empujar hacia confianzas extremas y G recibe gradientes útiles. Lo consideraremos útil si **la FID-ResNet18 baja respecto a la base,
> la diversidad de las muestras aumenta y la pérdida de D no se queda cerca de cero**.*

**Experimento de estabilización (normalización espectral en D):**
> *Creemos que la **normalización espectral en D** mejorará **la estabilidad del juego y la diversidad** porque acota la constante de Lipschitz de D,
> lo que evita que domine a G desde el inicio (en la base, D(x) ≈ 0.98 en la primera época). Lo consideraremos útil si **las curvas de G y D se estabilizan
> enfrentadas en lugar de que D gane, la FID-ResNet18 baja y la rejilla de ruido fijo muestra filas distintas entre sí**.*

### Experimentos adicionales de estabilización (agregados al ver colapso en la época 5–10; hipótesis escritas antes de entrenarlos)
| Corrida | Cambio único vs. base | Hipótesis |
|---|---|---|
| `mbstd` | capa *minibatch-std* en D | *Creemos que **minibatch-std** mejorará **la diversidad** porque D podrá detectar lotes sin variación y castigarlos. Lo consideraremos útil si la diversidad sube y la rejilla muestra filas distintas.* |
| `noise` | ruido gaussiano en entradas de D (σ = 0.1 → 0) | *Creemos que **el ruido de instancia** mejorará **el equilibrio G/D** porque oculta la textura ruidosa de G que D detecta trivialmente. Lo consideraremos útil si la pérdida de D deja de estar cerca de 0 y la diversidad sube.* |
| `ttur` | lr de D = 5e-5 (G = 2e-4) | *Creemos que **un D más lento** mejorará **el equilibrio y la diversidad** porque G tendrá tiempo de explorar antes de que D lo castigue. Lo consideraremos útil si D(G(z)) se mantiene en 0.3–0.6 y la diversidad sube.* |"""))

C.append(code("""if TRAIN:
    snaps = ['--snap_epochs', '1', '2', '5', '10', '15', '20', '--ckpt_every', '5']
    for args, ep in ((['--name','base','--loss','bce'], 10), (['--name','hinge','--loss','hinge'], 10), (['--name','sn','--loss','bce','--sn'], 10),
                     (['--name','mbstd','--loss','bce','--mbstd'], 20), (['--name','noise','--loss','bce','--inst_noise','0.1'], 15),
                     (['--name','ttur','--loss','bce','--lrD','5e-5'], 15)):
        subprocess.run([sys.executable, 'gan.py', 'train', *args, '--epochs', str(ep), *snaps], check=True)
subprocess.run([sys.executable, 'evaluate.py', 'figures', '--final', 'ttur'], check=True)
print(json.dumps(json.load(open('figuras/metricas.json')), indent=2))"""))

C.append(md("## Evidencia 1 — Rejilla de ruido fijo (los mismos 16 vectores z a lo largo del entrenamiento)"))
C.append(code("""for r in ['base', 'hinge', 'sn', 'mbstd', 'noise', 'ttur']:
    display(Image('figuras/ruido_fijo_%s.png' % r))"""))
C.append(md("## Evidencia 2 — Curvas de pérdida"))
C.append(code("display(Image('figuras/curvas_perdida.png'))"))
C.append(md("### Muestras finales por corrida (1,000 muestras para métricas; se muestran 64) y muestras reales"))
C.append(code("""for r in ['reales', 'base', 'hinge', 'sn', 'mbstd', 'noise', 'ttur']:
    print(r); display(Image('figuras/muestras_%s.png' % r, width=520))"""))
C.append(md(r"""### Resultados de los experimentos (todas las corridas comparadas en la época 10 = 570 iteraciones, con la misma semilla, los mismos datos y el mismo ruido)
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
"""))

C.append(md("""## Galería final
Se generaron **200 candidatos** con el modelo elegido (semillas 0–199, un z por semilla, `z = torch.randn(1, 100, generator=torch.Generator().manual_seed(semilla))`)
y se eligieron 10. La celda siguiente regenera la galería exactamente a partir de `runs/<modelo>/G_final.pt` y las semillas declaradas en `galeria/galeria.json`."""))
C.append(code("""meta = json.load(open('galeria/galeria.json'))
run = meta['modelo'].split('/')[1]
seeds = [str(p['semilla']) for p in meta['personajes']]
subprocess.run([sys.executable, 'galeria.py', 'candidates', '--run', run], check=True)
display(Image('figuras/candidatos.png'))
subprocess.run([sys.executable, 'galeria.py', 'gallery', '--run', run, '--seeds', *seeds], check=True)
print('Tasa de selección:', meta['tasa_seleccion'])"""))
C.append(md("## Evidencia 3 — Prueba de vecinos más cercanos (ResNet18 ImageNet, similitud coseno; y distancia L2 en píxeles)"))
C.append(code("""display(Image('figuras/vecinos_mas_cercanos.png'))
meta = json.load(open('galeria/galeria.json'))
print('Referencia: similitud coseno de cada imagen REAL con su vecina real más cercana:', meta['referencia_real_nn_cos'])
for p in meta['personajes']:
    print(p['n'], 'semilla', p['semilla'], 'cos=%.3f' % p['nn_cnn_cos'], 'L2=%.1f' % p['nn_pixel_l2'],
          '| %% de reales cuyo vecino real es aún más parecido: %.0f%%' % (100 * p['frac_reales_con_vecino_mas_cercano']))"""))
C.append(md("""### Interpretación de la prueba de vecinos más cercanos
**TODO (pareja):** para cada personaje, ¿es nuevo o casi idéntico a su vecino? Compara su coseno con la referencia real (media y p95).

### Reflexión final
**TODO (pareja):** ¿qué personaje costó más defender como \"nuevo\" y qué dice eso de lo que aprendió la GAN?"""))

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "proyecto2_gan.ipynb")
print("ok")
