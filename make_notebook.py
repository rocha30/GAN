"""Construye proyecto2_gan.ipynb (notebook principal de la entrega)."""
import nbformat as nbf

from analisis import INTERP_GALERIA, INTERP_VECINOS, REFLEXION

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
from IPython.display import Image, Markdown, display
import torch, torchvision, numpy, matplotlib
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)
TRAIN = False   # True para re-entrenar las 8 corridas (las 7 comparativas tardan ~7 min c/u en un Apple M3; la final ~2 h en CPU)"""))

C.append(md("""## Arquitectura
**Base: DCGAN** (Radford et al., 2016) a 64×64, siguiendo la recomendación del enunciado de *empezar simple*: es la arquitectura convolucional de referencia,
cabe en el cómputo disponible y todos los experimentos de pérdida y estabilización se pueden expresar como **un solo cambio** sobre ella.
- **G:** z ∈ R¹⁰⁰ → ConvTranspose 4×4 → 8×8 → 16×16 → 32×32 → 64×64×3, con BatchNorm + ReLU y salida Tanh.
- **D:** Conv stride 2 + LeakyReLU(0.2) y BatchNorm → un logit.
- **Hiperparámetros:** Adam(lr = 2e-4, β = (0.5, 0.999)), batch 64, ngf = ndf = 64; comparación en la época 10 (570 iteraciones), inicialización N(0, 0.02).
- **Comunes a todas las corridas comparativas:** semilla global 2026, ruido fijo de evaluación (16 z, semilla 7), mismos datos y mismas épocas.

**Modelo final: WGAN-GP** (Gulrajani et al., 2017), elegido **después** de diagnosticar por qué colapsaban todas las variantes DCGAN (ver "Diagnóstico del colapso").
Mismo generador DCGAN; el discriminador pasa a ser un **crítico** sin BatchNorm (la penalización de gradiente es por muestra y BatchNorm la rompe), **con sesgos**,
pérdida de Wasserstein + penalización de gradiente (λ = 10), 5 pasos del crítico por paso de G, Adam(1e-4, β = (0, 0.9)).
Por el costo de 5 pasos de D por paso de G en CPU, usamos ngf = ndf = 32 (≈ 4× menos operaciones) y 270 épocas (≈ 3,000 pasos de G).

Código: `gan.py` (datos, modelos, pérdidas y entrenamiento), `evaluate.py` (figuras y métricas) y `galeria.py` (galería y vecinos más cercanos)."""))

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
| `ttur` | lr de D = 5e-5 (G = 2e-4) | *Creemos que **un D más lento** mejorará **el equilibrio y la diversidad** porque G tendrá tiempo de explorar antes de que D lo castigue. Lo consideraremos útil si D(G(z)) se mantiene en 0.3–0.6 y la diversidad sube.* |

### Parte 2: diagnóstico y modelo final (hipótesis escritas antes de las corridas completas)
| Corrida | Cambio | Hipótesis |
|---|---|---|
| `sn_bias` | `sn` + **sesgos en las convoluciones de D** (único cambio respecto a `sn`, mismo protocolo de 10 épocas) | *Creemos que **agregar sesgos a D** mejorará **el fondo y el color de las muestras** porque sin sesgos ni BatchNorm D es homogénea (D(c·x) = c·D(x)) y no puede castigar imágenes grises cercanas a 0. Lo consideraremos útil si el fondo medio de las muestras se acerca a 1 (blanco) y el logit de D para los falsos se aleja de 0. **No** esperamos que, por sí solo, rompa el colapso en z.* |
| `wgangp` | modelo final WGAN-GP (varios cambios a la vez: **no** es un experimento controlado, sino la configuración elegida tras el diagnóstico; un piloto de 300 pasos de G en una copia temporal mostró siluetas distintas por z) | *Creemos que **WGAN-GP** mejorará **la diversidad** porque el crítico sin BatchNorm evalúa cada muestra por separado y la distancia de Wasserstein da gradientes útiles aunque las distribuciones no se solapen. Lo consideraremos útil si la rejilla de ruido fijo muestra filas distintas, la diversidad ResNet18 sube al menos un orden de magnitud respecto a `sn` y la FID-ResNet18 baja.* |"""))

C.append(code("""if TRAIN:
    snaps = ['--snap_epochs', '1', '2', '5', '10', '15', '20', '--ckpt_every', '5']
    for args, ep in ((['--name','base','--loss','bce'], 10), (['--name','hinge','--loss','hinge'], 10), (['--name','sn','--loss','bce','--sn'], 10),
                     (['--name','mbstd','--loss','bce','--mbstd'], 20), (['--name','noise','--loss','bce','--inst_noise','0.1'], 15),
                     (['--name','ttur','--loss','bce','--lrD','5e-5'], 15), (['--name','sn_bias','--loss','bce','--sn','--dbias'], 10)):
        subprocess.run([sys.executable, 'gan.py', 'train', *args, '--epochs', str(ep), *snaps], check=True)
    subprocess.run([sys.executable, 'gan.py', 'train', '--name', 'wgangp', '--loss', 'wgangp', '--ngf', '32', '--ndf', '32', '--lr', '1e-4',
                    '--beta1', '0.0', '--beta2', '0.9', '--epochs', '270', '--ckpt_every', '10',
                    '--snap_epochs', '1', '5', '10', '25', '50', '100', '150', '200', '270'], check=True)
subprocess.run([sys.executable, 'evaluate.py', 'figures', '--final', 'ttur', 'wgangp'], check=True)
M = json.load(open('figuras/metricas.json'))"""))

C.append(md("## Evidencia 1 — Rejilla de ruido fijo (los mismos 16 vectores z a lo largo del entrenamiento)"))
C.append(code("""for r in ['wgangp', 'sn', 'sn_bias', 'base', 'hinge', 'mbstd', 'noise', 'ttur']:
    display(Image('figuras/ruido_fijo_%s.png' % r))"""))
C.append(md("""**Lectura.** En las 7 corridas DCGAN, desde la época 2–5 las 16 columnas son el mismo personaje: **colapso de modos total** (G ignora z).
En `sn` además el personaje es gris y sin color; en `sn_bias` (mismo modelo con sesgos en D) recupera fondo blanco y color, pero sigue colapsado.
En el modelo final `wgangp`, desde la época 25 cada columna tiene una silueta distinta y la diferencia se mantiene hasta el final: es la única corrida sin colapso."""))

C.append(md("## Evidencia 2 — Curvas de pérdida"))
C.append(code("display(Image('figuras/curvas_perdida.png'))\ndisplay(Image('figuras/curvas_wgangp.png'))"))
C.append(md("### Muestras finales por corrida (1,000 muestras para métricas; se muestran 64) y muestras reales"))
C.append(code("""for r in ['reales', 'wgangp_final', 'sn', 'sn_bias', 'base', 'hinge', 'mbstd', 'noise', 'ttur']:
    print(r); display(Image('figuras/muestras_%s.png' % r, width=520))"""))

C.append(md("### Tabla de métricas (calculada desde `figuras/metricas.json`)"))
C.append(code("""rows = [('base', '—'), ('hinge', 'pérdida hinge'), ('sn', 'spectral norm en D'), ('mbstd', 'minibatch-std en D'),
        ('noise', 'ruido en entradas de D'), ('ttur', 'lr de D = 5e-5'), ('ttur_final', 'ttur, época 15'),
        ('sn_bias', 'sn + sesgos en D'), ('wgangp_final', 'modelo final WGAN-GP')]
t = '| corrida | cambio | FID-ResNet18 ↓ | diversidad ↑ | fondo medio (real = %.2f) | desv. entre muestras (real = %.3f) | %% iters pérdida D < 0.05 |\\n|---|---|---|---|---|---|---|\\n' % (
    M['real']['fondo_medio'], M['real']['desv_entre_muestras'])
for r, c in rows:
    m = M[r]; f = m['frac_iters_lossD_lt_0.05']
    t += '| %s | %s | %.3f | %.3f | %.2f | %.3f | %s |\\n' % (r, c, m['fid_resnet18'], m['diversidad'], m['fondo_medio'], m['desv_entre_muestras'],
                                                       'n/a (WGAN)' if f is None else '%.0f %%' % (100 * f))
t += '| real | — | — | %.3f | %.2f | %.3f | — |' % (M['real']['diversidad'], M['real']['fondo_medio'], M['real']['desv_entre_muestras'])
display(Markdown(t))"""))

C.append(md(r"""### Resultados de los experimentos controlados (todas las corridas comparadas en la época 10 = 570 iteraciones, con la misma semilla, los mismos datos y el mismo ruido)
Métricas con 1,000 muestras (semilla 123). **FID-ResNet18** es la distancia de Fréchet entre features de ResNet18 de las muestras reales y las generadas (menor es mejor; *no* es comparable con el FID estándar de Inception).
**Diversidad** es 1 − la similitud coseno media entre pares de muestras (mayor es más diverso). **Fondo medio**: valor medio del borde de 4 px (en los datos reales es 1.0 = blanco).

**Hallazgo principal de la parte 1: colapso de modos en TODAS las configuraciones DCGAN.** Desde la época 2–5, los 16 vectores z fijos producen prácticamente el mismo personaje
(una silueta humanoide promedio con textura ruidosa), y la diversidad generada (0.012–0.024) está un orden de magnitud por debajo de la real (0.274).
Lo verificamos de tres formas: (1) la rejilla de ruido fijo, con filas idénticas; (2) generando en modo `train()` y en modo `eval()` de BatchNorm (misma desviación entre muestras, así que no es un artefacto de las estadísticas de BN);
(3) repitiendo la base en CPU, con las mismas pérdidas (lossG ≈ 8.8 en la época 1), así que no es un bug del backend MPS. Al inicializar, G sí varía con z (desviación entre muestras de 0.13), por lo que el colapso ocurre durante el entrenamiento (medimos que cae a 0.045 en solo 20 iteraciones).

**Experimento de pérdida (hinge vs. BCE):** la hipótesis se cumplió solo en parte. La FID bajó (1.151 → 1.034) y la diversidad subió (0.016 → 0.024),
pero el colapso persiste y D sigue dominando (el margen se satisface en el 38 % de las iteraciones). Conclusión: hinge mejora un poco, pero no basta.

**Experimento de estabilización (spectral norm en D):** la mejor de las DCGAN en FID (1.027) y la única con curvas "enfrentadas" sin que la pérdida de D caiga bajo 0.05.
Pero la hipótesis de que la rejilla mostraría filas distintas **no se cumplió**. El diagnóstico de abajo muestra que esas curvas "sanas" eran engañosas: D(G(z)) ≈ 0.5 porque D estaba atrapado en logit ≈ 0, no porque G lo engañara.
**Experimentos adicionales** (minibatch-std, ruido de instancia, TTUR), cada uno con un solo cambio respecto a la base: ninguno evitó el colapso. TTUR equilibró el juego pero no aumentó la diversidad;
minibatch-std y ruido empeoraron el dominio de D (pérdida de D < 0.05 en ~45 % de las iteraciones, una señal de alarma).

**Interpretación de las curvas.** En base, hinge, mbstd y noise, la pérdida de D cae casi a 0 y la de G se dispara (5–20), con D(x) ≈ 0.95 y D(G(z)) < 0.1: D gana y G recibe gradientes poco informativos.
En sn y ttur las curvas se estabilizan enfrentadas y aun así hay colapso. **Lección: unas curvas sanas no garantizan diversidad**; hace falta mirar la rejilla de ruido fijo y medir la diversidad."""))

C.append(md(r"""## Diagnóstico del colapso (parte 2)
Antes de elegir la galería generamos los 200 candidatos de `sn`: **los 200 eran el mismo personaje gris** (diversidad 0.024). En lugar de elegir entre 200 copias, buscamos la causa.

**1. Sin bug en los datos ni en el bucle.** Los datos tienen 78 % de píxeles exactamente blancos (fondo); G sí puede aprender a producir fondo blanco
(entrenado con pérdida L2 directa contra imágenes reales, el fondo pasa de 0.13 a 0.80 en 200 iteraciones). D sí distingue reales de falsos (logit +2.8 vs. +0.04).

**2. Causa A — un D homogéneo en `sn`.** Al quitar BatchNorm para usar normalización espectral, D quedó **sin ningún sesgo** (todas las convoluciones tienen `bias=False`, pensado para cuando hay BN).
Una red de convoluciones sin sesgo + LeakyReLU es **positivamente homogénea**: D(c·x) = c·D(x) y D(0) = 0 (verificado numéricamente en `metricas.json → homogeneidad_D`: el cociente D(0.25x)/D(x) es exactamente 0.25; con sesgos, ≈ 1.03).
Consecuencia: una imagen gris de bajo contraste (≈ 0 en [-1, 1]) recibe logit ≈ 0 y casi ningún gradiente; G se queda allí. Lo medimos: el gradiente de la pérdida de G sobre los píxeles del fondo es ~10⁻⁷
y el logit medio de D para los falsos en `sn` fue ≈ 0.04 (D(G(z)) ≈ 0.51) durante todo el entrenamiento. **Las curvas "sanas" de `sn` eran este punto muerto.**

**3. Causa B — colapso en z, común a todas las DCGAN.** Con BatchNorm, D normaliza los lotes de reales y de falsos **por separado**, así que un lote falso donde todas las muestras son iguales
se ve, tras la normalización, igual que cualquier otro: D casi no puede castigar la falta de variedad. Y G aprende muy rápido a producir el "personaje promedio"
(la plantilla corporal compartida por los 3,648 sprites es un mínimo muy atractivo).

**Experimento controlado `sn_bias`** (único cambio: sesgos en D; mismo protocolo de 10 épocas): el fondo y el color se recuperan y el logit de los falsos baja a ≈ −0.74,
**pero las 16 columnas siguen siendo el mismo personaje**. Confirma que A y B son problemas distintos: los sesgos arreglan A, no B.

Pilotos rápidos en una copia temporal (ngf = 32, 300–800 iteraciones, no incluidos en `runs/`) con hinge + SN + TTUR + β = (0, 0.9), con y sin sesgos y con minibatch-std: **todos siguieron colapsados en z**.
El único que mostró siluetas distintas por z fue **WGAN-GP**, cuyo crítico no usa BatchNorm y cuya pérdida da gradientes útiles aunque las distribuciones no se solapen. Por eso es el modelo final."""))

C.append(md("## Modelo final: WGAN-GP"))
C.append(code("""display(Image('figuras/ruido_fijo_wgangp.png'))
display(Image('figuras/curvas_wgangp.png'))
for r in ['sn', 'sn_bias', 'wgangp_final']:
    print(r, {k: round(v, 3) for k, v in M[r].items() if isinstance(v, float)})"""))
C.append(md(INTERP_GALERIA[0]))

C.append(md("""## Galería final
Se generaron **200 candidatos** con el modelo final (semillas 0–199, un z por semilla, `z = torch.randn(1, 100, generator=torch.Generator().manual_seed(semilla))`)
y se eligieron 10. La celda siguiente regenera la galería exactamente a partir de `runs/<modelo>/G_final.pt`, las semillas y los nombres declarados en `galeria/galeria.json`,
y `galeria.py verify` comprueba bit a bit que los PNG y los z regenerados son idénticos a los guardados."""))
C.append(code("""meta = json.load(open('galeria/galeria.json', encoding='utf-8'))
run = meta['modelo'].split('/')[1]
seeds = [str(p['semilla']) for p in meta['personajes']]
json.dump([{'nombre': p['nombre'], 'clase': p['clase']} for p in meta['personajes']], open('nombres.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
subprocess.run([sys.executable, 'galeria.py', 'candidates', '--run', run], check=True)
display(Image('figuras/candidatos.png'))
subprocess.run([sys.executable, 'galeria.py', 'gallery', '--run', run, '--seeds', *seeds, '--names', 'nombres.json', '--criterio', meta['criterio']], check=True)
subprocess.run([sys.executable, 'galeria.py', 'verify'], check=True)
meta = json.load(open('galeria/galeria.json', encoding='utf-8'))
print('Tasa de selección:', meta['tasa_seleccion'])
print('Criterio:', meta['criterio'])
display(Image('figuras/galeria.png'))"""))
C.append(md(INTERP_GALERIA[1]))

C.append(md("## Evidencia 3 — Prueba de vecinos más cercanos (ResNet18 ImageNet, similitud coseno; y distancia L2 en píxeles)"))
C.append(code("""display(Image('figuras/vecinos_mas_cercanos.png'))
meta = json.load(open('galeria/galeria.json', encoding='utf-8'))
print('Referencia: similitud coseno de cada imagen REAL con su vecina real más cercana:', meta['referencia_real_nn_cos'])
print('Referencia: distancia L2 de cada imagen REAL a su vecina real más cercana:', meta['referencia_real_nn_l2'])
for p in meta['personajes']:
    print(p['n'], p['nombre'], '| semilla', p['semilla'], '| cos=%.3f' % p['nn_cnn_cos'], 'L2=%.1f' % p['nn_pixel_l2'],
          '| %% de reales cuyo vecino real es aún más parecido: %.0f%%' % (100 * p['frac_reales_con_vecino_mas_cercano']))"""))
C.append(md("### Interpretación de la prueba de vecinos más cercanos\n" + INTERP_VECINOS))
C.append(md("### Reflexión final\n" + REFLEXION))

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "proyecto2_gan.ipynb")
print("ok")
