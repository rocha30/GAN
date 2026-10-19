"""Textos de interpretación de los resultados (parte 2), compartidos por make_notebook.py y make_presentacion.py.

Las cifras citadas salen de figuras/metricas.json y galeria/galeria.json (modelo runs/wgangp/G_final.pt, galería
con semillas 60 10 6 66 83 109 13 168 50 54). Si se regenera algo, hay que revisar que sigan coincidiendo.
Los textos PDF_* usan el marcado de reportlab (<b>, <i>, &lt;); los del notebook, Markdown.
"""

# ------------------------------------------------------------------------------------------------ notebook (Markdown)
INTERP_GALERIA = (
    """**Resultado del modelo final.** WGAN-GP es la única corrida sin colapso:
- **FID-ResNet18 0.181**, frente a 1.027 de la mejor DCGAN (`sn`): 5.7 veces menor.
- **Diversidad 0.221**, frente a 0.023 de `sn` y 0.274 de los datos reales: 9.4 veces la de `sn` y el 80 % de la real.
- **Fondo medio 0.998** (real 1.000) y **desviación entre muestras 0.217** (real 0.222): el fondo es blanco y la variación entre muestras es como la de los datos.
- **Curvas:** la distancia de Wasserstein estimada C(x) − C(G(z)) baja de ~340 a ~12 en los primeros ~1,500 pasos de G y luego se estabiliza. Esta vez la curva sí acompaña lo que se ve en la rejilla.

**Contraste con la hipótesis previa.** Se cumplió en la rejilla de ruido fijo (columnas distintas desde la época 25) y en la FID.
En diversidad la mejora fue de 9.4× respecto a `sn`: prácticamente un orden de magnitud, aunque **estrictamente no alcanzó el umbral de 10× que nos fijamos**.
Recordatorio: `wgangp` cambia varias cosas a la vez (pérdida, crítico sin BN y con sesgos, optimizador y ancho), así que es el modelo elegido tras el diagnóstico, no un experimento controlado.

**`sn_bias` frente a su hipótesis previa:** se cumplió.
- El fondo medio pasó de 0.19 a 0.93.
- El logit medio de los falsos se alejó de 0 (≈ −0.74).
- La FID bajó de 1.027 a 0.915.
- Como esperábamos, el colapso en z siguió: la diversidad fue 0.013 y las 16 columnas de la rejilla son iguales.""",

    """**Galería.** Los 10 personajes tienen silueta humanoide completa, contorno oscuro y fondo blanco, como pide el universo de Pixelia.
Combinan cuatro vistas (frente, espalda, perfil), cabello rojo, verde, negro, azul, castaño, blanco, cobrizo, rubio y rosa, y ropa de distintos colores.
- **Diversidad interna:** 0.237 (1 − coseno medio entre pares), comparable a la de los datos reales (0.274).
- **Pareja más parecida:** coseno 0.897 entre Garrik (#7) y Lysa (#8). Aun así se distinguen a simple vista: uno está de espaldas con armadura oscura y la otra de perfil, rubia y con un arma.
- **Reemplazo durante la selección:** el candidato de la semilla 64 se descartó porque se parecía demasiado a Albarán (#6) (coseno 0.863: cabello claro, cuerpo amarillo, de espaldas). Lo sustituyó la semilla 54.

**Defectos honestos:** comparados con el pixel art real, los personajes tienen textura ruidosa (píxeles sueltos de colores, sobre todo en la ropa), contornos menos limpios y algunas manchas fuera del cuerpo (#2, #4).
Se reconocen como habitantes de Pixelia, pero no tienen la nitidez de un sprite LPC hecho a mano.""",
)

INTERP_VECINOS = """**Referencia real→real.** Cada imagen real comparada con su vecina real más cercana tiene, en features ResNet18:
- coseno medio 0.952, mediana 0.946, p5 0.885 y p95 1.000;
- en píxeles, L2 medio 14.6 y mediana 17.2.

El p95 = 1.000 se debe a que el 10.7 % de los sprites reales tiene un **duplicado exacto** al voltearlo: las vistas izquierda y derecha de un mismo personaje son espejo una de otra. Por eso la referencia es exigente.

**Generados.**

| # | Personaje | Semilla | Coseno vecino CNN | L2 vecino píxel | % reales con un vecino real más parecido (coseno / L2) |
|---|---|---|---|---|---|
| 1 | Rhosyn de Brasaviva | 60 | **0.911** | 31.5 | 79 % / 93 % |
| 2 | Ivra Hojaverde | 10 | 0.907 | 41.4 | 83 % / 99 % |
| 3 | Corvin Sombrafría | 6 | 0.892 | **25.7** | 92 % / 78 % |
| 4 | Nerea Mareaclara | 66 | 0.868 | 33.2 | 98 % / 96 % |
| 5 | Tobin Cardoverde | 83 | 0.898 | 36.4 | 89 % / 98 % |
| 6 | Albarán el Escarchado | 109 | 0.881 | 30.0 | 96 % / 91 % |
| 7 | Garrik Forjarroja | 13 | 0.875 | 33.2 | 97 % / 96 % |
| 8 | Lysa Trigodorado | 168 | 0.896 | 29.2 | 90 % / 89 % |
| 9 | Mirela de Valcastaño | 50 | 0.851 | 31.0 | 99 % / 92 % |
| 10 | Ysolde Rosalba | 54 | 0.892 | 32.3 | 92 % / 94 % |

**Lectura.**
- **Ningún personaje es una copia.** Los 10 cosenos (0.846–0.911) quedan por debajo de la media (0.952) y de la mediana (0.946) de la referencia real. Entre el 79 % y el 99 % de las imágenes reales tienen una vecina real *más* parecida que la que tiene cada personaje generado.
- **En píxeles, ninguno está cerca de un duplicado.** Las distancias L2 de los generados (25.7–41.4) son mayores que la mediana real (17.2).
- **Los parecidos son de plantilla, no de copia.** Los vecinos comparten vista, peinado o color de cabello (Ivra y su vecina de cabello verde largo, de espaldas; Garrik y su vecino de cabello cobrizo, de espaldas), pero difieren en ropa, colores del cuerpo o raza. Tobin, de camisa verde, tiene como vecino CNN a un orco verde: la CNN empareja por color y pose, no por identidad.
- **Las dos métricas señalan imágenes distintas en 9 de 10 casos.** Ningún sprite de entrenamiento explica a la vez la forma (píxeles) y el contenido (CNN) del personaje.
- **La excepción es Rhosyn (#1, semilla 60).** Tiene el coseno más alto (0.911), y su vecino CNN (`2/440`) y su vecino en píxeles (`3/440`) son **el mismo personaje real, el #440, en dos de sus vistas**: una chica de cabello rojo con armadura dorada. Rhosyn conserva el peinado, el color de cabello, la cara y la pose, pero cambia la armadura por una camisa blanca. Aun así, el 79 % de los reales tiene una vecina más parecida que la suya, así que la contamos como nueva, aunque es la más cercana a un modelo concreto.
- **Corvin (#3)** tiene la menor distancia en píxeles (25.7), pero su vecino en píxeles es otro personaje (cabello oscuro, otra ropa), y el 22 % de los reales están más lejos de su propio vecino real. Es un parecido de silueta y tono de piel, no una copia.

**Limitaciones de la prueba.** ResNet18 se entrenó con fotos de ImageNet y no con pixel art, y la L2 en píxeles castiga mucho el ruido de textura de los generados. Ambas pueden subestimar el parecido.
Por eso no nos basamos solo en las cifras: revisamos a ojo los 10 pares de la figura."""

REFLEXION = """**El que más costó defender fue Rhosyn de Brasaviva (#1, semilla 60).** Tiene el mayor coseno (0.911) y es el único cuyos dos vecinos (CNN y píxeles) son el mismo personaje real (#440): cabello rojo con el mismo peinado, la misma cara y la misma pose.
Lo que lo hace nuevo es la ropa: cambia la armadura dorada por una camisa blanca. Aun así, el 79 % de las imágenes reales tiene una vecina real más parecida que la suya.

**Qué dice de lo que aprendió la GAN.** Los sprites LPC se construyen por capas (cuerpo, cabello, ropa, armas) sobre una plantilla corporal común y cuatro vistas, y la GAN aprendió esa **estructura combinatoria**: la plantilla, las vistas y las capas como piezas intercambiables.
Por eso sus personajes son recombinaciones nuevas y no copias. Pero cuando una pieza es poco frecuente, como este peinado rojo, el generador se queda pegado a los pocos ejemplos que la tienen y solo cambia las demás capas.
Además, aprendió mejor *qué* va en cada lugar (forma y color) que el *cómo* del pixel art: los contornos y las texturas son más ruidosos que en los originales. La GAN aprendió la gramática de Pixelia, pero no domina todavía su caligrafía."""

# ------------------------------------------------------------------------------------------------ presentación (reportlab)
CRITERIO_CORTO = ("revisamos a ojo los 200 y exigimos (1) silueta humanoide legible, (2) sin artefactos graves y (3) que no se "
                  "parezcan entre sí en cabello, vista ni ropa. La sugerencia automática (greedy max-min en ResNet18) se descartó "
                  "porque favorecía candidatos con artefactos. La semilla 64 se cambió por la 54 por parecerse a #6 (coseno 0.863).")

PDF_RUIDO_FIJO = ("<b>Modelo final WGAN-GP (abajo):</b> desde la época 25 cada columna tiene una silueta distinta (ancho, piernas, "
                  "cabeza); hacia la época 100 aparecen el color y los detalles (cabello rojo, verde, azul o blanco; varias vistas), "
                  "y las columnas <b>no convergen</b> entre sí hasta la época 270. Es la única corrida sin colapso: diversidad 0.221 "
                  "(real 0.274) frente a 0.012–0.024 de todas las DCGAN.")

PDF_CURVAS_WGAN = ("<b>WGAN-GP (abajo):</b> la pérdida del crítico es negativa por diseño (−W + GP), así que la alarma «D cerca "
                   "de 0» no aplica. La distancia de Wasserstein estimada baja de ~340 a ~12 en ~1,500 pasos de G y se "
                   "estabiliza: G se acerca a los datos y el crítico no lo aplasta. Esta vez la curva sí coincide con la rejilla "
                   "(sin colapso) y con la FID (0.181 frente a 1.027 de sn).")

PDF_LECCION = ("<b>Lección:</b> las curvas «sanas» de sn engañaban y la FID sola tampoco bastaba. Lo que destrabó el proyecto fue no "
               "conformarse con 200 candidatos idénticos y medir directamente: el fondo, la variación entre muestras, el logit de "
               "D para los falsos y la homogeneidad de D. Cada hipótesis de causa se puso a prueba con un experimento "
               "(sn_bias) o con pilotos antes de cambiar de arquitectura.")

PDF_GALERIA = ("<b>Lectura:</b> los 10 tienen silueta completa, contorno oscuro y fondo blanco, en cuatro vistas, con cabello y ropa "
               "distintos. Diversidad interna 0.237 (real 0.274); la pareja más parecida (#7–#8, coseno 0.897) se distingue a simple "
               "vista. <b>Defectos:</b> textura ruidosa y contornos menos limpios que el pixel art LPC.")

PDF_VECINOS = ("<b>Resultado:</b> ningún personaje es copia. Los 10 cosenos (0.846–0.911) están bajo la media real (0.952) y entre "
               "el 79 % y el 99 % de los reales tiene un vecino real más parecido; en píxeles, L2 de 25.7–41.4 frente a una mediana "
               "real de 17.2. En 9 de 10 casos el vecino CNN y el vecino en píxeles son personajes distintos. La excepción es "
               "<b>#1 Rhosyn</b> (coseno 0.911): ambos vecinos son el mismo personaje real, #440, en dos vistas; comparte peinado y "
               "pose, pero no la armadura. El p95 = 1.0 de la referencia se debe a que el 10.7 % de los reales tiene un "
               "duplicado exacto en espejo.")

PDF_CONCLUSIONES = (
    "• <b>Todas las DCGAN colapsaron</b> (base + 5 experimentos de un solo cambio + sn_bias). Hinge y SN mejoraron la FID, "
    "pero ninguna recuperó la diversidad.<br/>"
    "• <b>Encontramos dos causas</b>: un D homogéneo en sn (sin sesgos ni BN), confirmado con sn_bias, y que BatchNorm en D no "
    "castiga lotes sin variedad.<br/>"
    "• <b>WGAN-GP rompió el colapso</b>: FID 0.181 (5.7× menor), diversidad 0.221 (9.4× mayor que sn) y fondo correcto.<br/>"
    "• La galería es <b>nueva</b> según los vecinos más cercanos y <b>coherente</b> con Pixelia.<br/>"
    "• Ni las curvas sanas ni una sola métrica alcanzan: hacen falta la rejilla, la diversidad y los vecinos.")

PDF_LIMITACIONES = (
    "• Solo CPU: el modelo final usa ngf = 32, y los experimentos controlados duraron 570 iteraciones; más tiempo podría "
    "cambiar el ranking.<br/>"
    "• wgangp cambia varias cosas a la vez: sabemos que funciona, no cuál de los cambios es imprescindible.<br/>"
    "• FID-ResNet18 no es el FID estándar, y ResNet18 no se entrenó con pixel art.<br/>"
    "• El criterio de selección (10 de 200) es en parte subjetivo; por eso lo declaramos y lo apoyamos con cosenos.<br/>"
    "• Textura ruidosa: los sprites no alcanzan la limpieza del pixel art hecho a mano.")

PDF_REFLEXION = (
    "<b>Rhosyn de Brasaviva (#1, semilla 60)</b> fue el que más costó defender. Tiene el mayor coseno (0.911) y es el único cuyos "
    "dos vecinos son el mismo personaje real (#440): mismo cabello rojo, cara y pose. Lo nuevo es la ropa, una camisa blanca en "
    "lugar de armadura dorada, y aun así el 79 % de los reales tiene una vecina más parecida.",
    "<b>Qué aprendió la GAN:</b> los sprites LPC se arman por capas (cuerpo, cabello, ropa, armas) sobre una plantilla común y "
    "cuatro vistas. La GAN aprendió esa <b>estructura combinatoria</b> y recombina capas, por eso inventa personajes en vez de "
    "copiarlos. Pero cuando una capa es rara, como este peinado, se pega a los pocos ejemplos que la tienen. Y aprendió mejor "
    "<i>qué</i> va en cada lugar (forma y color) que el <i>cómo</i> del pixel art, con contornos y texturas ruidosos.",
    "<b>En una frase:</b> aprendió la gramática de Pixelia, todavía no su caligrafía.")

MATRIZ = [
    ["Evidencia", "Diapositiva", "Conclusión", "Limitación"],
    ["Rejilla de ruido fijo", "6 (+ anexo)",
     "Las 7 DCGAN colapsan desde la época 2–5 (16 columnas iguales); WGAN-GP mantiene 16 personajes distintos hasta la época 270.",
     "16 z son una muestra pequeña; por eso la completamos con la diversidad sobre 1,000 muestras."],
    ["Curvas de pérdida", "7",
     "D gana en base, hinge, mbstd y noise; en sn y ttur las curvas «sanas» esconden colapso; en WGAN-GP la W estimada baja de "
     "~340 a ~12 y se estabiliza.",
     "Las curvas no miden diversidad, y la pérdida del crítico de WGAN no es comparable con la de BCE."],
    ["Prueba de vecinos más cercanos", "10",
     "Ningún personaje es copia: coseno 0.846–0.911 frente a una media real de 0.952; en 9 de 10, el vecino CNN y el de píxeles son "
     "personajes distintos. #1 es el más cercano (comparte peinado con el real #440).",
     "ResNet18 no se entrenó con pixel art; la L2 castiga el ruido; la referencia está inflada por duplicados en espejo (10.7 %)."],
    ["Galería y tasa de selección", "9 (+ anexo)",
     "10 de 200 (5 %) con criterio declarado; 10 personajes distintos (diversidad interna 0.237, real 0.274), coherentes con "
     "Pixelia y regenerables bit a bit (galeria.py verify).",
     "Criterio en parte subjetivo; textura ruidosa frente al pixel art real."],
    ["Experimento de pérdida (hinge vs. BCE)", "4",
     "Hipótesis cumplida en parte: FID 1.151 → 1.034 y diversidad 0.016 → 0.024, pero el colapso persiste.",
     "Solo 570 iteraciones por corrida; una sola semilla."],
    ["Experimento de estabilización (SN + mbstd, noise, ttur, sn_bias)", "5 y 8",
     "SN: mejor FID de las DCGAN, pero colapso. sn_bias confirma la causa A (fondo 0.19 → 0.93). Ninguna técnica sobre DCGAN "
     "evita el colapso; WGAN-GP sí.",
     "Experimentos cortos (10 épocas); wgangp combina varios cambios, así que no aísla una sola causa."],
    ["Diagnóstico del colapso", "8",
     "Dos causas: un D sin sesgos ni BN es homogéneo (D(0.25x)/D(x) = 0.25, D(0) = 0) y BatchNorm en D no castiga lotes sin "
     "variedad. Justifican el cambio a WGAN-GP.",
     "La causa B se apoya en pilotos cortos (300–800 iteraciones) fuera de runs/, no en corridas controladas completas."],
]
