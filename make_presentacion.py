"""Genera presentacion.pdf (12 diapositivas 16:9 + anexos) a partir de figuras/, galeria/ y las métricas guardadas.

  python make_presentacion.py

Las cifras se leen de figuras/metricas.json y galeria/galeria.json, así que el PDF siempre coincide con los resultados.
"""
import json
import os
from pathlib import Path

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parent
FIG, GAL = ROOT / "figuras", ROOT / "galeria"
W, H = 960, 540                                   # 16:9 en puntos
M = 36                                            # margen

# fuente con →, ≈, σ, β, ×: DejaVu viene con matplotlib, así el PDF se genera igual en cualquier máquina
_fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_fd / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(_fd / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(_fd / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVBI", str(_fd / "DejaVuSans-BoldOblique.ttf")))
from reportlab.lib.fonts import addMapping  # noqa: E402
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DVB"); addMapping("DV", 0, 1, "DVI"); addMapping("DV", 1, 1, "DVBI")

INK = colors.HexColor("#1f2430")
MUTED = colors.HexColor("#5b6170")
ACCENT = colors.HexColor("#8a3b12")               # tono "pergamino/óxido" acorde a la fantasía medieval
BAND = colors.HexColor("#f3ece2")
RULE = colors.HexColor("#d8cbb8")


def style(size=12, lead=None, color=INK, bold=False, align=0):
    return ParagraphStyle("s", fontName="DVB" if bold else "DV", fontSize=size, leading=lead or size * 1.3,
                          textColor=color, alignment=align, spaceAfter=size * 0.35)


BODY, SMALL, TINY = style(12.5), style(10.5), style(8.5, color=MUTED)


class Deck:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.c.setTitle("Proyecto 2 — Reinos de Pixelia (GAN)")
        self.n = 0

    def slide(self, title, kicker=None, annex=False):
        if self.n:
            self.c.showPage()
        self.n += 1
        c = self.c
        c.setFillColor(colors.white); c.rect(0, 0, W, H, fill=1, stroke=0)
        c.setFillColor(BAND); c.rect(0, H - 64, W, 64, fill=1, stroke=0)
        c.setFillColor(ACCENT); c.rect(0, H - 64, 8, 64, fill=1, stroke=0)
        c.setFillColor(INK); c.setFont("DVB", 22); c.drawString(M, H - 42, title)
        if kicker:
            c.setFillColor(MUTED); c.setFont("DV", 11); c.drawRightString(W - M, H - 40, kicker)
        c.setFillColor(MUTED); c.setFont("DV", 8.5)
        c.drawString(M, 14, "Proyecto 2 · Diseño generativo de personajes · Reinos de Pixelia")
        c.drawRightString(W - M, 14, f"Anexo {self.n - 12}" if annex else f"{self.n} / 12")

    def text(self, x, y_top, w, h, paras, st=BODY):
        """paras: lista de str (markup de reportlab: <b>, <i>, <font>) o Paragraph."""
        items = [p if isinstance(p, Paragraph) else Paragraph(p, st) for p in paras]
        f = Frame(x, y_top - h, w, h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0)
        left = f.addFromList(items, self.c)
        assert not items, f"texto desbordado en la diapositiva {self.n}: {[i.text[:60] for i in items]}"
        return left

    def image(self, path, x, y_top, w, h, caption=None, align="center"):
        """Escala la imagen para que quepa en (w, h) conservando la proporción; devuelve el ancho/alto usados."""
        iw, ih = PILImage.open(path).size
        s = min(w / iw, h / ih)
        dw, dh = iw * s, ih * s
        dx = x + (w - dw) / 2 if align == "center" else x
        self.c.drawImage(ImageReader(str(path)), dx, y_top - dh, dw, dh, mask="auto")
        if caption:
            self.c.setFillColor(MUTED); self.c.setFont("DV", 8.5)
            self.c.drawCentredString(dx + dw / 2, y_top - dh - 11, caption)
        return dw, dh

    def table(self, x, y_top, data, widths, size=9.5, header=True, bold_rows=()):
        st = style(size, lead=size * 1.22)
        stb = style(size, lead=size * 1.22, bold=True)
        rows = [[Paragraph(str(cell), stb if (header and i == 0) or i in bold_rows else st) for cell in row]
                for i, row in enumerate(data)]
        t = Table(rows, colWidths=widths)
        cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if header:
            cmds += [("BACKGROUND", (0, 0), (-1, 0), BAND), ("LINEBELOW", (0, 0), (-1, 0), 0.9, ACCENT)]
        t.setStyle(TableStyle(cmds))
        _, th = t.wrapOn(self.c, sum(widths), H)
        t.drawOn(self.c, x, y_top - th)
        return th

    def box(self, x, y_top, w, h, paras, st=SMALL, fill=BAND):
        """Recuadro con texto; con h=None la altura se ajusta al contenido."""
        if h is None:
            items = [p if isinstance(p, Paragraph) else Paragraph(p, st) for p in paras]
            h = sum(it.wrap(w - 20, H)[1] + it.style.spaceAfter for it in items) + 18
            paras = items
        self.c.setFillColor(fill); self.c.setStrokeColor(RULE)
        self.c.roundRect(x, y_top - h, w, h, 6, fill=1, stroke=1)
        self.text(x + 10, y_top - 8, w - 20, h - 14, paras, st)

    def save(self):
        self.c.save()


# ----------------------------------------------------------------------------------------------------------- contenido
def q(s):
    """Cita de hipótesis previa."""
    return Paragraph(f"<i>{s}</i>", style(10.5, color=colors.HexColor("#3d3326")))


def build():
    import analisis as A
    Mx = json.load(open(FIG / "metricas.json", encoding="utf-8"))
    G = json.load(open(GAL / "galeria.json", encoding="utf-8"))
    ref = G["referencia_real_nn_cos"]
    f3 = lambda r, k: f"{Mx[r][k]:.3f}"
    pct = lambda r: "n/a" if Mx[r]["frac_iters_lossD_lt_0.05"] is None else f"{100 * Mx[r]['frac_iters_lossD_lt_0.05']:.0f} %"
    d = Deck(ROOT / "presentacion.pdf")

    # 1 ── universo
    d.slide("Reinos de Pixelia", "Universo")
    d.text(M, H - 92, 440, 60, [Paragraph("Una GAN que inventa habitantes para un RPG de fantasía medieval en pixel art",
                                          style(15, bold=True, color=ACCENT))])
    d.text(M, H - 150, 440, 330, [
        "Pixelia es un mundo de fantasía medieval en <b>pixel art de 16 bits</b>, como los RPG clásicos. Sus habitantes "
        "(guerreros, magos, aldeanos, exploradores) aparecen de <b>cuerpo completo</b>, en <b>64×64 px</b>, sobre fondo neutro y "
        "vistos desde uno de <b>cuatro ángulos</b>.",
        "<b>Rasgos esperados</b> (la vara con la que juzgamos la galería):",
        "• silueta humanoide legible: cabeza, torso, brazos y piernas<br/>"
        "• paleta limitada con contornos oscuros<br/>"
        "• vestimenta variada: túnicas, armaduras, capuchas, cabello y tonos de piel diversos<br/>"
        "• personajes distintos entre sí y que no copien el conjunto de entrenamiento",
        "<font color='#5b6170'>Integrantes: Mario Rocha · Luis Pedro Lira — Deep Learning 2026, UVG</font>"])
    d.image(FIG / "muestras_reales.png", 500, H - 84, 424, 400, "64 sprites reales de TinyHero (datos de entrenamiento)")

    # 2 ── datos
    d.slide("Datos: TinyHero", "Origen, permiso y preparación")
    d.table(M, H - 90, [
        ["Aspecto", "Detalle"],
        ["Origen", "TinyHero (github.com/AgaMiko/pixel_character_generator), generado con el Universal LPC Spritesheet "
                   "(github.com/makrohn/Universal-LPC-spritesheet)"],
        ["Licencia", "Dual <b>CC-BY-SA 3.0 / GNU GPL 3.0</b>: permite usar y modificar citando la fuente. Sin rostros de personas "
                     "reales ni personajes protegidos."],
        ["Tamaño", "<b>3,648</b> sprites RGBA de 64×64 px, 912 por cada vista (frente, espalda, izquierda, derecha)"],
        ["Preparación", "RGBA compuesto sobre <b>fondo blanco</b> (78 % de los píxeles quedan exactamente blancos). Sin recorte ni "
                        "redimensionamiento (ya miden 64×64). Normalización a [−1, 1] para la salida tanh."],
        ["Aumento", "<b>Volteo horizontal</b> (p = 0.5): coherente porque las vistas izquierda y derecha son espejo una de otra. "
                    "Por eso la prueba de vecinos busca también entre las imágenes volteadas."],
        ["Reproducibilidad", "Semilla de entrenamiento 2026 (random, numpy, torch y orden del DataLoader); ruido fijo de "
                             "evaluación: 16 z con semilla 7; métricas: 1,000 z con semilla 123; galería: semillas 0–199."],
    ], [130, 510], size=10.5)
    d.image(FIG / "muestras_reales.png", 700, H - 90, 224, 224)
    d.box(700, H - 330, 224, None, [
        "<b>Por qué este conjunto:</b> estilo muy consistente (misma plantilla corporal y paleta LPC) y pocos miles de imágenes, "
        "como recomienda el enunciado. Esa misma homogeneidad resultó un riesgo: la «silueta promedio» atrae al generador."])

    # 3 ── arquitectura
    d.slide("Arquitectura y diseño experimental", "DCGAN → diagnóstico → WGAN-GP")
    d.text(M, H - 86, 430, 420, [
        "<b>Base: DCGAN 64×64</b> (Radford et al., 2016). La elegimos por ser la referencia convolucional, caber en el cómputo y "
        "permitir que cada experimento sea <b>un solo cambio</b>.",
        "• <b>G:</b> z∈ℝ¹⁰⁰ → ConvT 4→8→16→32→64 px, BatchNorm + ReLU, salida tanh<br/>"
        "• <b>D:</b> Conv stride 2 + LeakyReLU(0.2) + BatchNorm → 1 logit<br/>"
        "• Adam(2e-4, β=(0.5, 0.999)), batch 64, ngf = ndf = 64, init N(0, 0.02)",
        "<b>Modelo final: WGAN-GP</b> (Gulrajani et al., 2017), elegido tras diagnosticar el colapso (diap. 8): mismo G; crítico "
        "<b>sin BatchNorm y con sesgos</b>, λ = 10 (penalización de gradiente), 5 pasos del crítico por paso de G, Adam(1e-4, β=(0, 0.9)); "
        "ngf = ndf = 32 y 270 épocas (≈3,000 pasos de G) por el costo en CPU.",
        "<b>Por qué WGAN-GP:</b> el crítico evalúa cada muestra por separado (sin BatchNorm) y la distancia de Wasserstein da "
        "gradientes útiles aunque reales y falsos no se solapen, justo lo que faltaba en las DCGAN."], SMALL)
    d.table(490, H - 86, [
        ["Corrida", "Cambio único vs. control", "Épocas"],
        ["base", "— (control: BCE no saturante, BN en D)", "10"],
        ["hinge", "pérdida hinge", "10"],
        ["sn", "normalización espectral en D (sin BN)", "10"],
        ["mbstd", "capa minibatch-std en D", "10 (+10)"],
        ["noise", "ruido gaussiano en entradas de D (σ 0.1→0)", "10 (+5)"],
        ["ttur", "lr de D = 5e-5 (G = 2e-4)", "10 (+5)"],
        ["sn_bias", "sn + sesgos en D (control: sn)", "10"],
        ["wgangp", "modelo final (varios cambios: no es experimento controlado)", "270"],
    ], [62, 300, 72], size=9.5)
    d.box(490, H - 262, 434, None, [
        "<b>Protocolo:</b> las 7 corridas DCGAN se comparan en el <b>checkpoint de la época 10</b> (570 iteraciones) con la misma "
        "semilla, los mismos datos y el mismo ruido fijo. Métricas con 1,000 muestras: <b>FID-ResNet18</b> (Fréchet sobre "
        "features de ResNet18; no comparable con el FID de Inception) y <b>diversidad</b> = 1 − coseno medio entre pares "
        f"(reales: {Mx['real']['diversidad']:.3f})."])

    # 4 ── experimento de pérdida
    d.slide("Experimento de pérdida: hinge vs. BCE no saturante", "Requisito 4")
    d.text(M, H - 84, 440, 120, [
        "<b>Hipótesis previa</b> (escrita antes de entrenar):",
        q("«Creemos que la pérdida hinge mejorará la estabilidad y la diversidad porque solo penaliza los logits que no superan el "
          "margen, así D deja de empujar hacia confianzas extremas. Lo consideraremos útil si la FID-ResNet18 baja, la diversidad "
          "aumenta y la pérdida de D no se queda cerca de cero.»")])
    d.table(M, H - 180, [
        ["Control: misma semilla, datos y épocas", "base (BCE)", "hinge"],
        ["FID-ResNet18 ↓", f3("base", "fid_resnet18"), f3("hinge", "fid_resnet18")],
        ["Diversidad ↑ (real 0.274)", f3("base", "diversidad"), f3("hinge", "diversidad")],
        ["% iteraciones con pérdida D &lt; 0.05", pct("base"), pct("hinge") + "*"],
    ], [220, 100, 100], size=10.5)
    d.text(M, H - 272, 440, 230, [
        "<b>Resultado:</b> la hipótesis se cumplió <b>solo en parte</b>. La FID bajó 10 % y la diversidad subió 50 %, pero sigue "
        "un orden de magnitud por debajo de la real y <b>el colapso persiste</b>: D sigue ganando (el margen se cumple en ambos "
        "lados en el 38 % de las iteraciones).",
        "<b>Conclusión:</b> cambiar la pérdida suaviza el juego pero no basta; el problema de fondo estaba en el discriminador "
        "(diap. 8). Más adelante, la pérdida de Wasserstein (WGAN-GP) sí rompió el colapso.",
        "<font size=8.5 color='#5b6170'>* En hinge, pérdida de D = 0 significa que se cumple el margen con reales y falsos: "
        "el equivalente a que D «gane».</font>"], SMALL)
    d.image(FIG / "muestras_base.png", 500, H - 84, 205, 205, "base (época 10)")
    d.image(FIG / "muestras_hinge.png", 720, H - 84, 205, 205, "hinge (época 10)")
    d.box(500, H - 320, 424, None, [
        "<b>Lectura visual:</b> ambas producen la misma silueta oscura y ruidosa en las 64 muestras. La mejora numérica de hinge "
        "no se traduce en personajes distintos; por eso, además de la FID, medimos la diversidad y miramos la rejilla de ruido fijo."])

    # 5 ── estabilización
    d.slide("Experimentos de estabilización", "Requisito 5")
    d.text(M, H - 84, 470, 115, [
        "<b>Hipótesis previa (principal, normalización espectral en D):</b>",
        q("«Creemos que la normalización espectral en D mejorará la estabilidad y la diversidad porque acota la constante de "
          "Lipschitz de D y evita que domine a G. Lo consideraremos útil si las curvas se estabilizan enfrentadas, la FID baja y "
          "la rejilla de ruido fijo muestra filas distintas.»")])
    rows = [["Corrida", "FID ↓", "Divers. ↑", "Fondo (real 1.00)", "% D&lt;0.05", "¿Se cumplió la hipótesis?"]]
    verdict = {"base": "control", "sn": "parcial: mejor FID y curvas enfrentadas, pero colapso",
               "mbstd": "no: D domina más", "noise": "no: D domina más", "ttur": "parcial: equilibrio, sin diversidad",
               "sn_bias": "sí en fondo y color; no rompe el colapso (no se esperaba)"}
    for r in ["base", "sn", "mbstd", "noise", "ttur", "sn_bias"]:
        rows.append([r, f3(r, "fid_resnet18"), f3(r, "diversidad"), f"{Mx[r]['fondo_medio']:.2f}", pct(r), verdict[r]])
    d.table(M, H - 176, rows, [56, 48, 58, 80, 56, 172], size=9.2, bold_rows=(2,))
    d.text(M, H - 338, 470, 130, [
        "<b>Resultado:</b> SN fue la mejor DCGAN (menor FID, nunca D &lt; 0.05) pero <b>la rejilla siguió colapsada</b>. "
        "Los 3 experimentos extra (un solo cambio cada uno, con hipótesis previa en el notebook) tampoco lo evitaron. "
        "<b>sn_bias</b> recuperó fondo blanco y color (logit medio de los falsos ≈ 0 → −0.74): confirma la causa A de la diap. 8."],
        SMALL)
    d.image(FIG / "muestras_sn.png", 520, H - 84, 196, 196, "sn (época 10): gris, sin color")
    d.image(FIG / "muestras_sn_bias.png", 728, H - 84, 196, 196, "sn_bias (época 10): fondo y color")
    d.box(520, H - 310, 404, None, [
        "<b>Por qué las curvas «sanas» de SN engañaban:</b> sin BatchNorm y sin sesgos, D es homogénea: D(c·x) = c·D(x). "
        "G produjo imágenes grises cercanas a 0, donde D da logit ≈ 0 y D(G(z)) ≈ 0.5: <b>parecía equilibrio, era un punto "
        f"muerto</b>. Fondo medio de sn: {Mx['sn']['fondo_medio']:.2f}; con sesgos: {Mx['sn_bias']['fondo_medio']:.2f}."])

    # 6 ── evidencia 1: ruido fijo
    d.slide("Evidencia 1: rejilla de ruido fijo", "16 vectores z fijos (semilla 7) a lo largo del entrenamiento")
    _, h1 = d.image(FIG / "ruido_fijo_sn.png", M, H - 78, 540, 150, align="left")
    d.image(FIG / "ruido_fijo_wgangp.png", M, H - 78 - h1 - 10, 540, H - 78 - h1 - 10 - 30, align="left")
    d.text(600, H - 86, 324, 420, [
        "<b>Cómo se lee:</b> cada columna es el mismo z; cada fila, una época. Si las columnas se vuelven iguales, G ignora z: "
        "<b>colapso de modos</b>.",
        "<b>DCGAN (arriba, sn; las demás en el anexo):</b> desde la época 2–5 las 16 columnas son el mismo personaje. Pasa en "
        "<b>todas</b> las configuraciones DCGAN, también con minibatch-std y TTUR.",
        A.PDF_RUIDO_FIJO], SMALL)

    # 7 ── evidencia 2: curvas
    d.slide("Evidencia 2: curvas de pérdida interpretadas", "G y D deben estabilizarse enfrentadas, no bajar juntas")
    d.image(FIG / "curvas_perdida.png", M, H - 76, W - 2 * M, 215)
    d.image(FIG / "curvas_wgangp.png", M, H - 302, 480, 190, align="left")
    d.text(535, H - 300, 389, 230, [
        "<b>DCGAN (arriba):</b> en base, hinge, mbstd y noise la pérdida de D cae casi a 0 y la de G se dispara (5–20) con "
        "D(G(z)) &lt; 0.1: <b>D gana</b>, señal de alarma. En sn y ttur se ven «enfrentadas», pero con colapso: "
        "<b>curvas sanas no garantizan diversidad</b>.",
        A.PDF_CURVAS_WGAN], SMALL)

    # 8 ── problemas y decisiones
    d.slide("Problemas encontrados y decisiones", "Diagnóstico del colapso")
    d.table(M, H - 84, [
        ["Problema", "Evidencia", "Decisión"],
        ["Colapso en las 6 corridas DCGAN desde la época 2–5",
         "rejillas idénticas; diversidad 0.012–0.024 vs. 0.274 real; la desviación entre muestras cae de 0.13 a 0.045 en "
         "20 iteraciones", "detener todas en la época 10 y comparar en igualdad de condiciones; probar 3 técnicas extra"],
        ["¿Bug? (BN en eval, backend MPS, datos)",
         "mismo resultado en train()/eval() y en CPU; G sí aprende el fondo blanco con pérdida L2 directa",
         "descartado: no es un bug del código ni de los datos"],
        ["<b>Causa A:</b> D sin sesgos ni BN en sn",
         "D(0.25x)/D(x) = 0.25 exacto y D(0) = 0; gradiente sobre el fondo ~10⁻⁷; logit de los falsos ≈ 0 todo el entrenamiento",
         f"experimento controlado <b>sn_bias</b>: fondo {Mx['sn']['fondo_medio']:.2f} → {Mx['sn_bias']['fondo_medio']:.2f} "
         "y color, pero sigue colapsado"],
        ["<b>Causa B:</b> D con BN ve reales y falsos en lotes separados",
         "un lote falso sin variedad no se distingue tras normalizar; minibatch-std, TTUR y SN + hinge (pilotos) no alcanzaron",
         "<b>WGAN-GP</b>: crítico sin BN que evalúa cada muestra; un piloto de 300 pasos de G lo confirmó"],
        ["Cómputo: solo CPU, sin GPU", "WGAN-GP hace 5 pasos de D por paso de G",
         "ngf = ndf = 32 y 270 épocas (~2 h); checkpoints cada 10 épocas"],
        ["Los 200 candidatos de sn eran idénticos", "hoja de contacto con 200 copias del mismo personaje gris",
         "no elegir una galería de copias: se toma del modelo final, declarando la tasa de selección"],
    ], [190, 380, 318], size=9.6)
    d.image(FIG / "ruido_fijo_sn_bias.png", M, H - 280, 560, 128, align="left")
    d.text(610, H - 284, 314, 130, [
        "<b>Rejilla de sn_bias</b> (único cambio vs. sn: sesgos en D): desde la época 2 el fondo es blanco y hay color, "
        "pero las 16 columnas siguen siendo el mismo personaje. Arregla la causa A, no la B."], SMALL)
    d.text(M, H - 420, W - 2 * M, 100, [A.PDF_LECCION], SMALL)

    # 9 ── galería
    d.slide("Galería final: 10 habitantes de Pixelia", f"{G['tasa_seleccion']} · {G['modelo']}")
    cw, ch, img = 120, 214, 104                    # celda de 5 x 2 con la imagen x4 y nombre, clase y semilla debajo
    for i, p in enumerate(G["personajes"]):
        x0, y0 = M + (i % 5) * cw, H - 82 - (i // 5) * ch
        d.c.setStrokeColor(RULE); d.c.setFillColor(colors.white)
        d.c.rect(x0 + (cw - img) / 2 - 2, y0 - img - 2, img + 4, img + 4, fill=1, stroke=1)
        d.c.drawImage(ImageReader(str(ROOT / p["archivos"]["png_x4"])), x0 + (cw - img) / 2, y0 - img, img, img)
        d.text(x0 + 2, y0 - img - 8, cw - 4, 90, [
            Paragraph(f"<b>{p['nombre']}</b>", style(9, align=1)),
            Paragraph(p["clase"], style(8.5, color=ACCENT, align=1)),
            Paragraph(f"#{p['n']} · semilla {p['semilla']}", style(8, color=MUTED, align=1))])
    d.box(660, H - 80, 264, None, [
        f"<b>Tasa de selección:</b> {G['tasa_seleccion']} candidatos (semillas 0–199 del modelo final; hoja completa en el anexo).",
        f"<b>Criterio:</b> {A.CRITERIO_CORTO}"])
    d.text(660, H - 290, 264, 200, [A.PDF_GALERIA], SMALL)

    # 10 ── vecinos
    d.slide("Evidencia 3: prueba de vecinos más cercanos", "Contra los 3,648 datos de entrenamiento y sus volteados")
    d.image(FIG / "vecinos_mas_cercanos.png", M, H - 74, W - 2 * M, 285)
    d.text(M, H - 372, 400, 150, [
        "<b>Método:</b> features ResNet18 (ImageNet) normalizadas → coseno (argmax), y L2 en píxeles (argmin). "
        f"<b>Referencia real→real</b> (cada imagen real contra su vecina real más cercana): coseno medio {ref['media']:.3f}, "
        f"p95 {ref['p95']:.3f}. Un generado más parecido a su vecino que lo típico entre reales sería sospechoso de memorización."],
        SMALL)
    d.text(460, H - 372, 464, 150, [A.PDF_VECINOS], SMALL)

    # 11 ── conclusiones
    d.slide("Conclusiones, limitaciones y reflexión final")
    d.text(M, H - 84, 440, 420, ["<b>Conclusiones</b>", A.PDF_CONCLUSIONES, "<b>Limitaciones</b>", A.PDF_LIMITACIONES], SMALL)
    d.box(500, H - 84, 424, None, [
        "<b>Reflexión final.</b> <i>De sus 10 personajes, ¿cuál le costó más defender como «nuevo» frente a su vecino más "
        "cercano? ¿Qué le dice ese caso sobre lo que su GAN aprendió de su universo?</i>", *A.PDF_REFLEXION], SMALL)

    # 12 ── matriz de evidencias
    d.slide("Matriz de evidencias", "Guía de calificación")
    d.table(M, H - 82, A.MATRIZ, [150, 86, 372, 280], size=10.2)

    # anexos
    d.slide("Anexo: 200 candidatos del modelo final", G["tasa_seleccion"], annex=True)
    d.image(FIG / "candidatos.png", M, H - 76, W - 2 * M, 440)
    d.slide("Anexo: rejillas de ruido fijo de las demás corridas DCGAN", "todas colapsan", annex=True)
    for i, r in enumerate(["base", "hinge", "mbstd", "noise", "ttur", "sn_bias"]):
        d.image(FIG / f"ruido_fijo_{r}.png", M + (i % 2) * 448, H - 76 - (i // 2) * 142, 440, 134)
    d.slide("Anexo: muestras del modelo final vs. datos reales", "64 muestras con la semilla 123", annex=True)
    d.image(FIG / "muestras_wgangp_final.png", M, H - 80, 430, 400, "WGAN-GP (modelo final)")
    d.image(FIG / "muestras_reales.png", 494, H - 80, 430, 400, "reales")
    d.save()
    print(f"presentacion.pdf: {d.n} páginas (12 diapositivas + {d.n - 12} anexos)")


if __name__ == "__main__":
    build()
