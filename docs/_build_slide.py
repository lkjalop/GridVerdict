"""
Generate GridVerdict architecture slides as a .pptx file.
Run: python docs/_build_slide.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree

# ── Colours ──────────────────────────────────────────────────────────────────
BG          = RGBColor(0x0D, 0x11, 0x17)
BG_PANEL    = RGBColor(0x0F, 0x15, 0x1F)
BG_CHAT     = RGBColor(0x11, 0x1D, 0x2E)
BORDER      = RGBColor(0x1E, 0x30, 0x48)

GREEN       = RGBColor(0x22, 0xC5, 0x5E)
BLUE        = RGBColor(0x60, 0xA5, 0xFA)
CYAN        = RGBColor(0x38, 0xBD, 0xF8)
PURPLE      = RGBColor(0xA7, 0x8B, 0xFA)
AMBER       = RGBColor(0xFB, 0xBF, 0x24)
ORANGE      = RGBColor(0xFB, 0x92, 0x3C)
RED_SOFT    = RGBColor(0xF8, 0x71, 0x71)

TEXT_PRI    = RGBColor(0xDD, 0xE6, 0xF0)
TEXT_SEC    = RGBColor(0x8A, 0xA0, 0xB8)
TEXT_MUT    = RGBColor(0x3A, 0x50, 0x70)
TEXT_DIM    = RGBColor(0x2D, 0x40, 0x60)

# Stage accent colours  (name, border-rgb, bullet-rgb)
STAGES = [
    ("QUERY",    RGBColor(0x1E,0x4A,0x8A), BLUE),
    ("DECOMPOSE",RGBColor(0x17,0x60,0x90), CYAN),
    ("GATHER",   RGBColor(0x14,0x5A,0x38), GREEN),
    ("DEEPEN",   RGBColor(0x3D,0x2D,0x8A), PURPLE),
    ("PLAN",     RGBColor(0x7A,0x5A,0x12), AMBER),
    ("VERDICT",  RGBColor(0x9A,0x58,0x10), ORANGE),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def rgb(r,g,b): return RGBColor(r,g,b)

def add_rect(slide, x, y, w, h, fill_rgb, border_rgb=None, border_pt=1.5, radius=0):
    """Add a rounded rectangle with solid fill."""
    from pptx.util import Emu
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        x, y, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    if border_rgb:
        shape.line.color.rgb = border_rgb
        shape.line.width = Pt(border_pt)
    else:
        shape.line.fill.background()
    # Rounded corners via XML
    if radius:
        sp = shape._element
        spPr = sp.find('.//' + nsmap.qn('p:spPr'))
        if spPr is None:
            spPr = sp.find('.//' + nsmap.qn('a:spPr'))
        prstGeom = spPr.find(nsmap.qn('a:prstGeom'))
        if prstGeom is not None:
            prstGeom.set('prst', 'roundRect')
            avLst = prstGeom.find(nsmap.qn('a:avLst'))
            if avLst is None:
                avLst = etree.SubElement(prstGeom, nsmap.qn('a:avLst'))
            avLst.clear()
            gd = etree.SubElement(avLst, nsmap.qn('a:gd'))
            gd.set('name', 'adj')
            gd.set('fmla', f'val {radius}')
    return shape

def add_text(slide, text, x, y, w, h, font_size, bold=False, italic=False,
             color=TEXT_PRI, align=PP_ALIGN.LEFT, wrap=True):
    """Add a text box."""
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Segoe UI"
    return txb

def add_multiline(slide, lines, x, y, w, h, font_size=11, color=TEXT_SEC,
                  line_color=None, bold_first=False, first_color=None, first_size=None):
    """Add a text box with multiple lines (list of strings)."""
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf  = txb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = line
        run.font.name = "Segoe UI"
        run.font.size  = Pt(first_size if (i == 0 and first_size) else font_size)
        run.font.bold  = (i == 0 and bold_first)
        run.font.color.rgb = (first_color if (i == 0 and first_color) else
                              (line_color if line_color else color))
    return txb


# ── SLIDE 1: 5 Deterministic Layers ──────────────────────────────────────────

def build_slide1(prs):
    W = Inches(13.33)   # 16:9 width
    H = Inches(7.5)

    slide_layout = prs.slide_layouts[6]   # blank
    slide = prs.slides.add_slide(slide_layout)

    # Background
    add_rect(slide, 0, 0, W, H, BG)

    # ── Header bar ──────────────────────────────────────────────
    HDR_H = Inches(0.55)
    add_rect(slide, 0, 0, W, HDR_H, rgb(0x0D,0x11,0x17), border_rgb=rgb(0x1A,0x25,0x36))

    # Brand
    bx = slide.shapes.add_textbox(Inches(0.35), Inches(0.08), Inches(1.6), Inches(0.4))
    tf = bx.text_frame
    p  = tf.paragraphs[0]
    r1 = p.add_run(); r1.text = "Grid"; r1.font.size = Pt(22); r1.font.bold = True
    r1.font.color.rgb = TEXT_PRI; r1.font.name = "Segoe UI"
    r2 = p.add_run(); r2.text = "Verdict"; r2.font.size = Pt(22); r2.font.bold = True
    r2.font.color.rgb = GREEN; r2.font.name = "Segoe UI"

    # Region chip
    add_rect(slide, Inches(1.9), Inches(0.1), Inches(0.8), Inches(0.35),
             rgb(0x1A,0x25,0x36), rgb(0x26,0x34,0x48), 1)
    add_text(slide, "NSW1 ▾", Inches(1.93), Inches(0.11), Inches(0.75), Inches(0.32),
             10, color=TEXT_SEC, align=PP_ALIGN.CENTER)

    # Price
    add_text(slide, "● $66.53/MWh", Inches(2.8), Inches(0.1), Inches(1.6), Inches(0.36),
             15, bold=True, color=TEXT_PRI)

    # Regime
    add_rect(slide, Inches(4.45), Inches(0.1), Inches(0.85), Inches(0.35),
             rgb(0x0F,0x2D,0x1F), GREEN, 1)
    add_text(slide, "NORMAL", Inches(4.46), Inches(0.11), Inches(0.83), Inches(0.32),
             10, bold=True, color=GREEN, align=PP_ALIGN.CENTER)

    # Model chips
    chips = ["LNN ✓", "LEAR ✓", "QRA ✓", "TCN ✓", "LNN_LTC ✓", "META ✓"]
    cx = Inches(8.6)
    for chip in chips:
        add_rect(slide, cx, Inches(0.1), Inches(0.75), Inches(0.35),
                 rgb(0x13,0x1D,0x2B), rgb(0x26,0x34,0x48), 1)
        add_text(slide, chip, cx+Inches(0.03), Inches(0.11), Inches(0.72), Inches(0.32),
                 9, color=GREEN, align=PP_ALIGN.CENTER)
        cx += Inches(0.82)

    add_text(slide, "● LIVE", Inches(13.05), Inches(0.14), Inches(0.55), Inches(0.28),
             11, bold=True, color=GREEN)

    # ── Left: Chat panel ────────────────────────────────────────
    PX = Inches(0.35)
    PY = Inches(0.7)
    PW = Inches(3.55)

    add_text(slide, "CHAT", PX, PY, PW, Inches(0.22), 8,
             bold=True, color=TEXT_DIM)

    # You bubble
    add_rect(slide, PX, Inches(0.97), PW, Inches(0.9),
             rgb(0x11,0x1D,0x2E), BORDER, 1.2, radius=30000)
    add_text(slide, "You", PX+Inches(0.15), Inches(1.0), PW-Inches(0.2), Inches(0.22),
             9, bold=True, color=GREEN)
    add_multiline(slide,
        ["What are prices on Monday June 8th?", "Which to buy?"],
        PX+Inches(0.15), Inches(1.22), PW-Inches(0.2), Inches(0.58),
        font_size=14, color=TEXT_PRI)

    # Answer card
    add_rect(slide, PX, Inches(1.95), PW, Inches(3.55),
             BG_PANEL, BORDER, 1.2, radius=30000)

    # Badge row
    add_rect(slide, PX+Inches(0.15), Inches(2.08), Inches(1.35), Inches(0.3),
             rgb(0x49,0x22,0x00), rgb(0xFB,0x92,0x3C), 1)
    add_text(slide, "LOW CONFIDENCE", PX+Inches(0.15), Inches(2.08),
             Inches(1.35), Inches(0.3), 8, bold=True,
             color=ORANGE, align=PP_ALIGN.CENTER)

    add_rect(slide, PX+Inches(1.57), Inches(2.08), Inches(0.7), Inches(0.3),
             rgb(0x1E,0x16,0x40), rgb(0x6D,0x28,0xD9,), 1)
    add_text(slide, "HOLD", PX+Inches(1.57), Inches(2.08),
             Inches(0.7), Inches(0.3), 8, bold=True,
             color=PURPLE, align=PP_ALIGN.CENTER)

    add_text(slide, "75%", PX+Inches(2.35), Inches(2.07),
             Inches(0.5), Inches(0.32), 14, bold=True, color=TEXT_MUT)

    # Answer text
    add_multiline(slide, [
        "GridVerdict does not have live data for June 8",
        "— that date hasn't occurred yet.",
        "",
        "Evidence-grounded estimate, built from 3 years",
        "of June history + the BOM 7-day forecast —",
        "with every gap named.",
    ], PX+Inches(0.15), Inches(2.46), PW-Inches(0.3), Inches(2.3),
       font_size=12, color=TEXT_SEC)

    add_text(slide, "See answer →", PX+Inches(0.15), Inches(5.15),
             Inches(1.5), Inches(0.28), 11, color=GREEN)

    # VS box
    add_rect(slide, PX, Inches(5.52), PW, Inches(0.85),
             rgb(0x09,0x0E,0x17), BORDER, 1, radius=20000)
    add_multiline(slide, [
        "A raw LLM gives a fluent guess.",
        "GridVerdict: verdict + confidence score + missing list.",
    ], PX+Inches(0.15), Inches(5.6), PW-Inches(0.3), Inches(0.7),
       font_size=11, color=TEXT_MUT,
       bold_first=False, first_color=rgb(0xF8,0x71,0x71))

    # ── Right: Pipeline ─────────────────────────────────────────
    PIPE_X  = Inches(4.1)
    PIPE_Y  = Inches(0.72)
    PIPE_W  = W - PIPE_X - Inches(0.3)
    BOX_H   = Inches(5.1)

    add_text(slide, "HOW THE ANSWER IS BUILT — 5 DETERMINISTIC LAYERS",
             PIPE_X, PIPE_Y, PIPE_W, Inches(0.25), 9,
             bold=True, color=TEXT_DIM)

    # Stage definitions
    stage_data = [
        ("QUERY",      "natural language",
         ["· region + raw text", "· session carry-fwd", "· last 3 Q&A merged"]),
        ("DECOMPOSE",  "rules first, LLM enriches",
         ["· regex → intent", "· entity + flags", "· LLM adds sub-Qs",
          "· routing key set", "  ↳ adjacent path exits here"]),
        ("GATHER",     "all sources, parallel",
         ["· AEMO live + notices", "· LNN·LEAR·QRA·TCN",
          "· HippoGraph analogs", "· FCAS prices",
          "· unit dispatch", "· market drivers",
          "· live feed context", "· weather consensus"]),
        ("DEEPEN",     "sequential, DB-safe",
         ["· TemporalRAG 4h", "· fuel mix split",
          "· BOM 7-day", "· 3yr P10/P50/P90",
          "· OpenNEM trend", "· causal chain inference"]),
        ("PLAN",       "routes by output type",
         ["· 14 planners", "· 8 adjacent handlers",
          "· evidence + missing", "· claim map built"]),
        ("VERDICT",    "",
         ["HOLD", "75% confidence", "+ claim verifier guard"]),
    ]

    n = len(stage_data)
    ARROW_W = Inches(0.28)
    # Total space for boxes = PIPE_W minus arrows
    BOX_W_EACH = (PIPE_W - ARROW_W * (n - 1)) / n
    BOX_Y = Inches(1.03)

    for i, (name, subtitle, bullets) in enumerate(stage_data):
        bx = PIPE_X + (BOX_W_EACH + ARROW_W) * i
        border_c, name_c = STAGES[i][1], STAGES[i][2]
        bg_c = BG_PANEL

        add_rect(slide, bx, BOX_Y, BOX_W_EACH, BOX_H,
                 bg_c, border_c, 1.5, radius=28000)

        # Stage name
        add_text(slide, name, bx+Inches(0.1), BOX_Y+Inches(0.12),
                 BOX_W_EACH-Inches(0.15), Inches(0.3),
                 11, bold=True, color=name_c)

        # Subtitle
        if subtitle:
            add_text(slide, subtitle, bx+Inches(0.1), BOX_Y+Inches(0.42),
                     BOX_W_EACH-Inches(0.15), Inches(0.22),
                     8, italic=True, color=rgb(
                         int(name_c[0]*0.5), int(name_c[1]*0.5), int(name_c[2]*0.5)))

        # Bullets / verdict content
        bullet_y = BOX_Y + Inches(0.68)
        if name == "VERDICT":
            add_text(slide, "HOLD", bx+Inches(0.1), bullet_y, BOX_W_EACH-Inches(0.15),
                     Inches(0.5), 22, bold=True, color=ORANGE)
            add_text(slide, "75% confidence", bx+Inches(0.1), bullet_y+Inches(0.52),
                     BOX_W_EACH-Inches(0.15), Inches(0.28), 10, color=TEXT_SEC)
            add_text(slide, "+ claim verifier guard", bx+Inches(0.1), bullet_y+Inches(0.82),
                     BOX_W_EACH-Inches(0.15), Inches(0.28), 9, italic=True,
                     color=rgb(0x9A,0x58,0x10))
        else:
            bullet_lines = []
            for b in bullets:
                bullet_lines.append(b)
            add_multiline(slide, bullet_lines, bx+Inches(0.1), bullet_y,
                          BOX_W_EACH-Inches(0.15), BOX_H-Inches(0.75),
                          font_size=10, color=TEXT_MUT)

        # Arrow between boxes
        if i < n - 1:
            ax = bx + BOX_W_EACH + Inches(0.02)
            add_text(slide, "▶", ax, BOX_Y + BOX_H/2 - Inches(0.25),
                     ARROW_W - Inches(0.04), Inches(0.35),
                     14, color=TEXT_DIM, align=PP_ALIGN.CENTER)

    # Adjacent path callout
    ADJ_Y = Inches(6.25)
    add_rect(slide, PIPE_X, ADJ_Y, PIPE_W, Inches(0.9),
             rgb(0x09,0x10,0x1A), rgb(0x1A,0x2D,0x40), 1, radius=15000)
    # Left accent bar
    add_rect(slide, PIPE_X, ADJ_Y, Inches(0.04), Inches(0.9), CYAN)
    add_text(slide, "ADJACENT PATH", PIPE_X+Inches(0.12), ADJ_Y+Inches(0.08),
             Inches(1.15), Inches(0.28), 8, bold=True, color=CYAN)
    add_multiline(slide, [
        "Non-NEM queries (household solar, gas markets, policy, macro, LCOE) exit at DECOMPOSE →",
        "route to 1 of 8 adjacent handlers — no scatter-gather. Each returns a PARTIAL_SCOPE verdict",
        "with an explicit data-boundary. Clients: GBB gas spot · ISP scenarios · LCOE sensitivity · fiscal budget.",
    ], PIPE_X+Inches(1.35), ADJ_Y+Inches(0.05), PIPE_W-Inches(1.4), Inches(0.82),
       font_size=10, color=TEXT_MUT)

    # Footer
    add_rect(slide, 0, Inches(7.22), W, Inches(0.28), rgb(0x09,0x0E,0x17), rgb(0x1A,0x25,0x36), 0.5)
    add_text(slide,
             'Not "what sounds right" -- but "what can we prove from live data, and what can\'t we?"',
             0, Inches(7.24), W, Inches(0.24), 12,
             italic=True, color=GREEN, align=PP_ALIGN.CENTER)


# ── SLIDE 2: Four Security Gates ─────────────────────────────────────────────

def build_slide2(prs):
    W = Inches(13.33)
    H = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, W, H, BG)

    # ── Header (same as slide 1) ─────────────────────────────────
    HDR_H = Inches(0.55)
    add_rect(slide, 0, 0, W, HDR_H, rgb(0x0D,0x11,0x17), rgb(0x1A,0x25,0x36))
    bx = slide.shapes.add_textbox(Inches(0.35), Inches(0.08), Inches(1.6), Inches(0.4))
    tf = bx.text_frame; p = tf.paragraphs[0]
    r1=p.add_run(); r1.text="Grid"; r1.font.size=Pt(22); r1.font.bold=True
    r1.font.color.rgb=TEXT_PRI; r1.font.name="Segoe UI"
    r2=p.add_run(); r2.text="Verdict"; r2.font.size=Pt(22); r2.font.bold=True
    r2.font.color.rgb=GREEN; r2.font.name="Segoe UI"
    add_rect(slide, Inches(1.9), Inches(0.1), Inches(0.8), Inches(0.35),
             rgb(0x1A,0x25,0x36), rgb(0x26,0x34,0x48))
    add_text(slide, "NSW1 ▾", Inches(1.93), Inches(0.11), Inches(0.75), Inches(0.32),
             10, color=TEXT_SEC, align=PP_ALIGN.CENTER)
    add_text(slide, "● $66.53/MWh", Inches(2.8), Inches(0.1), Inches(1.6), Inches(0.36),
             15, bold=True, color=TEXT_PRI)
    add_rect(slide, Inches(4.45), Inches(0.1), Inches(0.85), Inches(0.35),
             rgb(0x0F,0x2D,0x1F), GREEN)
    add_text(slide, "NORMAL", Inches(4.46), Inches(0.11), Inches(0.83), Inches(0.32),
             10, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    chips = ["LNN ✓","LEAR ✓","QRA ✓","TCN ✓","LNN_LTC ✓","META ✓"]
    cx = Inches(8.6)
    for chip in chips:
        add_rect(slide, cx, Inches(0.1), Inches(0.75), Inches(0.35),
                 rgb(0x13,0x1D,0x2B), rgb(0x26,0x34,0x48))
        add_text(slide, chip, cx+Inches(0.03), Inches(0.11), Inches(0.72), Inches(0.32),
                 9, color=GREEN, align=PP_ALIGN.CENTER)
        cx += Inches(0.82)
    add_text(slide, "● LIVE", Inches(13.05), Inches(0.14), Inches(0.55), Inches(0.28),
             11, bold=True, color=GREEN)

    # ── Left: chat ───────────────────────────────────────────────
    PX = Inches(0.35); PY = Inches(0.7); PW = Inches(3.55)
    add_text(slide, "CHAT", PX, PY, PW, Inches(0.22), 8, bold=True, color=TEXT_DIM)
    add_rect(slide, PX, Inches(0.97), PW, Inches(0.9),
             rgb(0x11,0x1D,0x2E), BORDER, 1.2, radius=30000)
    add_text(slide, "You", PX+Inches(0.15), Inches(1.0), PW-Inches(0.2), Inches(0.22),
             9, bold=True, color=GREEN)
    add_multiline(slide,
        ["Why pay double for coal now?", "Wind was $15–25 this morning."],
        PX+Inches(0.15), Inches(1.22), PW-Inches(0.2), Inches(0.58),
        font_size=14, color=TEXT_PRI)

    add_rect(slide, PX, Inches(1.95), PW, Inches(3.55), BG_PANEL, BORDER, 1.2, radius=30000)
    add_rect(slide, PX+Inches(0.15), Inches(2.08), Inches(1.35), Inches(0.3),
             rgb(0x3A,0x10,0x10), rgb(0xF8,0x71,0x71))
    add_text(slide, "INSUFFICIENT DATA", PX+Inches(0.15), Inches(2.08),
             Inches(1.35), Inches(0.3), 8, bold=True,
             color=rgb(0xF8,0x71,0x71), align=PP_ALIGN.CENTER)
    add_rect(slide, PX+Inches(1.57), Inches(2.08), Inches(0.7), Inches(0.3),
             rgb(0x1A,0x14,0x00), rgb(0x8A,0x6A,0x1A))
    add_text(slide, "MONITOR", PX+Inches(1.57), Inches(2.08), Inches(0.7), Inches(0.3),
             8, bold=True, color=AMBER, align=PP_ALIGN.CENTER)
    add_text(slide, "35%", PX+Inches(2.35), Inches(2.07), Inches(0.5), Inches(0.32),
             14, bold=True, color=TEXT_MUT)
    add_multiline(slide, [
        "The price shift is the NEM's normal",
        "diurnal cycle — not a market fault.",
        "",
        "Coal modelled at $73. Low confidence",
        "= a data gap, not a model failure.",
        "The system says so instead of guessing.",
    ], PX+Inches(0.15), Inches(2.46), PW-Inches(0.3), Inches(2.3),
       font_size=12, color=TEXT_SEC)
    add_text(slide, "See answer →", PX+Inches(0.15), Inches(5.15),
             Inches(1.5), Inches(0.28), 11, color=GREEN)

    # ── Right: Security gates ────────────────────────────────────
    GX = Inches(4.1)
    GY = Inches(0.7)
    GW = W - GX - Inches(0.3)

    add_text(slide, "CHECKED AT EVERY STEP — FOUR SECURITY GATES RIDE THE PIPELINE",
             GX, GY, GW, Inches(0.25), 9, bold=True, color=TEXT_DIM)

    # Gate numbering legend
    legend_txt = "①②③④  =  a regex pass that can  HALT  the request on risk (score ≥ 80)"
    add_multiline(slide, [legend_txt],
                  GX, Inches(1.02), GW, Inches(0.28),
                  font_size=11, color=TEXT_SEC)

    # Gate numbers above boxes
    gate_labels = ["① input", "② intent", "③ external data", "④ final answer"]
    gate_colours = [BLUE, CYAN, GREEN, ORANGE]
    N = 5  # boxes: QUERY, DECOMPOSE, GATHER, PLAN, VERDICT
    GATE_BOX_W = GW / N
    BOX_Y2 = Inches(1.85)
    BOX_H2 = Inches(3.2)

    # Draw gate number labels (only on QUERY, DECOMPOSE, GATHER, VERDICT = positions 0,1,2,4)
    gate_positions = [(0, gate_labels[0], gate_colours[0]),
                      (1, gate_labels[1], gate_colours[1]),
                      (2, gate_labels[2], gate_colours[2]),
                      (4, gate_labels[3], gate_colours[3])]

    for pos, lbl, col in gate_positions:
        lx = GX + GATE_BOX_W * pos
        add_text(slide, lbl, lx, Inches(1.35), GATE_BOX_W, Inches(0.25),
                 9, bold=True, color=col, align=PP_ALIGN.CENTER)
        # Arrow down
        add_text(slide, "↓", lx, Inches(1.6), GATE_BOX_W, Inches(0.22),
                 13, color=col, align=PP_ALIGN.CENTER)

    # Boxes
    sec_stages = [
        ("QUERY",    "input",    ["· raw text", "· region", "· session"]),
        ("DECOMPOSE","intent",   ["· regex first", "· LLM sub-Qs", "· conf score"]),
        ("GATHER",   "parallel", ["· AEMO · LNN", "· LEAR · QRA", "· unit · fuel", "· FCAS · wx"]),
        ("PLAN",     "by type",  ["· 14 planners", "· 8 adj handlers", "· evidence", "· missing[]"]),
        ("VERDICT",  "+ guard",  ["INSUFF.", "DATA 35%", "· claim chk"]),
    ]

    for i, (name, sub, buls) in enumerate(sec_stages):
        bx2 = GX + GATE_BOX_W * i + Inches(0.08)
        bw2 = GATE_BOX_W - Inches(0.16)
        col = STAGES[i][1] if i < 5 else STAGES[4][1]
        nc  = STAGES[i][2] if i < 5 else STAGES[4][2]
        add_rect(slide, bx2, BOX_Y2, bw2, BOX_H2,
                 BG_PANEL, col, 1.5, radius=25000)
        add_text(slide, name, bx2+Inches(0.1), BOX_Y2+Inches(0.1),
                 bw2-Inches(0.15), Inches(0.28), 11, bold=True, color=nc)
        add_text(slide, sub, bx2+Inches(0.1), BOX_Y2+Inches(0.4),
                 bw2-Inches(0.15), Inches(0.22), 8, italic=True, color=TEXT_DIM)
        add_multiline(slide, buls, bx2+Inches(0.1), BOX_Y2+Inches(0.68),
                      bw2-Inches(0.15), BOX_H2-Inches(0.75),
                      font_size=10, color=TEXT_MUT)
        if i < N - 1:
            add_text(slide, "▶",
                     GX + GATE_BOX_W*(i+1)-Inches(0.16),
                     BOX_Y2 + BOX_H2/2 - Inches(0.2),
                     Inches(0.22), Inches(0.3), 13, color=TEXT_DIM, align=PP_ALIGN.CENTER)

    # Three principle bullets
    principles = [
        ("●", BLUE,   "Rules decide routing.  The LLM only enriches wording — it never picks the answer."),
        ("●", AMBER,  "No source for a claim → it's marked MISSING, not guessed."),
        ("●", GREEN,  "Every model down → it still runs on rules alone. Each gate logs to an audit table."),
    ]
    py = Inches(5.22)
    for dot_ch, dot_col, txt in principles:
        add_rect(slide, GX, py, Inches(0.06), Inches(0.26), dot_col)
        add_text(slide, txt, GX+Inches(0.14), py, GW-Inches(0.18), Inches(0.28),
                 11, color=TEXT_SEC)
        py += Inches(0.38)

    # Footer
    add_rect(slide, 0, Inches(7.22), W, Inches(0.28), rgb(0x09,0x0E,0x17), rgb(0x1A,0x25,0x36))
    add_text(slide,
             "① input  ② intent  ③ external data  ④ final answer — four checks, in-process, all logged to audit.",
             0, Inches(7.24), W, Inches(0.24), 12,
             italic=True, color=GREEN, align=PP_ALIGN.CENTER)


# ── SLIDE 3: Same Pipeline, Two Honest Outcomes ───────────────────────────────

def build_slide3(prs):
    W = Inches(13.33)
    H = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, W, H, BG)

    # ── Header ───────────────────────────────────────────────────
    HDR_H = Inches(0.55)
    add_rect(slide, 0, 0, W, HDR_H, rgb(0x0D,0x11,0x17), rgb(0x1A,0x25,0x36))
    bx = slide.shapes.add_textbox(Inches(0.35), Inches(0.08), Inches(1.6), Inches(0.4))
    tf = bx.text_frame; p = tf.paragraphs[0]
    r1=p.add_run(); r1.text="Grid"; r1.font.size=Pt(22); r1.font.bold=True
    r1.font.color.rgb=TEXT_PRI; r1.font.name="Segoe UI"
    r2=p.add_run(); r2.text="Verdict"; r2.font.size=Pt(22); r2.font.bold=True
    r2.font.color.rgb=GREEN; r2.font.name="Segoe UI"
    add_rect(slide, Inches(1.9), Inches(0.1), Inches(0.8), Inches(0.35),
             rgb(0x1A,0x25,0x36), rgb(0x26,0x34,0x48))
    add_text(slide, "NSW1 ▾", Inches(1.93), Inches(0.11), Inches(0.75), Inches(0.32),
             10, color=TEXT_SEC, align=PP_ALIGN.CENTER)
    add_text(slide, "● $66.53/MWh", Inches(2.8), Inches(0.1), Inches(1.6), Inches(0.36),
             15, bold=True, color=TEXT_PRI)
    add_rect(slide, Inches(4.45), Inches(0.1), Inches(0.85), Inches(0.35),
             rgb(0x0F,0x2D,0x1F), GREEN)
    add_text(slide, "NORMAL", Inches(4.46), Inches(0.11), Inches(0.83), Inches(0.32),
             10, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
    chips = ["LNN ✓","LEAR ✓","QRA ✓","TCN ✓","LNN_LTC ✓","META ✓"]
    cx = Inches(8.6)
    for chip in chips:
        add_rect(slide, cx, Inches(0.1), Inches(0.75), Inches(0.35),
                 rgb(0x13,0x1D,0x2B), rgb(0x26,0x34,0x48))
        add_text(slide, chip, cx+Inches(0.03), Inches(0.11), Inches(0.72), Inches(0.32),
                 9, color=GREEN, align=PP_ALIGN.CENTER)
        cx += Inches(0.82)
    add_text(slide, "● LIVE", Inches(13.05), Inches(0.14), Inches(0.55), Inches(0.28),
             11, bold=True, color=GREEN)

    # ── Left: chat ───────────────────────────────────────────────
    PX = Inches(0.35); PW = Inches(3.55)
    add_text(slide, "CHAT", PX, Inches(0.7), PW, Inches(0.22), 8, bold=True, color=TEXT_DIM)
    add_rect(slide, PX, Inches(0.97), PW, Inches(0.85),
             rgb(0x11,0x1D,0x2E), BORDER, 1.2, radius=30000)
    add_text(slide, "You", PX+Inches(0.15), Inches(1.0), PW, Inches(0.22),
             9, bold=True, color=GREEN)
    add_text(slide, "What are prices on June 8th?",
             PX+Inches(0.15), Inches(1.22), PW-Inches(0.2), Inches(0.5),
             14, color=TEXT_PRI)

    add_rect(slide, PX, Inches(1.92), PW, Inches(3.58), BG_PANEL, BORDER, 1.2, radius=30000)
    add_rect(slide, PX+Inches(0.15), Inches(2.05), Inches(1.35), Inches(0.3),
             rgb(0x49,0x22,0x00), rgb(0xFB,0x92,0x3C))
    add_text(slide, "LOW CONFIDENCE", PX+Inches(0.15), Inches(2.05),
             Inches(1.35), Inches(0.3), 8, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    add_rect(slide, PX+Inches(1.57), Inches(2.05), Inches(0.7), Inches(0.3),
             rgb(0x1E,0x16,0x40), rgb(0x6D,0x28,0xD9))
    add_text(slide, "HOLD", PX+Inches(1.57), Inches(2.05),
             Inches(0.7), Inches(0.3), 8, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
    add_text(slide, "75%", PX+Inches(2.35), Inches(2.04), Inches(0.5), Inches(0.32),
             14, bold=True, color=TEXT_MUT)
    add_multiline(slide, [
        "EVIDENCE shown. MISSING shown.",
        "",
        "The product names the gap between its",
        "answer and the better answer it could",
        "give with more data.",
        "",
        "Recheck closer to the date.",
    ], PX+Inches(0.15), Inches(2.43), PW-Inches(0.3), Inches(2.4),
       font_size=12, color=TEXT_SEC)
    add_text(slide, "See answer →", PX+Inches(0.15), Inches(5.12),
             Inches(1.5), Inches(0.28), 11, color=GREEN)

    # ── Right: three pipeline traces ─────────────────────────────
    RX = Inches(4.1)
    RW = W - RX - Inches(0.3)

    add_text(slide, "SAME PIPELINE, THREE HONEST OUTCOMES",
             RX, Inches(0.7), RW, Inches(0.25), 9, bold=True, color=TEXT_DIM)

    # Helper: draw a mini pipeline row
    def mini_row(y, query_label, steps, verdict_label, verdict_col):
        """steps = list of (box_label, sub_label) tuples."""
        STEP_W = Inches(1.38)
        ARROW  = Inches(0.22)
        ROW_H  = Inches(1.0)
        row_x  = RX
        # Label
        add_text(slide, query_label, row_x, y-Inches(0.24), RW, Inches(0.22),
                 12, italic=True, color=TEXT_PRI)
        for j, (bname, bsub) in enumerate(steps):
            bc = STAGES[j][1] if j < len(STAGES) else STAGES[-1][1]
            nc = STAGES[j][2] if j < len(STAGES) else STAGES[-1][2]
            add_rect(slide, row_x, y, STEP_W, ROW_H, BG_PANEL, bc, 1.2, radius=18000)
            add_text(slide, bname, row_x+Inches(0.08), y+Inches(0.06),
                     STEP_W-Inches(0.12), Inches(0.26), 10, bold=True, color=nc)
            add_text(slide, bsub, row_x+Inches(0.08), y+Inches(0.34),
                     STEP_W-Inches(0.12), Inches(0.55), 9, italic=True, color=TEXT_MUT)
            row_x += STEP_W
            if j < len(steps) - 1:
                add_text(slide, "▶", row_x, y + ROW_H/2 - Inches(0.18),
                         ARROW, Inches(0.28), 12, color=TEXT_DIM, align=PP_ALIGN.CENTER)
                row_x += ARROW
        # Verdict box
        add_rect(slide, row_x, y, Inches(1.2), ROW_H,
                 rgb(0x1A,0x10,0x04), verdict_col, 1.5, radius=18000)
        add_text(slide, verdict_label, row_x+Inches(0.06), y+Inches(0.06),
                 Inches(1.1), ROW_H - Inches(0.12), 12, bold=True,
                 color=verdict_col, align=PP_ALIGN.CENTER)

    STEPS_A = [
        ("DECOMPOSE", "future date ref"),
        ("GATHER",    "AEMO + 3yr hist"),
        ("DEEPEN",    "BOM 7d limit"),
        ("PLAN",      "Winter table"),
    ]
    mini_row(Inches(1.05), '"June 8th prices?"', STEPS_A, "HOLD\n75%", PURPLE)

    STEPS_B = [
        ("DECOMPOSE", "fuel source"),
        ("GATHER",    "AEMO + forecast"),
        ("DEEPEN",    "fuel mix + history"),
        ("PLAN",      "diurnal cycle"),
    ]
    mini_row(Inches(2.45), '"Why pay double -- wind was $15?"', STEPS_B,
             "INSUFF\nDATA·35%", rgb(0xF8,0x71,0x71))

    STEPS_C = [
        ("DECOMPOSE", "adjacent: LCOE"),
        ("PLAN",      "adj handler\n(no gather)"),
    ]
    # Shorter row — only 2 steps
    mini_row(Inches(3.85), '"Is rooftop solar worth it in Adelaide?"', STEPS_C,
             "PARTIAL\nSCOPE·70%", CYAN)

    # Bottom callout
    add_rect(slide, RX, Inches(5.1), RW, Inches(1.05),
             rgb(0x09,0x10,0x1A), rgb(0x49,0x22,0x00), 1.5, radius=15000)
    add_multiline(slide, [
        "The second answer is low-confidence on purpose.  The data to prove the fuel rank",
        "wasn't there — so it says so, instead of inventing a number.",
        "The third exits at DECOMPOSE — adjacent handler, no scatter-gather, explicit scope boundary.",
    ], RX+Inches(0.2), Inches(5.18), RW-Inches(0.35), Inches(0.92),
       font_size=12, color=TEXT_SEC)

    # Footer
    add_rect(slide, 0, Inches(7.22), W, Inches(0.28), rgb(0x09,0x0E,0x17), rgb(0x1A,0x25,0x36))
    add_text(slide,
             "Built for ANZ energy desks & regulated industries — verdicts you can take to an audit.",
             0, Inches(7.24), W, Inches(0.24), 12,
             italic=True, color=GREEN, align=PP_ALIGN.CENTER)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    build_slide1(prs)
    build_slide2(prs)
    build_slide3(prs)

    out = "docs/GridVerdict_Architecture_v2.pptx"
    prs.save(out)
    print(f"Saved -> {out}")
