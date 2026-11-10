import logging
import re
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.audit import Audit
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.services import report_service

log = logging.getLogger(__name__)

_FONT_FILES = {
    "DejaVuSans": "DejaVuSans.ttf",
    "DejaVuSans-Bold": "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
    "DejaVuSans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
    "DejaVuSansMono": "DejaVuSansMono.ttf",
}
_fonts: tuple[str, str, str, str] | None = None      # (regular, bold, italic, mono)
CODE_WIDTH = 100                                     # characters per line in code blocks
DEFAULT_WIDTH = 170 * 2.8346                         # 170 mm in points (A4 minus 20 mm margins)

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_LIST = re.compile(r"^(?:([-*+\u2022])|(\d{1,3})[.)])\s+(.*)$")
_TABLE_ROW = re.compile(r"^\|.*\|$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
_RULE = re.compile(r"^[-*_]{3,}$")
_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")


# ---------- fonts and styles ----------

def _font_dir() -> Path | None:
    candidates: list[Path] = []
    if settings.PDF_FONT_DIR:
        candidates.append(Path(settings.PDF_FONT_DIR))
    try:
        import matplotlib
        candidates.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except Exception:
        pass
    for folder in candidates:
        if all((folder / f).is_file() for f in _FONT_FILES.values()):
            return folder
    return None


def _setup_fonts() -> tuple[str, str, str, str]:
    global _fonts
    if _fonts:
        return _fonts
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    folder = _font_dir()
    if folder is None:
        log.warning("DejaVu fonts not found: the PDF falls back to Helvetica and cannot show Greek text. "
                    "Set PDF_FONT_DIR to a folder containing DejaVuSans*.ttf.")
        _fonts = ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier")
        return _fonts
    for name, filename in _FONT_FILES.items():
        pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
    pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
                                  italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-BoldOblique")
    _fonts = ("DejaVuSans", "DejaVuSans-Bold", "DejaVuSans-Oblique", "DejaVuSansMono")
    return _fonts


def _styles(fonts: tuple[str, str, str, str]) -> dict:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle

    regular, bold, _, mono = fonts
    body = ParagraphStyle("body", fontName=regular, fontSize=10, leading=14, spaceAfter=6)
    return {
        "title": ParagraphStyle("title", parent=body, fontName=bold, fontSize=18, leading=22, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=body, fontName=bold, fontSize=14, leading=18, spaceBefore=10, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=body, fontName=bold, fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=body, fontName=bold, fontSize=11, leading=14, spaceBefore=6, spaceAfter=3),
        "body": body,
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=18, bulletIndent=6,
                                 bulletFontName=regular, spaceAfter=3),
        "code": ParagraphStyle("code", parent=body, fontName=mono, fontSize=7.5, leading=9.5, leftIndent=6,
                               spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#333333")),
        "meta_key": ParagraphStyle("meta_key", parent=body, fontName=bold, fontSize=9, leading=12, spaceAfter=0),
        "meta_val": ParagraphStyle("meta_val", parent=body, fontSize=9, leading=12, spaceAfter=0),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11, spaceAfter=0),
        "cell_head": ParagraphStyle("cell_head", parent=body, fontName=bold, fontSize=8.5, leading=11,
                                    spaceAfter=0),
    }


# ---------- markdown subset -> reportlab flowables ----------

def _rich(text: str, mono: str) -> str:
    t = escape(text)                                           # the text is data: never interpreted as markup
    t = _CODE.sub(lambda m: f'<font face="{mono}">{m.group(1)}</font>', t)
    t = _BOLD.sub(r"<b>\1</b>", t)
    t = _ITALIC.sub(r"<i>\1</i>", t)
    return t


def _paragraph(text: str, style, mono: str, bullet: str | None = None):
    from reportlab.platypus import Paragraph
    try:
        return Paragraph(_rich(text, mono), style, bulletText=bullet)
    except Exception:                                          # badly nested markup: fall back to plain text
        return Paragraph(escape(text), style, bulletText=bullet)


def _code_block(lines: list[str], styles: dict):
    """Monospace block. Preformatted does not wrap, so long lines are wrapped here."""
    from reportlab.platypus import Preformatted
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=CODE_WIDTH, replace_whitespace=False,
                                     drop_whitespace=False, break_long_words=True,
                                     break_on_hyphens=False) or [""])
    return Preformatted("\n".join(wrapped), styles["code"])


def _table(lines: list[str], styles: dict, mono: str, width: float) -> list:
    """Markdown table -> reportlab Table. Column widths follow the content length."""
    from reportlab.lib import colors
    from reportlab.platypus import Spacer, Table, TableStyle

    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines if not _TABLE_SEP.match(ln)]
    if not rows:
        return []
    cols = max(len(r) for r in rows)
    rows = [r + [""] * (cols - len(r)) for r in rows]

    weights = [max(6, min(30, max(len(r[j]) for r in rows))) for j in range(cols)]
    total = sum(weights)
    col_widths = [width * w / total for w in weights]

    data = [[_paragraph(c, styles["cell_head"] if i == 0 else styles["cell"], mono) for c in r]
            for i, r in enumerate(rows)]
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [table, Spacer(1, 8)]


def markdown_to_flowables(markdown: str, styles: dict, mono: str, width: float = DEFAULT_WIDTH) -> list:
    """Supports headings, paragraphs, bullet and numbered lists, tables, fenced code blocks,
    **bold**, *italic*, `code`."""
    from reportlab.platypus import Spacer

    flow: list = []
    para: list[str] = []
    items: list[tuple[str, str]] = []                          # (bullet, text)
    code: list[str] = []
    table_lines: list[str] = []
    in_code = False

    def flush_para() -> None:
        if para:
            flow.append(_paragraph(" ".join(para), styles["body"], mono))
            para.clear()

    def flush_items() -> None:
        for bullet, text in items:
            flow.append(_paragraph(text, styles["bullet"], mono, bullet=bullet))
        items.clear()

    def flush_code() -> None:
        if code:
            flow.append(_code_block(code, styles))
            code.clear()

    def flush_table() -> None:
        if table_lines:
            flow.extend(_table(table_lines, styles, mono, width))
            table_lines.clear()

    for raw in markdown.splitlines():
        if raw.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_para()
                flush_items()
                flush_table()
                in_code = True
            continue
        if in_code:
            code.append(raw.rstrip())
            continue

        line = raw.strip()
        if _TABLE_ROW.match(line):
            flush_para()
            flush_items()
            table_lines.append(line)
            continue
        flush_table()                                          # any non-table line ends the table

        if not line:
            flush_para()
            flush_items()
            continue
        if _RULE.match(line):
            flush_para()
            flush_items()
            flow.append(Spacer(1, 6))
            continue
        m = _HEADING.match(line)
        if m:
            flush_para()
            flush_items()
            flow.append(_paragraph(m.group(2), styles[f"h{min(len(m.group(1)), 3)}"], mono))
            continue
        m = _LIST.match(line)
        if m:
            flush_para()
            items.append(("\u2022" if m.group(1) else f"{m.group(2)}.", m.group(3)))
            continue
        if items and raw[:1] in (" ", "\t"):                   # continuation line of a list item
            bullet, text = items[-1]
            items[-1] = (bullet, f"{text} {line}")
            continue
        flush_items()
        para.append(line)
    flush_para()
    flush_items()
    flush_table()
    flush_code()                                               # an unclosed fence is still rendered
    return flow


# ---------- document ----------

def render_summary_pdf(title: str, meta: list[tuple[str, str]], body_markdown: str, footer_note: str) -> bytes:
    """Pure function (no database): builds the PDF. Raises RuntimeError if reportlab is not installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as e:
        raise RuntimeError("PDF export is unavailable: the 'reportlab' package is not installed") from e

    fonts = _setup_fonts()
    regular, mono = fonts[0], fonts[3]
    styles = _styles(fonts)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm, title=title, author="XAI Model Auditor")

    flow: list = [Paragraph(escape(title), styles["title"])]
    if meta:
        key_width = 52 * mm
        rows = [[Paragraph(escape(k), styles["meta_key"]), Paragraph(escape(v), styles["meta_val"])]
                for k, v in meta]
        table = Table(rows, colWidths=[key_width, doc.width - key_width], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        flow += [table, Spacer(1, 12)]
    flow += markdown_to_flowables(body_markdown, styles, mono, doc.width)

    def footer(canvas, d) -> None:
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(d.leftMargin, 10 * mm, footer_note)
        canvas.drawRightString(A4[0] - d.rightMargin, 10 * mm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def _num(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _when(dt: datetime | None) -> str:
    return "-" if dt is None else dt.strftime("%Y-%m-%d %H:%M UTC")


def summary_pdf(db: Session, audit: Audit) -> tuple[bytes, str]:
    """Builds the PDF of a finished summary. Returns (pdf bytes, file name)."""
    status = audit.llm_summary_status
    if status is None:
        raise HTTPException(404, "No summary has been requested for this audit")
    if status != "done":
        raise HTTPException(409, f"Summary is '{status}': the PDF is available when it is 'done'")

    report = report_service.build_audit_report(db, audit)
    model, dataset = db.get(MLModel, audit.model_id), db.get(Dataset, audit.dataset_id)
    params = audit.llm_summary_params or {}
    dry_run = audit.llm_summary_model == "dry-run"
    tags = ", ".join(f"{s.key} {s.count}" for s in report.by_auto_tag if s.count) or "-"
    classes = ", ".join(f"{c.class_name} ({c.detections})" for c in report.classes[:6]) or "-"
    meta = [
        ("Audit", str(audit.id)),
        ("Created", _when(audit.created_at)),
        ("Audit status", audit.status),
        ("Model", model.name if model else "-"),
        ("Dataset", dataset.name if dataset else "-"),
        ("Images", f"{report.processed_images} processed, {report.failed_images} failed, "
                   f"{report.evaluated_images} evaluated"),
        ("Auto tags", tags),
        ("Detected classes", classes),
        ("Avg. model confidence", _num(report.avg_confidence, 3)),
        ("Avg. reviewer trust", _num(report.avg_trust, 1)),
        ("Avg. agreement", _num(report.avg_agreement, 3)),
        ("Summary generated", _when(audit.llm_summary_finished_at)),
        ("Summary model", audit.llm_summary_model or "-"),
        ("Summary language", {"en": "English", "el": "Greek"}.get(params.get("language", "en"), "-")),
    ]
    if dry_run:
        meta.append(("Mode", "DRY RUN (no LLM was called)"))
    footer = ("Local dry-run summary (no LLM). Verify the figures against the audit report." if dry_run
              else "AI-generated summary. Verify the figures against the audit report.")
    try:
        pdf = render_summary_pdf("Audit summary (dry run)" if dry_run else "Audit summary",
                                 meta, audit.llm_summary or "", footer)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return pdf, f"audit_{str(audit.id)[:8]}_summary.pdf"