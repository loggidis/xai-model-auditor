import pytest

pytest.importorskip("reportlab")
from app.services.pdf_service import render_summary_pdf

GREEK_WORD = "\u039a\u03b5\u03af\u03bc\u03b5\u03bd\u03bf"          # a Greek word, to check the font coverage


def test_pdf_renders_markdown_and_greek():
    md = (
        "# Title\n\n## Overview\n"
        "Plain **bold** and *italic* with `code`, <tags> & an ampersand.\n\n"
        "- first\n- second\n  continued\n\n1. one\n2. two\n\n---\n\n"
        f"{GREEK_WORD}\n"
    )
    pdf = render_summary_pdf("Audit summary", [("Audit", "abc"), ("Model", "m")], md, "footer")
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500


def test_pdf_survives_badly_nested_markup_and_long_tokens():
    md = "**bold `code** broken`\n\n" + "x" * 5000 + "\n\n[DRY RUN] " + '{"a":"<b>"}' * 200
    pdf = render_summary_pdf("Audit summary", [], md, "footer")
    assert pdf.startswith(b"%PDF")

def test_pdf_renders_fenced_code_blocks():
    long_line = '{"key": "' + "v" * 400 + '"}'
    md = ("## Prompt preview\n\n```json\n{\n  \"a\": 1,\n  \"b\": [1, 2, 3]\n}\n" + long_line
          + "\n```\n\nText after the block.\n\n```\nunclosed fence")
    pdf = render_summary_pdf("Audit summary (dry run)", [("Mode", "DRY RUN")], md, "footer")
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500