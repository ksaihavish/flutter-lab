"""Render the generated paper draft (Markdown) to a self-contained HTML file and a PDF.

Only the Markdown the generator writes is supported: headings, paragraphs, nested bullet lists, pipe tables,
images, bold, italic and inline code. Standard library only; the PDF is printed by a local Chrome or Edge in
headless mode (no LaTeX needed). For a venue template, convert the Markdown instead (see submission-checklist.md).
"""

from __future__ import annotations

import base64
import html
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Times New Roman", Georgia, serif; font-size: 10.5pt; line-height: 1.38; color: #111;
       background: #fff; max-width: 175mm; margin: 0 auto; }
h1 { font-size: 17pt; line-height: 1.25; text-align: center; margin: 0 0 6pt; }
h1 + p { text-align: center; font-size: 11pt; margin-bottom: 6pt; }
h1 + p + p { text-align: center; font-size: 8.5pt; color: #444; margin-bottom: 14pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; border-bottom: 0.6pt solid #999; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
ul { margin: 0 0 7pt; padding-left: 16pt; }
li { margin: 0 0 3pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8pt; width: 100%; page-break-inside: avoid; }
th, td { border: 0.5pt solid #bbb; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.5pt; }
img { display: block; max-width: 100%; max-height: 120mm; margin: 4pt auto 12pt; page-break-inside: avoid; }
"""

_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", out)
    return out


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _image(src: str, alt: str, base: Path) -> str:
    path = (base / src).resolve()
    if path.exists():
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        src = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    return f'<img src="{html.escape(src)}" alt="{html.escape(alt)}">'


def markdown_to_html(md: str, base: Path, title: str = "Paper draft") -> str:
    """Convert the draft's Markdown subset to one HTML page; images under ``base`` are embedded."""
    lines = md.splitlines()
    body: list[str] = []
    para: list[str] = []
    list_depth = 0

    def flush_para() -> None:
        if para:
            body.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    def close_lists(depth: int = 0) -> None:
        nonlocal list_depth
        while list_depth > depth:
            body.append("</li></ul>")
            list_depth -= 1

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_para()
            close_lists()
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        bullet = re.match(r"^(\s*)-\s+(.*)$", line)
        if heading:
            flush_para()
            close_lists()
            n = len(heading.group(1))
            body.append(f"<h{n}>{_inline(heading.group(2))}</h{n}>")
        elif stripped.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip()):
            flush_para()
            close_lists()
            rows = [f"<tr>{''.join(f'<th>{_inline(c)}</th>' for c in _cells(stripped))}</tr>"]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(f"<tr>{''.join(f'<td>{_inline(c)}</td>' for c in _cells(lines[i]))}</tr>")
                i += 1
            body.append(f"<table>{''.join(rows)}</table>")
            continue
        elif _IMAGE.match(stripped):
            flush_para()
            close_lists()
            m = _IMAGE.match(stripped)
            assert m is not None
            body.append(_image(m.group(2), m.group(1), base))
        elif bullet:
            flush_para()
            depth = len(bullet.group(1)) // 2 + 1
            if depth > list_depth:
                while list_depth < depth:
                    body.append("<ul><li>")
                    list_depth += 1
            else:
                close_lists(depth)
                body.append("</li><li>")
            body.append(_inline(bullet.group(2)))
        elif list_depth:
            body.append(" " + _inline(stripped))  # continuation of a list item
        else:
            para.append(stripped)
        i += 1
    flush_para()
    close_lists()
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body>\n" + "\n".join(body) + "\n</body></html>\n")


def find_browser() -> str | None:
    """A Chrome, Chromium or Edge executable for headless printing (``FLUTTER_AI_BROWSER`` overrides)."""
    env = os.environ.get("FLUTTER_AI_BROWSER")
    if env and Path(env).exists():
        return env
    for name in ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for root in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")):
        if not root:
            continue
        for rel in ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe"):
            p = Path(root) / rel
            if p.exists():
                return str(p)
    return None


def render_pdf(md_path: Path, pdf_path: Path | None = None, timeout_s: float = 120.0) -> Path:
    """Print the draft to PDF with a headless browser. Raises FileNotFoundError if no browser is found."""
    md_path = Path(md_path)
    pdf_path = Path(pdf_path or md_path.with_suffix(".pdf"))
    md = md_path.read_text(encoding="utf-8")
    title = md.splitlines()[0].lstrip("# ").strip() if md else "Paper draft"
    page = markdown_to_html(md, md_path.parent, title)
    browser = find_browser()
    if browser is None:
        raise FileNotFoundError("No Chrome/Edge found for PDF printing; set FLUTTER_AI_BROWSER to its executable.")
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "paper.html"
        html_path.write_text(page, encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf_path.resolve()}",
                        html_path.resolve().as_uri()],
                       check=True, timeout=timeout_s, capture_output=True)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"The browser did not write {pdf_path}")
    return pdf_path
