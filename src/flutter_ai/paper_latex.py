"""Export the generated paper draft to LaTeX for Overleaf: IEEE conference (IEEEtran) or a single-column article
for Springer journals (IJASS: numbered references, flat folder, figures named Fig1.png ...).

`python -m flutter_ai paper --ieee` / `--springer` writes ``docs/research/<style>/`` (paper.tex, references.bib,
figures) and ``docs/research/paper-<style>-overleaf.zip`` (Overleaf: New project -> Upload project). Citations become
numbered ``\\cite{}`` entries from references.bib; numbers still come only from the result files (via paper.build).
Compile with pdfLaTeX + BibTeX. Standard library only.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .paper import AFFILIATION, AFFILIATIONS, AUTHORS, CORRESPONDING, KEYWORDS, RESEARCH, _bib, build, springer_ref

# Non-ASCII characters the draft uses -> LaTeX (declared in the preamble for pdfLaTeX).
UNICODE = {
    "·": r"\ensuremath{\cdot}", "±": r"\ensuremath{\pm}", "°": r"\ensuremath{^\circ}", "×": r"\ensuremath{\times}",
    "§": r"\S{}", "†": r"\dag{}", "−": r"\ensuremath{-}", "≤": r"\ensuremath{\le}", "≥": r"\ensuremath{\ge}",
    "≈": r"\ensuremath{\approx}", "∈": r"\ensuremath{\in}", "→": r"\ensuremath{\rightarrow}", "√": r"\ensuremath{\surd}",
    "ᵀ": r"\ensuremath{^{\mathsf{T}}}", "α": r"\ensuremath{\alpha}", "β": r"\ensuremath{\beta}",
    "λ": r"\ensuremath{\lambda}", "μ": r"\ensuremath{\mu}", "ξ": r"\ensuremath{\xi}", "σ": r"\ensuremath{\sigma}",
    "ω": r"\ensuremath{\omega}", "Δ": r"\ensuremath{\Delta}", "Σ": r"\ensuremath{\Sigma}",
}
NATIVE = set("öäüéèãñ–—‘’“”")  # handled by pdfLaTeX's utf8 input with T1 fonts
_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
_FIG_CAPTION = re.compile(r"^\*\*Figure \d+\.\*\*\s*(.*)$")
_TAB_CAPTION = re.compile(r"^\*\*Table \d+\.\*\*\s*(.*)$")
_NUMBER = re.compile(r"^\d+(\.\d+)*\.?\s+")


def _esc(text: str) -> str:
    return re.sub(r"[\\&%$#_{}~^]", lambda m: _SPECIAL[m.group()], text)


def tex_inline(text: str) -> str:
    """Markdown inline text -> LaTeX (escaping, code, bold, italic, quotes, U_F, citation markers); $math$ is kept."""
    out = []
    for part in re.split(r"(`[^`]+`|\$[^$\n]+\$)", text):
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append(part)
            continue
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(r"\texttt{" + _esc(part[1:-1]) + "}")
            continue
        s = _esc(part)
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
        s = re.sub(r'"([^"\n]+)"', r"``\1''", s)
        s = s.replace(r"U\_F", "$U_F$")
        out.append(s)
    return re.sub(r"⟦cite:([^⟧]+)⟧", lambda m: r"\cite{" + m.group(1) + "}", "".join(out))


def _blocks(md: str) -> list[tuple]:
    """Split the draft into blocks: ('h', level, text), ('p', text), ('ul', [(depth, text)]),
    ('table', header, rows), ('img', alt, src)."""
    blocks: list[tuple] = []
    para: list[str] = []
    items: list[list] = []
    lines = md.splitlines()

    def flush() -> None:
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()
        if items:
            blocks.append(("ul", [(d, t) for d, t in items]))
            items.clear()

    i = 0
    while i < len(lines):
        line, s = lines[i], lines[i].strip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", s)
        bullet = re.match(r"^(\s*)-\s+(.*)$", line)
        if not s:
            flush()
        elif heading:
            flush()
            blocks.append(("h", len(heading.group(1)), heading.group(2)))
        elif s.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip()):
            flush()
            header = [c.strip() for c in s.strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            blocks.append(("table", header, rows))
            continue
        elif _IMAGE.match(s):
            flush()
            m = _IMAGE.match(s)
            assert m is not None
            blocks.append(("img", m.group(1), m.group(2)))
        elif bullet:
            if para:
                flush()
            items.append([len(bullet.group(1)) // 2, bullet.group(2)])
        elif items:
            items[-1][1] += " " + s
        else:
            para.append(s)
        i += 1
    flush()
    return blocks


def _png_ratio(path: Path) -> float:
    """Width / height of a PNG (from its header); 1.0 if unreadable."""
    try:
        with path.open("rb") as f:
            head = f.read(24)
        w, h = struct.unpack(">II", head[16:24])
        return w / h if h else 1.0
    except (OSError, struct.error):
        return 1.0


def _itemize(items: list[tuple[int, str]]) -> list[str]:
    out, depth = [], -1
    for d, text in items:
        while depth < d:
            out.append(r"\begin{itemize}")
            depth += 1
        while depth > d:
            out.append(r"\end{itemize}")
            depth -= 1
        out.append(r"\item{} " + tex_inline(text))
    out.extend([r"\end{itemize}"] * (depth + 1))
    return out


def _table(header: list[str], rows: list[list[str]], caption: str, n: int, springer: bool = False) -> list[str]:
    wide = not springer and (len(header) >= 5 or max(len(" ".join(r)) for r in [header, *rows]) > 70)
    env = "table*" if wide else ("sidewaystable" if springer and len(header) > 12 else "table")
    body = [" & ".join(rf"\textbf{{{tex_inline(c)}}}" for c in header) + r" \\", r"\midrule"]
    body += [" & ".join(tex_inline(c) for c in r) + r" \\" for r in rows]
    return [rf"\begin{{{env}}}[{'!tp' if wide else '!htbp'}]", r"\centering\footnotesize", rf"\caption{{{caption}}}", rf"\label{{tab:{n}}}",
            r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{'l' * len(header)}}}", r"\toprule",
            *body, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", rf"\end{{{env}}}", ""]


def _raw(latex: str) -> tuple[str, str, str]:
    return ("raw", latex, "")


def markdown_to_latex(md: str, base: Path, springer: bool = False) -> tuple[str, list[tuple[str, str]]]:
    """The generated draft -> (LaTeX source, [(figure path relative to ``base``, its name in the LaTeX)]).
    IEEEtran conference layout by default; ``springer`` gives a single-column article with numbered references
    and figures named Fig1.png ... next to the .tex (the IJASS guidelines ask for no subfolders)."""
    blocks = _blocks(md)
    title, note = "", ""
    body: list[tuple[str, str, str]] = []  # (kind, LaTeX, Markdown source of a paragraph)
    abstract: list[str] = []
    figures: list[tuple[str, str]] = []
    keywords = KEYWORDS
    mode = "body"  # body | abstract | skip
    last_heading, n_tab, n_fig = "", 0, 0
    i = 0
    while i < len(blocks):
        b = blocks[i]
        kind = b[0]
        if kind == "h":
            level, text = b[1], b[2]
            if level == 1:
                title = text
            elif level == 2:
                name = _NUMBER.sub("", text)
                mode = "body"
                if text == "Abstract":
                    mode = "abstract"
                elif text in ("References",) or text.startswith("Appendix"):
                    mode = "skip"
                elif text != "Figures":
                    body.append(_raw((r"\section{" if _NUMBER.match(text) else r"\section*{") + tex_inline(name) + "}"))
                last_heading = name
            elif mode == "body":
                name = _NUMBER.sub("", text)
                body.append(_raw(("\\subsection{" if level == 3 else "\\subsubsection{") + tex_inline(name) + "}"))
                last_heading = name
        elif mode == "skip":
            pass
        elif kind == "p" and title and not abstract and mode == "body" and not body and b[1].startswith(AUTHORS[0]):
            pass  # the author line: printed by \author below
        elif kind == "p" and title and not abstract and mode == "body" and not body and not note:
            note = b[1]  # the generator's note under the title -> a comment
        elif kind == "p" and mode == "abstract":
            if b[1].startswith("**Keywords:**"):
                keywords = tex_inline(b[1].split("**Keywords:**", 1)[1].strip())
            else:
                abstract.append(tex_inline(b[1]))
        elif kind == "p" and b[1].startswith("$$") and b[1].endswith("$$"):  # display equation, \tag{n} inside
            body.append(_raw("\\begin{equation}\n" + b[1][2:-2].strip() + "\n\\end{equation}"))
        elif kind == "p":
            m = _FIG_CAPTION.match(b[1])
            if m and i + 1 < len(blocks) and blocks[i + 1][0] == "img":
                n_fig += 1
                srcs = []
                i += 1
                while i < len(blocks) and blocks[i][0] == "img":  # one image per panel
                    srcs.append(blocks[i][2])
                    i += 1
                names = [f"Fig{n_fig}{'abcd'[q] if len(srcs) > 1 else ''}{Path(s).suffix}" if springer else s
                         for q, s in enumerate(srcs)]
                figures.extend(zip(srcs, names))
                env = "figure*" if not springer and (len(srcs) > 1 or _png_ratio(base / srcs[0]) > 1.9) else "figure"
                width = 0.98 / len(srcs) if len(srcs) > 1 else (0.85 if springer else 1.0)
                body.append(_raw("\n".join([
                    rf"\begin{{{env}}}[{'!htbp' if springer else '!t'}]", r"\centering",
                    r"\hfill".join(rf"\includegraphics[width={width:.2f}\linewidth]{{{name}}}" for name in names),
                    rf"\caption{{{tex_inline(m.group(1))}}}", rf"\label{{fig:{n_fig}}}", rf"\end{{{env}}}", ""])))
                continue
            body.append(("p", tex_inline(b[1]), b[1]))
        elif kind == "ul":
            body.append(_raw("\n".join(_itemize(b[1]))))
        elif kind == "table":
            n_tab += 1
            prev = body[-1] if body else None
            numbered = _TAB_CAPTION.match(prev[2]) if prev and prev[0] == "p" else None
            if numbered:  # the generator's own "**Table n.** caption" line
                body.pop()
                caption = tex_inline(numbered.group(1))
            elif prev and prev[0] == "p" and len(prev[2]) <= 260:
                body.pop()
                caption = prev[1].rstrip(":").rstrip()
                if len(prev[2]) < 30:  # a bare preset name: say which study it belongs to
                    caption = f"{tex_inline(last_heading)}: {caption}"
            elif prev and prev[0] == "p" and prev[2].startswith("**"):
                caption = tex_inline(prev[2].split("**")[1].rstrip("."))
            else:
                caption = tex_inline(last_heading)
            body.append(_raw("\n".join(_table(b[1], b[2], caption, n_tab, springer))))
        elif kind == "img":  # an image without the generator's caption line
            figures.append((b[2], b[2]))
            body.append(_raw(rf"\begin{{figure}}[!t]\centering\includegraphics[width=\columnwidth]{{{b[2]}}}\end{{figure}}"))
        i += 1

    text_body = "\n\n".join(x[1] for x in body)
    bibliography = [r"\bibliographystyle{IEEEtran}", r"\bibliography{references}"]
    if springer:  # IJASS: Springer basic style, written out here so no .bst file is needed
        bib = _bib(RESEARCH / "references.bib")
        keys = list(dict.fromkeys(k for m in re.findall(r"⟦cite:([^⟧]+)⟧", md) for k in m.split(",")))
        bibliography = [r"\begin{thebibliography}{99}"]
        for k in keys:
            ref = springer_ref(bib[k])
            url = re.search(r" (https://\S+)$", ref)
            bibliography.append(rf"\bibitem{{{k}}} " + (tex_inline(ref[:url.start()]) + rf" \url{{{url.group(1)}}}"
                                                       if url else tex_inline(ref)))
        bibliography.append(r"\end{thebibliography}")
    doc_text = title + " ".join(abstract) + text_body + "".join(bibliography)
    unknown = sorted({c for c in doc_text if ord(c) > 127 and c not in UNICODE and c not in NATIVE})
    if unknown:
        raise ValueError(f"No LaTeX mapping for characters {unknown}; add them to paper_latex.UNICODE")
    declares = [rf"\newunicodechar{{{c}}}{{{v}}}" for c, v in UNICODE.items() if c in doc_text]
    comment = "\n".join("% " + line for line in re.findall(r".{1,100}(?:\s|$)", note.strip("*"))) if note else ""
    if springer:
        marks = [str(q + 1) + ("*" if a == CORRESPONDING[0] else "") for q, a in enumerate(AUTHORS)]
        cls = [r"\documentclass[11pt,a4paper]{article}", r"\usepackage[margin=25mm]{geometry}",
               r"\usepackage[numbers,sort&compress]{natbib}", r"\usepackage{rotating}"]
        front = [rf"\title{{{tex_inline(title)}}}",
                 r"\author{" + ", ".join(rf"{tex_inline(a)}$^{{{mk}}}$" for a, mk in zip(AUTHORS, marks)) + r"\\[1.5ex]"
                 + "".join(rf"{{\small $^{{{q + 1}}}${tex_inline(aff)}}}\\" for q, aff in enumerate(AFFILIATIONS))
                 + rf"[1ex]{{\small $^{{*}}$Corresponding author. E-mail: \texttt{{{_esc(CORRESPONDING[1])}}}}}}}",
                 r"\date{}", r"\maketitle", r"\begin{abstract}", "\n\n".join(abstract), r"\end{abstract}",
                 r"\noindent\textbf{Keywords:} " + keywords, ""]
    else:
        cls = [r"\documentclass[conference]{IEEEtran}", r"\usepackage{cite}",
               r"% Arabic section numbers, so the text's ``Section 8.5'' references stay right.",
               r"\renewcommand\thesection{\arabic{section}}", r"\providecommand\thesectiondis{}",
               r"\renewcommand\thesectiondis{\arabic{section}.}",
               r"\renewcommand\thesubsection{\thesection.\arabic{subsection}}", r"\providecommand\thesubsectiondis{}",
               r"\renewcommand\thesubsectiondis{\thesection.\arabic{subsection}}"]
        front = [rf"\title{{{tex_inline(title)}}}",
                 rf"\author{{\IEEEauthorblockN{{{tex_inline(', '.join(AUTHORS))}}}"
                 rf"\IEEEauthorblockA{{{tex_inline(AFFILIATION)}}}}}",
                 r"\maketitle", r"\begin{abstract}", "\n\n".join(abstract), r"\end{abstract}",
                 r"\begin{IEEEkeywords}", keywords, r"\end{IEEEkeywords}", ""]
    tex = "\n".join([
        "% Generated by `python -m flutter_ai paper --ieee | --springer` from the generator of paper-draft.md.",
        "% Do not edit numbers here; edit prose in src/flutter_ai/paper.py and re-export.", comment,
        *cls,
        r"\usepackage{iftex}",
        r"\ifPDFTeX  % pdfLaTeX (Overleaf's default)",
        r"  \usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{textcomp}",
        r"\else  % XeLaTeX / LuaLaTeX (e.g. Tectonic): Times-like TeX Gyre Termes",
        r"  \usepackage{fontspec}",
        r"  \setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,"
        r" ItalicFont=*-italic, BoldItalicFont=*-bolditalic]",
        r"\fi",
        r"\usepackage{newunicodechar}",
        r"\usepackage{amsmath,amssymb}", r"\usepackage{graphicx}", r"\usepackage{booktabs}",
        r"\usepackage{adjustbox}", r"\usepackage{url}",
        *declares,
        r"\extrafloats{100}",
        r"\begin{document}",
        *front,
        text_body, "",
        *bibliography,
        r"\end{document}", ""])
    return tex, figures


def _export_bib(src: Path, dst: Path) -> None:
    """Copy references.bib without internal fields (read status, notes to the owner)."""
    keep = [line for line in src.read_text(encoding="utf-8").splitlines()
            if not re.match(r"\s*readstatus\s*=", line)
            and not (re.match(r"\s*note\s*=", line) and re.search(r"owner|full text read|tests/", line, re.I))]
    dst.write_text("\n".join(keep) + "\n", encoding="utf-8")


def find_tectonic() -> str | None:
    """The Tectonic LaTeX compiler (``FLUTTER_AI_TECTONIC``, PATH, or %LOCALAPPDATA%/tectonic), if installed."""
    for cand in (os.environ.get("FLUTTER_AI_TECTONIC"), shutil.which("tectonic"),
                 str(Path(os.environ.get("LOCALAPPDATA", "~")).expanduser() / "tectonic" / "tectonic.exe")):
        if cand and Path(cand).exists():
            return cand
    return None


def compile_latex(tex_path: Path, pdf_path: Path | None = None, timeout_s: float = 600.0) -> Path | None:
    """Compile with Tectonic (XeLaTeX + BibTeX) to ``pdf_path``; None if Tectonic is not installed."""
    exe = find_tectonic()
    if exe is None:
        return None
    subprocess.run([exe, "-X", "compile", tex_path.name], cwd=tex_path.parent, check=True, timeout=timeout_s,
                   capture_output=True)
    pdf_path = Path(pdf_path or RESEARCH / f"paper-{tex_path.parent.name}.pdf")
    shutil.copyfile(tex_path.with_suffix(".pdf"), pdf_path)
    return pdf_path


def export_latex(springer: bool = False, out_dir: Path | None = None, zip_path: Path | None = None) -> Path:
    """Write paper.tex + references.bib + figures to ``out_dir`` and an Overleaf zip; return the .tex path."""
    style = "springer" if springer else "ieee"
    out_dir = Path(out_dir or RESEARCH / style)
    zip_path = Path(zip_path or RESEARCH / f"paper-{style}-overleaf.zip")
    with tempfile.TemporaryDirectory() as tmp:
        md = build(Path(tmp) / "paper.md", copy_figures=True, cite_markers=True).read_text(encoding="utf-8")
    tex, figures = markdown_to_latex(md, RESEARCH, springer)
    if (out_dir / "figures").exists():  # drop stale figures; other files are overwritten in place
        shutil.rmtree(out_dir / "figures")
    for old in out_dir.glob("Fig*.png"):
        old.unlink()
    for rel, name in figures:
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RESEARCH / rel, out_dir / name)
    if springer:  # the reference list is inside paper.tex
        (out_dir / "references.bib").unlink(missing_ok=True)
    else:
        _export_bib(RESEARCH / "references.bib", out_dir / "references.bib")
    tex_path = out_dir / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out_dir).as_posix())
    return tex_path
