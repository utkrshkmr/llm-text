#!/usr/bin/env python3
"""Build the online (GitHub Pages) edition of the book from the LaTeX sources.

Pipeline
  1. figures : every top-level tikzpicture of every source file is typeset on its own
               cropped page with the `preview` package (one LaTeX run per file), then each
               page is converted to SVG with pdftocairo.
  2. labels  : the .aux files of a normal PDF build give the number and type of every
               \\label, so cross-references read exactly as in the PDF ("Figure 4.3").
  3. pandoc  : each source file is pre-processed (boxes, exercises, labs, numbering,
               references, figures) and converted to HTML with pandoc + a Lua filter.
  4. site    : pages are wrapped in a template with a sidebar, per-page contents,
               light/dark mode, search and a PDF download link.

Usage (from the repository root). The online book is the only published output; a draft-mode
LaTeX pass is still needed once so that the .aux files (label numbers) exist:
    (cd book && for i in 1 2; do pdflatex -draftmode -interaction=nonstopmode main.tex; done)
    python3 web/build_site.py --book book --aux book --out site
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- helpers

def read(p): return Path(p).read_text(encoding="utf-8")
def write(p, s):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(s, encoding="utf-8")


def match_brace(s: str, i: int, open_ch="{", close_ch="}") -> int:
    """s[i] == open_ch; return index of the matching close_ch (handles escapes)."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError("unbalanced group at %d" % i)


def strip_comments(s: str) -> str:
    """Remove LaTeX comments (outside verbatim-like environments)."""
    out, verb = [], False
    for line in s.split("\n"):
        if re.match(r"\s*\\begin\{(lstlisting|verbatim)\}", line):
            verb = True
        if verb:
            out.append(line)
            if re.search(r"\\end\{(lstlisting|verbatim)\}", line):
                verb = False
            continue
        m = re.search(r"(?<!\\)%", line)
        if m:
            line = line[:m.start()]
            if not line.strip():
                continue
        out.append(line)
    return "\n".join(out)


# ----------------------------------------------------------------------------- structure

def parse_main(book: Path):
    """Return the ordered list of (kind, part, relpath) from main.tex."""
    src = read(book / "main.tex")
    items, part, appendix = [], None, False
    for m in re.finditer(r"\\(part|include|input|appendix)\b(?:\{([^}]*)\})?", src):
        cmd, arg = m.group(1), m.group(2)
        if cmd == "part":
            part = arg
        elif cmd == "appendix":
            appendix, part = True, "Appendices"
        elif cmd in ("include", "input") and arg and not arg.startswith("preamble"):
            kind = "front" if arg.startswith("frontmatter") else ("app" if appendix else "chapter")
            items.append(dict(kind=kind, part=part if kind != "front" else None, src=arg))
    return items


def load_labels(aux_dir: Path):
    """label -> dict(number, type, stem). Reads every .aux under aux_dir."""
    labels = {}
    for aux in aux_dir.rglob("*.aux"):
        stem = aux.stem
        for line in read(aux).splitlines():
            m = re.match(r"\\newlabel\{(.+?)@cref\}\{\{\[(\w+)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}", line)
            if m:
                lab, typ, num = m.groups()
                labels.setdefault(lab, {}).update(type=typ, number=num, stem=stem)
                continue
            m = re.match(r"\\newlabel\{([^@}]+)\}\{\{([^}]*)\}\{([^}]*)\}\{(.*?)\}\{", line)
            if m:
                lab, num, _page, title = m.groups()
                d = labels.setdefault(lab, {})
                d.setdefault("number", num)
                d["title"] = title
                d.setdefault("stem", stem)
    return labels


REF_NAMES = {
    "chapter": ("Chapter", "Chapters"), "appendix": ("Appendix", "Appendices"),
    "section": ("Section", "Sections"), "subsection": ("Section", "Sections"),
    "subappendix": ("Appendix", "Appendices"), "figure": ("Figure", "Figures"),
    "table": ("Table", "Tables"), "listing": ("Listing", "Listings"),
    "equation": ("Equation", "Equations"), "exercise": ("Exercise", "Exercises"),
    "proposition": ("Proposition", "Propositions"), "definition": ("Definition", "Definitions"),
    "example": ("Example", "Examples"),
}


# ----------------------------------------------------------------------------- figures

PREVIEW_WRAPPER = r"""\documentclass[11pt,twoside,openany]{book}
\usepackage{%(book)s/preamble/llmsys}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{tikzpicture}
\renewenvironment{systemslens}[1][]{}{}
\renewenvironment{keyidea}[1][]{}{}
\renewenvironment{pitfall}[1][]{}{}
\renewenvironment{napkinmath}[1][]{}{}
\renewenvironment{deepdive}[1][]{}{}
\renewenvironment{labbox}[2]{}{}
\renewenvironment{tcolorbox}[1][]{}{}
\RenewDocumentCommand\resizebox{s m m m}{#4}
\RenewDocumentCommand\scalebox{m o m}{#3}
\setlength\PreviewBorder{6pt}
\graphicspath{{%(book)s/figures/}}
\begin{document}
\setcounter{chapter}{%(chnum)d}
\input{%(book)s/%(src)s}
\end{document}
"""


def count_tikz(src_text: str) -> int:
    """Number of top-level tikzpicture environments (nesting-aware)."""
    depth = n = 0
    for m in re.finditer(r"\\(begin|end)\{tikzpicture\}", strip_comments(src_text)):
        if m.group(1) == "begin":
            if depth == 0:
                n += 1
            depth += 1
        else:
            depth -= 1
    return n


def build_figures(book: Path, items, figdir: Path, workdir: Path, jobs: int, only=None):
    figdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for idx, it in enumerate(items):
        srcfile = book / (it["src"] + ".tex")
        if only and Path(it["src"]).name not in only:
            continue
        n = count_tikz(read(srcfile))
        if n == 0:
            continue
        stem = Path(it["src"]).name
        tasks.append((stem, it["src"], n, it.get("chnum", 1)))

    def run(task):
        stem, src, n, chnum = task
        d = workdir / stem
        d.mkdir(parents=True, exist_ok=True)
        write(d / "fig.tex", PREVIEW_WRAPPER % dict(book=str(book.resolve()), src=src,
                                                     chnum=max(chnum - 1, 0)))
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                               cwd=d, capture_output=True, text=True)
        pdf = d / "fig.pdf"
        if not pdf.exists():
            m = re.search(r"^!.*", r.stdout, flags=re.M)
            return stem, n, 0, "LaTeX failed: " + (m.group(0) if m else r.stdout[-300:])
        pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
        for k in range(1, pages + 1):
            out = figdir / f"{stem}-{k}.svg"
            subprocess.run(["pdftocairo", "-svg", "-f", str(k), "-l", str(k), str(pdf), str(out)],
                           check=True)
        return stem, n, pages, ""

    problems = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for stem, n, pages, err in ex.map(run, tasks):
            status = "ok" if n == pages else "MISMATCH"
            print(f"  figures {stem:32s} expected {n:3d} rendered {pages:3d} {status} {err[:200]}")
            if n != pages:
                problems.append(stem)
    return problems


# ----------------------------------------------------------------------------- LaTeX -> pandoc-friendly LaTeX

BOX_TITLES = {"systemslens": "Systems Lens", "keyidea": "Key Idea", "pitfall": "Pitfall",
              "napkinmath": "Napkin Math", "deepdive": "Deeper Dive"}


def clean_colspec(spec: str) -> str:
    """Simplify a tabular column spec to something pandoc understands."""
    out, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c in "><@!" and i + 1 < len(spec) and spec[i + 1] == "{":
            i = match_brace(spec, i + 1) + 1
            continue
        if c in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            j = match_brace(spec, i + 1)
            out.append("l"); i = j + 1
            continue
        if c == "*" and i + 1 < len(spec) and spec[i + 1] == "{":
            j = match_brace(spec, i + 1); k = match_brace(spec, j + 1)
            try:
                n = int(spec[i + 2:j])
                out.append(clean_colspec(spec[j + 2:k]) * n)
            except ValueError:
                pass
            i = k + 1
            continue
        if c in "lcrX":
            out.append("l" if c == "X" else c)
        elif c == "S":
            out.append("r")
        i += 1
    return "".join(out) or "l"


def longtable_to_table(s: str) -> str:
    """Rewrite longtable as table+tabular: pandoc handles the latter far better."""
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{longtable\}", s):
        if m.start() < pos:
            continue
        end = s.index("\\end{longtable}", m.end())
        j = m.end()
        while s[j] in " \n": j += 1
        if s[j] == "[":
            j = match_brace(s, j, "[", "]") + 1
            while s[j] in " \n": j += 1
        k = match_brace(s, j)
        spec = s[j + 1:k]
        body = s[k + 1:end]
        cap, lab = "", ""
        cm = re.search(r"\\caption\{", body)
        if cm:
            ce = match_brace(body, cm.end() - 1)
            cap = body[cm.start():ce + 1]
            rest = body[ce + 1:]
            lm = re.match(r"\s*(\\label\{[^}]*\})", rest)
            if lm:
                lab = lm.group(1); rest = rest[lm.end():]
            rest = re.sub(r"^\s*\\\\", "", rest)
            body = body[:cm.start()] + rest
        if "\\endfirsthead" in body:
            first, _, after = body.partition("\\endfirsthead")
            if "\\endhead" in after:
                after = after.partition("\\endhead")[2]
            body = first + after
        else:
            body = body.replace("\\endhead", "")
        for tok in ("\\endfoot", "\\endlastfoot"):
            if tok in body:
                parts = body.split(tok)
                body = parts[-1] if parts[-1].strip() else parts[0]
        new = ("\\begin{table}\n\\begin{tabular}{%s}%s\\end{tabular}\n%s%s\n\\end{table}"
               % (spec, body, cap, lab))
        out.append(s[pos:m.start()]); out.append(new); pos = end + len("\\end{longtable}")
    out.append(s[pos:])
    return "".join(out)


def clean_tables(s: str) -> str:
    s = longtable_to_table(s)
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{(tabularx|tabular\*?|longtable|tabulary)\}", s):
        if m.start() < pos:
            continue
        env = m.group(1)
        j = m.end()
        if env in ("tabularx", "tabular*", "tabulary"):      # width argument first
            while s[j] in " \n": j += 1
            j = match_brace(s, j) + 1
        while j < len(s) and s[j] in " \n": j += 1
        if j < len(s) and s[j] == "[":
            j = match_brace(s, j, "[", "]") + 1
        while s[j] in " \n": j += 1
        k = match_brace(s, j)
        spec = clean_colspec(s[j + 1:k])
        new_env = "longtable" if env == "longtable" else "tabular"
        out.append(s[pos:m.start()]); out.append("\\begin{%s}{%s}" % (new_env, spec)); pos = k + 1
    out.append(s[pos:])
    s = "".join(out)
    s = re.sub(r"\\end\{(tabularx|tabular\*|tabulary)\}", r"\\end{tabular}", s)
    out, pos = [], 0
    for m in re.finditer(r"\\multicolumn\{([^}]*)\}\{", s):
        if m.start() < pos:
            continue
        k = match_brace(s, m.end() - 1)
        out.append(s[pos:m.start()])
        out.append("\\multicolumn{%s}{%s}" % (m.group(1), clean_colspec(s[m.end():k])))
        pos = k + 1
    out.append(s[pos:])
    return "".join(out)


class Preprocessor:
    def __init__(self, labels, page_of_stem, stem, chnum, chlabel, is_appendix):
        self.labels, self.page_of_stem, self.stem = labels, page_of_stem, stem
        self.chnum, self.chlabel, self.is_appendix = chnum, chlabel, is_appendix

    # -- references -------------------------------------------------------------------
    def href(self, lab, text):
        info = self.labels.get(lab)
        if not info:
            return text
        page = self.page_of_stem.get(info.get("stem"), "")
        target = ("" if page == self.page_of_stem.get(self.stem) else page) + "\\#" + lab
        return "\\href{%s}{%s}" % (target, text)

    def ref_text(self, labs, capital=True, kind="cref"):
        labs = [l.strip() for l in labs.split(",") if l.strip()]
        if kind in ("ref", "eqref"):
            out = []
            for l in labs:
                num = self.labels.get(l, {}).get("number", "??")
                out.append(self.href(l, "(%s)" % num if kind == "eqref" else num))
            return ", ".join(out)
        groups = []
        for l in labs:
            info = self.labels.get(l, {})
            typ = info.get("type", "section")
            if groups and groups[-1][0] == typ:
                groups[-1][1].append(l)
            else:
                groups.append((typ, [l]))
        parts = []
        for typ, ls in groups:
            sing, plur = REF_NAMES.get(typ, ("Item", "Items"))
            nums = []
            for l in ls:
                num = self.labels.get(l, {}).get("number", "??")
                if typ == "equation":
                    num = "(%s)" % num
                nums.append(self.href(l, num))
            name = sing if len(nums) == 1 else plur
            if len(nums) == 1:
                joined = nums[0]
            else:
                joined = ", ".join(nums[:-1]) + " and " + nums[-1]
            parts.append(name + "~" + joined)
        txt = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        if not capital:
            txt = txt[0].lower() + txt[1:]
        return txt

    def replace_refs(self, s):
        def rep(m):
            cmd, labs = m.group(1), m.group(2)
            if cmd in ("cref", "Cref", "autoref", "Autoref"):
                return self.ref_text(labs, capital=True)  # book uses `capitalise`
            if cmd == "eqref":
                return self.ref_text(labs, kind="eqref")
            if cmd == "pageref":
                return ""
            return self.ref_text(labs, kind="ref")
        return re.sub(r"\\(cref|Cref|autoref|Autoref|ref|eqref|pageref)\*?\{([^}]*)\}", rep, s)

    # -- figures ----------------------------------------------------------------------
    def replace_tikz(self, s):
        out, i, k, depth, start = [], 0, 0, 0, None
        for m in re.finditer(r"\\(begin|end)\{tikzpicture\}", s):
            if m.group(1) == "begin":
                if depth == 0:
                    start = m.start()
                depth += 1
            else:
                depth -= 1
                if depth == 0 and start is not None:
                    k += 1
                    out.append(s[i:start])
                    out.append("\\includegraphics{figs/%s-%d.svg}" % (self.stem, k))
                    i = m.end()
                    start = None
        out.append(s[i:])
        s = "".join(out)
        # unwrap \resizebox{..}{..}{ \includegraphics{..} } and \scalebox
        s = re.sub(r"\\resizebox\*?\{[^{}]*\}\{[^{}]*\}\{%?\s*(\\includegraphics\{[^}]*\})\s*%?\s*\}",
                   r"\1", s)
        s = re.sub(r"\\scalebox\{[^{}]*\}(?:\[[^\]]*\])?\{%?\s*(\\includegraphics\{[^}]*\})\s*%?\s*\}",
                   r"\1", s)
        return s

    # -- environments with optional titles ---------------------------------------------
    def replace_boxes(self, s):
        for env, title in BOX_TITLES.items():
            pat = re.compile(r"\\begin\{%s\}" % env)
            pos = 0
            while True:
                m = pat.search(s, pos)
                if not m:
                    break
                j = m.end()
                opt = ""
                rest = s[j:]
                lead = len(rest) - len(rest.lstrip(" \t"))
                if rest[lead:lead + 1] == "[":
                    k = match_brace(s, j + lead, "[", "]")
                    opt = s[j + lead + 1:k]
                    j = k + 1
                head = title + (": " + opt if opt.strip() else "")
                rep = "\\begin{%s}\n\\textbf{%s}\n\n" % (env, head)
                s = s[:m.start()] + rep + s[j:]
                pos = m.start() + len(rep)
        return s

    def replace_labbox(self, s):
        def rep(m):
            j = m.end()
            # two mandatory args
            a0 = j
            a1 = match_brace(s, a0)
            b0 = a1 + 1
            while s[b0] in " \n\t":
                b0 += 1
            b1 = match_brace(s, b0)
            return (m.start(), b1 + 1, "\\begin{labbox}\n\\textbf{Lab %s: %s}\n\n" %
                    (s[a0 + 1:a1], s[b0 + 1:b1]))
        out, pos = [], 0
        for m in re.finditer(r"\\begin\{labbox\}", s):
            if m.start() < pos:
                continue
            st, en, txt = rep(m)
            out.append(s[pos:st]); out.append(txt); pos = en
        out.append(s[pos:])
        return "".join(out)

    def replace_exercises(self, s):
        n = [0]
        prefix = self.chnum

        def rep(m):
            n[0] += 1
            name = m.group(2)
            head = "\\textbf{Exercise %s.%d}" % (prefix, n[0])
            if name:
                head += " (\\emph{%s})" % name
            return "\\begin{exercise}\n%s. " % head
        return re.sub(r"\\begin\{exercise\}(\[([^\]]*)\])?", rep, s)

    # -- numbering of headings and captions ---------------------------------------------
    def number_headings(self, s):
        sec = sub = 0
        fig = tab = lst = 0
        p = self.chnum

        def num_heading(m):
            nonlocal sec, sub
            cmd, star = m.group(1), m.group(2)
            if star:
                return m.group(0)
            if cmd == "section":
                sec += 1; sub = 0
                return "\\section{%s.%d~~" % (p, sec)
            sub += 1
            return "\\subsection{%s.%d.%d~~" % (p, sec, sub)
        s = re.sub(r"\\(section|subsection)(\*?)\{", num_heading, s)
        if self.chlabel is not None:
            head = "\\chapter{%s~~" % (("Appendix " if self.is_appendix else "Chapter ") + str(p))
            s = re.sub(r"\\chapter\{", lambda _m: head, s, count=1)

        # captions: count floats in order
        out, pos = [], 0
        for m in re.finditer(r"\\begin\{(figure|table|longtable)\*?\}|\\caption\{|"
                             r"\\begin\{lstlisting\}\[", s):
            tok = m.group(0)
            if tok.startswith("\\begin{figure"):
                cur = "figure"; continue_ = True
                self._cur = "figure"
                continue
            if tok.startswith("\\begin{table") or tok.startswith("\\begin{longtable"):
                self._cur = "table"
                continue
            if tok.startswith("\\begin{lstlisting}"):
                k = match_brace(s, m.end() - 1, "[", "]")
                opts = s[m.end():k]
                if re.search(r"caption\s*=", opts):
                    lst += 1
                    new_opts = re.sub(r"caption\s*=\s*\{", "caption={Listing %s.%d: " % (p, lst),
                                      opts, count=1)
                    if new_opts == opts:
                        new_opts = re.sub(r"caption\s*=\s*", "caption=Listing %s.%d: " % (p, lst),
                                          opts, count=1)
                    out.append(s[pos:m.end()]); out.append(new_opts); pos = k
                continue
            if tok == "\\caption{":
                kind = getattr(self, "_cur", "figure")
                if kind == "figure":
                    fig += 1; label = "Figure %s.%d: " % (p, fig)
                else:
                    tab += 1; label = "Table %s.%d: " % (p, tab)
                out.append(s[pos:m.end()]); out.append(label); pos = m.end()
        out.append(s[pos:])
        return "".join(out)

    # -- equations ----------------------------------------------------------------------
    def fix_equations(self, s):
        def rep(m):
            env, body = m.group(1), m.group(2)
            labs = re.findall(r"\\label\{([^}]*)\}", body)
            body2 = re.sub(r"\\label\{[^}]*\}", "", body)
            anchors = "".join("\\hypertarget{%s}{}" % l for l in labs)
            if labs and not env.endswith("*") and env in ("equation", "align", "gather", "multline"):
                num = self.labels.get(labs[0], {}).get("number")
                if num and env == "equation":
                    body2 = body2 + "\\tag{%s}" % num
            return anchors + "\\begin{%s}%s\\end{%s}" % (env, body2, env)
        return re.sub(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}(.*?)\\end\{\1\}",
                      rep, s, flags=re.S)

    # -- misc macros ----------------------------------------------------------------------
    def simple_macros(self, s):
        s = re.sub(r"\\covers\{[^{}]*\}", "", s)
        s = re.sub(r"\\labdir\{([^}]*)\}",
                   lambda m: "\\textbf{Code:} \\texttt{code/labs/%s}\n\n" % m.group(1).replace("_", "\\_"), s)
        s = re.sub(r"\\labsection\{([^}]*)\}", r"\n\n\\textbf{\1}\n\n", s)
        s = re.sub(r"\\ident\{([^}]*)\}", lambda m: "\\texttt{%s}" % m.group(1).replace("_", "\\_"), s)
        s = s.replace("\\code{", "\\texttt{")
        # pandoc drops labels inside theorem-like/unknown environments: keep explicit anchors
        s = re.sub(r"\\label\{((?:ex|prop|def|defn|example|thm|lem|exer):[^}]*)\}",
                   lambda m: "\\hypertarget{%s}{}" % m.group(1), s)
        s = re.sub(r"\\marginpar\{[^{}]*\}", "", s)
        s = s.replace("\\noindent", "")
        # listings: language for highlighting
        def lst(m):
            opts = m.group(1) or ""
            if "style=shell" in opts:
                lang = "bash"
            elif "style=plain" in opts:
                lang = None
            elif "style=cpp" in opts or "style=cuda" in opts:
                lang = "cpp"
            elif "style=cmake" in opts:
                lang = "cmake"
            else:
                lang = "python"
            opts = re.sub(r"\s*style\s*=\s*\w+\s*,?", "", opts).strip(", \n\t")
            if lang and "language" not in opts:
                opts = ("language=%s," % lang) + opts
            opts = opts.strip(", \n\t")
            return ("\\begin{lstlisting}[%s]" % opts) if opts else "\\begin{lstlisting}"
        s = re.sub(r"\\begin\{lstlisting\}(?:\[(.*?)\])?(?=\s*\n)", lst, s, flags=re.S)
        return s

    def run(self, s):
        s = strip_comments(s)
        s = self.replace_tikz(s)
        s = self.simple_macros(s)
        s = clean_tables(s)
        s = self.replace_boxes(s)
        s = self.replace_labbox(s)
        s = self.replace_exercises(s)
        s = self.number_headings(s)
        s = self.fix_equations(s)
        s = self.replace_refs(s)
        return s


def math_macros(book: Path) -> str:
    """Translate the preamble's math macros into pandoc-expandable \\newcommand's."""
    sty = read(book / "preamble" / "llmsys.sty")
    out = [r"\newcommand{\bm}[1]{\boldsymbol{#1}}"]
    for m in re.finditer(r"\\DeclareMathOperator(\*?)\{(\\\w+)\}\{([^}]*)\}", sty):
        star, name, body = m.groups()
        out.append(r"\newcommand{%s}{\operatorname%s{%s}}" % (name, star, body))
    for line in sty.splitlines():
        line = line.strip()
        m = re.match(r"\\newcommand\{(\\\w+)\}(\[\d\])?\{(.*)\}$", line)
        if not m:
            continue
        name = m.group(1)
        if name in (r"\code", r"\ident", r"\labdir", r"\labsection", r"\covers"):
            continue
        out.append(line)
    out.append(r"\newcommand{\SI}[2]{#1\,\mathrm{#2}}")
    out.append(r"\newcommand{\si}[1]{\mathrm{#1}}")
    out.append(r"\newcommand{\num}[1]{#1}")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- site

def size_figures(body: str, out: Path, px_per_pt: float = 2.1) -> str:
    """Give every TikZ figure an explicit display width: about 2x its typeset size so figure
    text reads at body-text size, capped by the column width (CSS max-width: 100%)."""
    def rep(m):
        tag, src = m.group(0), m.group(1)
        svg = out / src
        try:
            head = svg.open(encoding="utf-8", errors="ignore").read(600)
            w = float(re.search(r'<svg[^>]*?\swidth="([\d.]+)(?:pt)?"', head).group(1))
        except Exception:
            return tag
        width = int(round(w * px_per_pt))
        if "style=" in tag:
            return tag
        return tag.replace("<img ", '<img style="width: %dpx" ' % width, 1)
    return re.sub(r'<img [^>]*src="(figs/[^"]+\.svg)"[^>]*>', rep, body)


def slug(src):
    return Path(src).name + ".html"


def run_pandoc(tex: str, macros: str, lua: Path, toc: bool):
    args = ["pandoc", "-f", "latex", "-t", "html5", "--mathjax", "--wrap=none", "--metadata", "title=x",
            "--lua-filter", str(lua), "--section-divs", "--highlight-style=pygments"]
    if toc:
        args += ["--toc", "--toc-depth=3", "--template", str(HERE / "fragment.html")]
    r = subprocess.run(args, input=macros + tex, capture_output=True, text=True)
    return r.stdout, r.stderr


def build_search_index(pages):
    entries = []
    for p in pages:
        body = p["body"]
        for m in re.finditer(r'<section id="([^"]+)"[^>]*>\s*<h([23])[^>]*>(.*?)</h\2>(.*?)(?=<section|</section>)',
                             body, flags=re.S):
            sid, _, title, text = m.groups()
            text = re.sub(r"<[^>]+>", " ", text)
            text = html.unescape(re.sub(r"\s+", " ", text)).strip()
            title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
            entries.append(dict(u=p["file"] + "#" + sid, p=p["title"], t=title, x=text[:600]))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="book")
    ap.add_argument("--aux", default="book", help="directory containing the .aux files of a PDF build")
    ap.add_argument("--pdf", default=None, help="unused (the PDF is no longer published)")
    ap.add_argument("--out", default="site")
    ap.add_argument("--work", default="_webbuild")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated source stems to (re)build")
    args = ap.parse_args()

    book, out, work = Path(args.book), Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    items = parse_main(book)
    labels = load_labels(Path(args.aux))

    # chapter numbers / page names
    chnum, appnum = 0, 0
    page_of_stem = {}
    for it in items:
        stem = Path(it["src"]).name
        it["stem"], it["file"] = stem, slug(it["src"])
        page_of_stem[stem] = it["file"]
        if it["kind"] == "chapter":
            chnum += 1; it["chnum"] = chnum; it["num"] = str(chnum)
        elif it["kind"] == "app":
            it["num"] = chr(ord("A") + appnum); appnum += 1; it["chnum"] = 1
        else:
            it["num"] = None; it["chnum"] = 0
    # solutions files are \InputIfFileExists'd by the last appendix: map them to that page
    sol_page = next((it["file"] for it in items if "solutions" in it["src"]), None)
    for f in (book / "solutions").glob("*.tex"):
        page_of_stem[f.stem] = sol_page

    only = set(args.only.split(",")) if args.only else None
    if not args.skip_figures:
        print("[1/3] rendering figures")
        bad = build_figures(book, items, out / "figs", work / "figs", args.jobs, only)
        if bad:
            print("  WARNING: figure count mismatch in", bad)

    print("[2/3] converting chapters")
    macros = math_macros(book)
    lua = HERE / "filters.lua"
    pages = []
    for it in items:
        src = read(book / (it["src"] + ".tex"))
        if "solutions" in it["src"]:
            def inc(m):
                p = book / (m.group(1) + ".tex")
                return read(p) if p.exists() else ""
            src = re.sub(r"\\InputIfFileExists\{([^}]*)\}\{\}\{\}", inc, src)
        chlabel = None
        m = re.search(r"\\chapter\{.*?\}\\label\{([^}]*)\}", src)
        if m:
            chlabel = m.group(1)
        num = it["num"] if it["kind"] != "front" else None
        pre = Preprocessor(labels, page_of_stem, it["stem"], num if num else 0,
                           chlabel if num else None, it["kind"] == "app")
        tex = pre.run(src)
        title_m = re.search(r"\\chapter\*?\{(.*?)\}", src)
        title = re.sub(r"\\[a-zA-Z]+\s*", "", title_m.group(1)) if title_m else it["stem"]
        title = title.replace("{", "").replace("}", "").replace("~", " ")
        it["title"] = title
        if only and it["stem"] not in only and (out / it["file"]).exists():
            pages.append(dict(it, body=None))
            continue
        body, err = run_pandoc(tex, macros, lua, toc=True)
        if err.strip():
            print("  pandoc warnings for", it["stem"], ":", err.strip().splitlines()[:3])
        pages.append(dict(it, body=body))
        print("  converted", it["stem"])

    print("[3/3] assembling site")
    template = read(HERE / "template.html")
    chev = ('<svg class="chev" viewBox="0 0 24 24" width="12" height="12" aria-hidden="true">'
            '<path d="M9 6l6 6-6 6" stroke="currentColor" stroke-width="2.4" fill="none" '
            'stroke-linecap="round"/></svg>')

    def link(it):
        label = ("%s&nbsp; " % it["num"]) if it["num"] else ""
        return ('<li><a href="%s" data-page="%s"><span class="num">%s</span>%s</a></li>' %
                (it["file"], it["file"], label, html.escape(it["title"])))

    nav = ['<ul class="nav-front">']
    nav += [link(it) for it in items if not it["part"]]
    nav.append("</ul>")
    parts = []
    for it in items:
        if it["part"] and (not parts or parts[-1][0] != it["part"]):
            parts.append((it["part"], []))
        if it["part"]:
            parts[-1][1].append(it)
    for k, (name, its) in enumerate(parts):
        nav.append('<div class="nav-part" data-part="p%d"><button class="part-toggle" type="button">%s<span>%s</span></button><ul>'
                   % (k, chev, html.escape(name)))
        nav += [link(it) for it in its]
        nav.append("</ul></div>")
    nav_html = "\n".join(nav)
    pdf_name = None

    for i, p in enumerate(pages):
        if p["body"] is None:
            continue
        toc_m = re.search(r"<!--TOC-->(.*?)<!--/TOC-->", p["body"], flags=re.S)
        toc = toc_m.group(1) if toc_m else ""
        # drop the chapter title level from the page contents
        inner = re.sub(r"^\s*<nav[^>]*>|</nav>\s*$", "", toc.strip())
        tm = re.match(r'\s*<ul>\s*<li><a[^>]*>.*?</a>\s*(<ul>.*</ul>)\s*</li>\s*</ul>\s*$', inner, flags=re.S)
        toc = '<nav id="TOC" role="doc-toc">' + (tm.group(1) if tm else inner) + '</nav>'
        body = size_figures(p["body"], out)
        p["body"] = body

        body = re.sub(r"<!--TOC-->.*?<!--/TOC-->", "", p["body"], flags=re.S)
        prev_l = ('<a class="prev" href="%s">&larr; %s</a>' % (pages[i - 1]["file"], html.escape(pages[i - 1]["title"]))) if i > 0 else "<span></span>"
        next_l = ('<a class="next" href="%s">%s &rarr;</a>' % (pages[i + 1]["file"], html.escape(pages[i + 1]["title"]))) if i + 1 < len(pages) else "<span></span>"
        page = (template.replace("$TITLE$", html.escape(p["title"]))
                .replace("$NAV$", nav_html).replace("$TOC$", toc).replace("$BODY$", body)
                .replace("$PREV$", prev_l).replace("$NEXT$", next_l)
                .replace("$PDF$", pdf_name or ""))
        write(out / p["file"], page)
        p["body_final"] = body

    # landing page
    cards = []
    cur_part = None
    for it in items:
        if it["part"] != cur_part and it["part"]:
            if cur_part is not None:
                cards.append("</ul></section>")
            cur_part = it["part"]
            cards.append('<section class="part-card"><h3>%s</h3><ul>' % html.escape(cur_part))
        if it["kind"] == "front":
            continue
        cards.append('<li><a href="%s"><span class="num">%s</span> %s</a></li>' %
                     (it["file"], it["num"], html.escape(it["title"])))
    cards.append("</ul></section>")
    landing = read(HERE / "landing.html")
    first = next(it["file"] for it in items)
    landing = (landing.replace("$NAV$", nav_html).replace("$CARDS$", "\n".join(cards))
               .replace("$FIRST$", first).replace("$PDF$", pdf_name or ""))
    write(out / "index.html", landing)

    # assets, search index, .nojekyll
    for f in (HERE / "assets").iterdir():
        shutil.copy(f, out / f.name)
    if (out / "mathjax").exists():
        shutil.rmtree(out / "mathjax")
    shutil.copytree(HERE / "vendor" / "mathjax", out / "mathjax")
    idx = build_search_index([dict(file=p["file"], title=p["title"], body=p.get("body_final", "") or "")
                              for p in pages])
    write(out / "search.json", json.dumps(idx, ensure_ascii=False))
    write(out / ".nojekyll", "")
    print("done:", out.resolve(), "pages:", len(pages), "search entries:", len(idx))


if __name__ == "__main__":
    main()
