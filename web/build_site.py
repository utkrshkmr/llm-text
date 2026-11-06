#!/usr/bin/env python3
"""Build the website of *Big Book of LLMs* from its Markdown sources.

    python3 web/build_site.py --book book --out site
    python3 -m http.server -d site 8000

Sources (book/)
    book.yml           title, front matter, parts with their chapters, appendices (planned order;
                       files that do not exist yet are skipped with a note)
    **/*.md            pandoc Markdown: maths in $...$ / $$...$$, fenced divs for boxes
                       (::: systemslens / keyidea / pitfall / napkinmath / deepdive / exercise /
                       proposition / labbox) whose first paragraph may be [Title]{.box-title}
    solutions/*.md     appended, in book order, to the solutions appendix
    figures/*.svg      figures

Numbering is done here, per chapter, in document order: chapters and appendices from book.yml,
## sections and ### subsections (unless {.unnumbered}), figures and tables with captions,
listings with a caption attribute, display equations with \\label{eq:...}, exercises,
propositions; a lab box takes its chapter's number. A cross-reference is a citation:
[@fig:x] renders "Figure 4.3", [@sec:a; @sec:b] "Sections 4.1 and 4.2", [-@eq:y] the bare number.
Unknown labels render as "??" and are listed at the end of the build.

Requirements: Python 3.9+, pandoc >= 3. No TeX.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figtheme import check_svg, dark_svg  # noqa: E402

HERE = Path(__file__).resolve().parent

KIND_NAMES = {
    "chapter": ("Chapter", "Chapters"), "appendix": ("Appendix", "Appendices"),
    "section": ("Section", "Sections"), "subsection": ("Section", "Sections"),
    "figure": ("Figure", "Figures"), "table": ("Table", "Tables"), "listing": ("Listing", "Listings"),
    "equation": ("Equation", "Equations"), "exercise": ("Exercise", "Exercises"),
    "proposition": ("Proposition", "Propositions"), "definition": ("Definition", "Definitions"),
    "example": ("Example", "Examples"), "lab": ("Lab", "Labs"),
}
COUNTED = ("figure", "table", "listing", "equation", "exercise", "proposition", "definition", "example")


def read(p): return Path(p).read_text(encoding="utf-8")
def write(p, s):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(s, encoding="utf-8")


# ----------------------------------------------------------------------------- book.yml
def parse_book_yml(path: Path) -> dict:
    """Minimal parser for the fixed shape of book.yml (no YAML dependency)."""
    meta, front, parts, apps = {}, [], [], []
    section = None
    for raw in read(path).splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("- title:") else raw.rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            key, _, val = line.partition(":")
            section = key.strip()
            if val.strip():
                meta[section] = val.strip()
            continue
        s = line.strip()
        if section == "front" and s.startswith("- "):
            front.append(s[2:].strip())
        elif section == "parts":
            if s.startswith("- title:"):
                parts.append({"title": s.split(":", 1)[1].strip(), "chapters": []})
            elif s.startswith("- "):
                parts[-1]["chapters"].append(s[2:].strip())
        elif section == "appendices" and s.startswith("- "):
            apps.append(s[2:].strip())
    return dict(meta=meta, front=front, parts=parts, appendices=apps)


def book_items(book: Path, spec: dict):
    """Ordered pages: dict(kind, part, src, num, stem, file). Numbers follow the planned order."""
    items, missing = [], []
    for src in spec["front"]:
        items.append(dict(kind="front", part=None, src=src, num=None))
    n = 0
    for part in spec["parts"]:
        for src in part["chapters"]:
            n += 1
            items.append(dict(kind="chapter", part=part["title"], src=src, num=str(n)))
    for k, src in enumerate(spec["appendices"]):
        items.append(dict(kind="app", part="Appendices", src=src, num=chr(ord("A") + k)))
    present = []
    for it in items:
        if (book / it["src"]).exists():
            it["stem"] = Path(it["src"]).stem
            it["file"] = it["stem"] + ".html"
            present.append(it)
        else:
            missing.append(it["src"])
    return present, missing


# ----------------------------------------------------------------------------- pandoc
def run_pandoc(md: str, lua: Path):
    args = ["pandoc", "-f", "markdown", "-t", "html5", "--mathjax", "--wrap=none",
            "--metadata", "title=x", "--lua-filter", str(lua), "--section-divs",
            "--highlight-style=pygments", "--toc", "--toc-depth=3",
            "--template", str(HERE / "fragment.html")]
    r = subprocess.run(args, input=md, capture_output=True, text=True)
    return r.stdout, r.stderr


MARK = re.compile(r'<span class="xn" data-k="([a-z]+)" data-id="([^"]*)"></span>')
XREF = re.compile(r'<a class="xref" data-ids="([^"]*)" data-mode="(cref|ref)"></a>')


def number_page(body: str, it: dict, labels: dict):
    """Pass 1: assign numbers to this page's markers; record labels. Returns per-marker texts."""
    p = it["num"]
    numbered = p is not None
    sec = sub = 0
    counts = {k: 0 for k in COUNTED}
    texts = []
    for m in MARK.finditer(body):
        kind, lid = m.group(1), html.unescape(m.group(2))
        num, text = "", ""
        if kind == "chapter":
            if numbered:
                num = p
                text = ("Appendix %s" if it["kind"] == "app" else "Chapter %s") % p + "&nbsp;&nbsp;"
            kind = "appendix" if it["kind"] == "app" else "chapter"
        elif kind == "section":
            sec += 1; sub = 0
            if numbered:
                num = "%s.%d" % (p, sec); text = num + "&nbsp;&nbsp;"
        elif kind == "subsection":
            sub += 1
            if numbered:
                num = "%s.%d.%d" % (p, sec, sub); text = num + "&nbsp;&nbsp;"
        elif kind == "paragraph":                     # labelled paragraph -> its section's number
            kind = "section"
            if numbered:
                num = ("%s.%d.%d" % (p, sec, sub)) if sub else ("%s.%d" % (p, sec))
        elif kind == "lab":
            num = p or ""
            text = "Lab %s" % num
        else:
            counts[kind] += 1
            num = ("%s.%d" % (p, counts[kind])) if numbered else str(counts[kind])
            name = KIND_NAMES[kind][0]
            if kind in ("figure", "table", "listing"):
                text = "%s %s: " % (name, num)
            elif kind == "equation":
                text = ""
            else:
                text = "%s %s" % (name, num)
        texts.append((kind, lid, num, text))
        if lid:
            if lid in labels:
                labels[lid]["dup"] = True
            else:
                labels[lid] = dict(kind=kind, num=num, file=it["file"])
    return texts


def renumber_source(md: str, num) -> str:
    """Numbers written into headings follow the book order: '## Lab 11:' takes the chapter's number."""
    if num is None:
        return md
    return re.sub(r"^## Lab [0-9A-Z]+:", "## Lab %s:" % num, md, flags=re.M)


def renumber_solutions(md: str, num) -> str:
    """A solutions file's 'Chapter N' heading and 'Exercise N.k' headings follow the book order."""
    if num is None:
        return md
    md = re.sub(r"^## (Chapter|Appendix) [0-9A-Z]+:", lambda m: "## %s %s:" % (m.group(1), num), md, flags=re.M)
    md = re.sub(r"\{#(chapter|appendix)-[0-9a-z]+-", lambda m: "{#%s-%s-" % (m.group(1), str(num).lower()), md)
    return re.sub(r"^(#{3,4} Exercise )[0-9A-Z]+\.", lambda m: "%s%s." % (m.group(1), num), md, flags=re.M)


def add_lab_labels(labels: dict):
    """[@lab:x] renders 'Lab N' for the chapter or appendix labelled ch:x / app:x, linking to its lab section."""
    for lid, info in list(labels.items()):
        kind = info.get("kind")
        if kind in ("chapter", "appendix") and ":" in lid:
            slug = lid.split(":", 1)[1]
            sec = "sec:%s:lab" % slug
            anchor = sec if sec in labels else lid
            labels.setdefault("lab:" + slug, dict(kind="lab", num=info["num"], file=labels.get(anchor, info)["file"],
                                                  anchor=anchor))


def fill_markers(body: str, texts):
    it = iter(texts)

    def rep(m):
        kind, lid, num, text = next(it)
        span = '<span class="num">%s</span>' % text if text else ""
        anchor = ' id="%s"' % html.escape(lid) if kind == "equation" and lid else ""
        return '<span class="xn"%s>%s</span>' % (anchor, span) if (span or anchor) else ""
    body = MARK.sub(rep, body)
    for kind, lid, num, _ in texts:
        if kind == "equation" and lid:
            body = body.replace("@@EQ:%s@@" % lid, num)
    return body


def resolve_xrefs(body: str, page_file: str, labels: dict, unresolved: dict, published=None):
    def link(lid, text):
        info = labels.get(lid)
        if not info:
            unresolved.setdefault(lid, set()).add(page_file)
            return '<span class="xref-missing">??</span>'
        if published is not None and info["file"] not in published:
            return '<span class="xref-unpub" title="Not published yet">%s</span>' % text
        target = ("" if info["file"] == page_file else info["file"]) + "#" + info.get("anchor", lid)
        return '<a href="%s">%s</a>' % (html.escape(target), text)

    def rep(m):
        ids = [x.strip() for x in html.unescape(m.group(1)).split(";") if x.strip()]
        if m.group(2) == "ref":
            return ", ".join(link(i, labels.get(i, {}).get("num", "??")) for i in ids)
        groups = []
        for i in ids:
            kind = labels.get(i, {}).get("kind", "?")
            if groups and groups[-1][0] == kind:
                groups[-1][1].append(i)
            else:
                groups.append((kind, [i]))
        parts = []
        for kind, ls in groups:
            sing, plur = KIND_NAMES.get(kind, ("", ""))
            nums = []
            for i in ls:
                num = labels.get(i, {}).get("num", "??")
                nums.append(link(i, "(%s)" % num if kind == "equation" else num))
            joined = nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]
            name = sing if len(nums) == 1 else plur
            parts.append((name + "&nbsp;" if name else "") + joined)
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return XREF.sub(rep, body)


# ----------------------------------------------------------------------------- site helpers
def size_figures(body: str, out: Path, px_per_pt: float = 2.1) -> str:
    """About 2x typeset size so figure text reads at body size (CSS caps it at the column)."""
    def rep(m):
        tag, src = m.group(0), m.group(1)
        try:
            head = (out / src).open(encoding="utf-8", errors="ignore").read(600)
            w = float(re.search(r'<svg[^>]*?\swidth="([\d.]+)(?:pt)?"', head).group(1))
        except Exception:
            return tag
        if "style=" in tag:
            return tag
        return tag.replace("<img ", '<img style="width: %dpx" ' % int(round(w * px_per_pt)), 1)
    return re.sub(r'<img [^>]*src="(figures/[^"]+\.svg)"[^>]*>', rep, body)


def theme_figures(body: str) -> str:
    """Each book figure is emitted twice, light and dark; CSS shows the one matching the theme."""
    def rep(m):
        tag = m.group(0)
        src = re.search(r'src="(figures/[^"]+)\.svg"', tag)
        if not src or 'class="fig"' not in tag:
            return tag
        light = tag.replace('class="fig"', 'class="fig fig-light"', 1)
        dark = (tag.replace('class="fig"', 'class="fig fig-dark" aria-hidden="true"', 1)
                .replace(src.group(0), 'src="%s-dark.svg"' % src.group(1), 1))
        return light + dark
    return re.sub(r"<img [^>]*>", rep, body)


def write_dark_figures(out: Path) -> list:
    """Write <name>-dark.svg next to every figure; return figures with colours the converter cannot read."""
    problems = []
    for f in sorted((out / "figures").glob("*.svg")):
        if f.stem.endswith("-dark"):
            continue
        text = f.read_text(encoding="utf-8")
        bad = check_svg(text)
        if bad:
            problems.append((f.name, bad[:5]))
        (f.parent / (f.stem + "-dark.svg")).write_text(dark_svg(text), encoding="utf-8")
    return problems


def build_search_index(pages):
    entries = []
    for p in pages:
        for m in re.finditer(r'<section id="([^"]+)"[^>]*>\s*<h([23])[^>]*>(.*?)</h\2>(.*?)(?=<section|</section>)',
                             p["body"], flags=re.S):
            sid, _, title, text = m.groups()
            text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))).strip()
            title = html.unescape(re.sub(r"<[^>]+>", "", title)).replace("\xa0", " ").strip()
            entries.append(dict(u=p["file"] + "#" + sid, p=p["title"], t=title, x=text[:600]))
    return entries


def page_title(md: str, stem: str) -> str:
    m = re.search(r"^#\s+(.+?)\s*(\{[^}]*\})?\s*$", md, flags=re.M)
    if not m:
        return stem
    t = re.sub(r"[*_`]", "", m.group(1))
    return re.sub(r"\$([^$]*)\$", r"\1", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="book")
    ap.add_argument("--out", default="site")
    ap.add_argument("--strict", action="store_true", help="fail on unresolved references")
    ap.add_argument("--all", action="store_true",
                    help="publish every written page, ignoring published_parts in book.yml (local preview)")
    args = ap.parse_args()
    book, out = Path(args.book), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = parse_book_yml(book / "book.yml")
    items, missing = book_items(book, spec)
    lua = HERE / "filters.lua"
    sol_page = spec["meta"].get("solutions_appendix")
    # Publication: every page is converted (so numbers and labels are right), but only the parts that are
    # finished get written to the site. Links into unpublished pages render as plain text.
    n_parts = spec["meta"].get("published_parts")
    pub_apps = set(spec["meta"].get("published_appendices", "").split())
    part_names = [pt["title"] for pt in spec["parts"]]
    for it in items:
        if args.all or n_parts is None or it["kind"] == "front":
            it["pub"] = True
        elif it["kind"] == "chapter":
            it["pub"] = part_names.index(it["part"]) < int(n_parts)
        else:
            it["pub"] = it["num"] in pub_apps
    hidden_parts = [] if (args.all or n_parts is None) else part_names[int(n_parts):]

    print("[1/3] converting %d pages (%d planned pages not written yet)" % (len(items), len(missing)))
    for it in items:
        md = renumber_source(read(book / it["src"]), it["num"])
        if sol_page and it["src"] == sol_page:
            for other in items:
                sol = book / "solutions" / (other["stem"].split("_")[0] + "_solutions.md")
                if other["kind"] != "front" and other["pub"] and sol.exists():
                    md += "\n\n" + renumber_solutions(read(sol), other["num"])
        it["title"] = page_title(md, it["stem"])
        body, err = run_pandoc(md, lua)
        warn = [l for l in err.strip().splitlines() if l.strip() and "Deprecated" not in l]
        if warn:
            print("  pandoc warnings for", it["stem"], ":", warn[:3])
        it["raw"] = body

    print("[2/3] numbering and cross-references")
    labels = {}
    for it in items:
        m = re.search(r"<!--TOC-->(.*?)<!--/TOC-->", it["raw"], flags=re.S)
        it["toc_raw"] = m.group(1) if m else ""
        it["body_raw"] = re.sub(r"<!--TOC-->.*?<!--/TOC-->", "", it["raw"], flags=re.S)
        it["texts"] = number_page(it["body_raw"], it, labels)
    add_lab_labels(labels)
    unresolved = {}
    by_id = {}
    for it in items:
        for kind, lid, num, text in it["texts"]:
            if kind in ("section", "subsection") and lid:
                by_id[lid] = text
    published = {it["file"] for it in items if it["pub"]}
    for it in items:
        body = fill_markers(it["body_raw"], it["texts"])
        it["body"] = resolve_xrefs(body, it["file"], labels, unresolved, published)
        toc = MARK.sub(lambda m: ('<span class="num">%s</span>' % by_id[html.unescape(m.group(2))])
                       if by_id.get(html.unescape(m.group(2))) else "", it["toc_raw"])
        toc = XREF.sub("", toc)
        inner = re.sub(r"^\s*<nav[^>]*>|</nav>\s*$", "", toc.strip())
        tm = re.match(r'\s*<ul>\s*<li><a[^>]*>.*?</a>\s*(<ul>.*</ul>)\s*</li>\s*</ul>\s*$', inner, flags=re.S)
        it["toc"] = '<nav id="TOC" role="doc-toc">' + (tm.group(1) if tm else inner) + "</nav>"

    print("[3/3] assembling site")
    built = items
    items = [it for it in items if it["pub"]]
    if (out / "figures").exists():
        shutil.rmtree(out / "figures")
    if (book / "figures").exists():
        shutil.copytree(book / "figures", out / "figures")
        if len(items) < len(built):              # ship only the figures that published pages use
            used = set()
            for it in items:
                used |= set(re.findall(r'figures/([^"\'?#\s)]+)', it["body"]))
            for f in (out / "figures").iterdir():
                if f.name not in used:
                    f.unlink()
        fig_problems = write_dark_figures(out)
        for name, bad in fig_problems:
            print("  figure colour the dark-mode converter cannot read:", name, bad)
        if fig_problems and args.strict:
            raise SystemExit("strict: %d figures use unsupported colour notation" % len(fig_problems))
    template = read(HERE / "template.html")
    chev = ('<svg class="chev" viewBox="0 0 24 24" width="12" height="12" aria-hidden="true">'
            '<path d="M9 6l6 6-6 6" stroke="currentColor" stroke-width="2.4" fill="none" '
            'stroke-linecap="round"/></svg>')

    def link(it):
        label = ("%s&nbsp; " % it["num"]) if it["num"] else ""
        return ('<li><a href="%s" data-page="%s"><span class="num">%s</span>%s</a></li>' %
                (it["file"], it["file"], label, html.escape(it["title"])))

    nav = ['<ul class="nav-front">'] + [link(it) for it in items if not it["part"]] + ["</ul>"]
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

    for it in built:                             # pages withdrawn since an earlier build into this directory
        if not it["pub"] and (out / it["file"]).exists():
            (out / it["file"]).unlink()
    for i, it in enumerate(items):
        body = theme_figures(size_figures(it["body"], out))
        it["body"] = body
        prev_l = ('<a class="prev" href="%s">&larr; %s</a>' % (items[i - 1]["file"], html.escape(items[i - 1]["title"]))) if i > 0 else "<span></span>"
        next_l = ('<a class="next" href="%s">%s &rarr;</a>' % (items[i + 1]["file"], html.escape(items[i + 1]["title"]))) if i + 1 < len(items) else "<span></span>"
        page = (template.replace("$TITLE$", html.escape(it["title"]))
                .replace("$NAV$", nav_html).replace("$TOC$", it["toc"]).replace("$BODY$", body)
                .replace("$PREV$", prev_l).replace("$NEXT$", next_l).replace("$PDF$", ""))
        write(out / it["file"], page)

    cards, cur = [], None
    for it in items:
        if it["kind"] == "front":
            continue
        if it["part"] != cur:
            if cur is not None:
                cards.append("</ul></section>")
            cur = it["part"]
            cards.append('<section class="part-card"><h3>%s</h3><ul>' % html.escape(cur))
        cards.append('<li><a href="%s"><span class="num">%s</span> %s</a></li>' %
                     (it["file"], it["num"], html.escape(it["title"])))
    cards.append("</ul></section>")
    if hidden_parts:
        cards.append('<p class="parts-coming">Coming next, each published as it is completed: %s.</p>' %
                     html.escape(", ".join(hidden_parts)))
    landing = read(HERE / "landing.html")
    landing = (landing.replace("$NAV$", nav_html).replace("$CARDS$", "\n".join(cards))
               .replace("$FIRST$", items[0]["file"]).replace("$PDF$", ""))
    write(out / "index.html", landing)

    for f in (HERE / "assets").iterdir():
        shutil.copy(f, out / f.name)
    if (out / "mathjax").exists():
        shutil.rmtree(out / "mathjax")
    shutil.copytree(HERE / "vendor" / "mathjax", out / "mathjax")
    idx = build_search_index(items)
    write(out / "search.json", json.dumps(idx, ensure_ascii=False))
    write(out / ".nojekyll", "")

    dups = sorted(k for k, v in labels.items() if v.get("dup") and ":" in k)
    print("done:", out.resolve(), "pages:", len(items), "labels:", len(labels), "search entries:", len(idx))
    if len(items) < len(built):
        print("built but not published (%d): %s" % (len(built) - len(items),
              ", ".join(it["stem"] for it in built if not it["pub"])))
    if missing:
        print("not written yet:", ", ".join(Path(m).stem for m in missing))
    if dups:
        print("duplicate labels:", ", ".join(dups))
    if unresolved:
        print("unresolved references (%d):" % len(unresolved))
        for lid, files in sorted(unresolved.items()):
            print("  %-40s in %s" % (lid, ", ".join(sorted(files))))
        allow = set()
        allow_file = book / "forward_refs.txt"        # labels of chapters/sections still to be written
        if allow_file.exists():
            allow = {l.split("#")[0].strip() for l in read(allow_file).splitlines()} - {""}
        blocking = sorted(set(unresolved) - allow)
        stale = sorted(allow - set(unresolved))
        if stale:
            print("forward_refs.txt entries that now resolve (remove them):", ", ".join(stale))
        if args.strict and blocking:
            raise SystemExit("strict: unresolved references not in book/forward_refs.txt: " + ", ".join(blocking))


if __name__ == "__main__":
    main()
