# Online edition

The book is published only as a website. Its sources are Markdown files in `book/` (pandoc
dialect) plus SVG figures in `book/figures/`; `build_site.py` turns them into the GitHub Pages site:
light, sepia and dark themes, adjustable text size, serif or sans text, collapsible chapter list and
page contents, search, large click-to-zoom figures, and maths with bundled MathJax.

```
python3 web/build_site.py --book book --out site
python3 -m http.server -d site 8000               # preview at http://localhost:8000
```

Requirements: Python 3.9+ and pandoc >= 3. Nothing else.

## Writing a chapter

`book/book.yml` lists the parts and chapters in reading order. A listed file that does not exist
yet is skipped, so chapter numbers follow the planned order and never shift. Each chapter is one
Markdown file:

```markdown
# Modern Architectures: Qwen3 from Scratch {#ch:modern}

## RMSNorm {#sec:modern:rmsnorm}

$$\mathrm{RMSNorm}(\boldsymbol{x}) = \boldsymbol{\gamma}\odot\frac{\boldsymbol{x}}{\mathrm{RMS}(\boldsymbol{x})}
\label{eq:modern:rms}$$

![Caption text.](figures/ch06_modern_arch-2.svg){#fig:modern:rmsvsln}

``` {#lst:modern:rms .python caption="RMSNorm with an fp32 accumulation."}
...
```

::: {#tab:modern:diff .table-float}
| Component | GPT-2 | Qwen3 |
|:--|:--|:--|
: What changed between GPT-2 and Qwen3.
:::

::: systemslens
[Why RMSNorm is faster]{.box-title}

Body text.
:::
```

- **Numbering** is automatic and per chapter: chapters, `##` sections and `###` subsections
  (add `{.unnumbered}` to skip one), captioned figures and tables, listings with a `caption`,
  display equations with `\label{eq:...}`, exercises and propositions.
- **Cross-references** are citations: `[@fig:modern:rmsvsln]` renders "Figure 6.2",
  `[@sec:a; @sec:b]` "Sections 6.1 and 6.2", and `[-@eq:modern:rms]` the bare number.
  The build lists every unresolved label; `--strict` makes them an error.
- **Boxes** are fenced divs: `systemslens`, `keyidea`, `pitfall`, `napkinmath`, `deepdive`,
  `exercise`, `proposition`, and `labbox` (numbered with its chapter). An optional first paragraph
  `[Title]{.box-title}` gives the title.
- **Figures** are SVG files in `book/figures/`. A figure with more than one image or extra text is
  a `::: {#fig:x .figure}` div whose last paragraph is `[Caption]{.figure-caption}`.
- **Solutions** live in `book/solutions/chNN_solutions.md` and are appended, in book order, to the
  solutions appendix.

The workflow in `.github/workflows/pages.yml` runs the same build on every push to `main` and
publishes the result to the `gh-pages` branch.
