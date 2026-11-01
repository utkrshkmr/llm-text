# Online edition

`build_site.py` turns the LaTeX sources into the GitHub Pages website (light and dark mode,
search, per-page contents, every TikZ figure rendered to SVG, maths with bundled MathJax).

```
cd book && latexmk -pdf main.tex && cd ..          # produces the .aux files the site needs
python3 web/build_site.py --book book --aux book --pdf book/main.pdf --out site
python3 -m http.server -d site 8000               # preview at http://localhost:8000
```

Requirements: TeX Live (with `preview`), pandoc >= 3, poppler (`pdftocairo`, `pdfinfo`).
The workflow in `.github/workflows/pages.yml` runs the same steps on every push to `main`
and publishes the result to the `gh-pages` branch.
