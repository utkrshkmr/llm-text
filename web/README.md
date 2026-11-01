# Online edition

`build_site.py` turns the LaTeX sources into the GitHub Pages website: light, sepia and dark themes,
adjustable text size, serif or sans text, collapsible chapter list and page contents, search, large
click-to-zoom SVG figures, and maths with bundled MathJax. No PDF is published.

```
(cd book && for i in 1 2; do pdflatex -draftmode -interaction=nonstopmode main.tex; done)  # .aux only
python3 web/build_site.py --book book --aux book --out site
python3 -m http.server -d site 8000               # preview at http://localhost:8000
```

Requirements: TeX Live (with `preview`), pandoc >= 3, poppler (`pdftocairo`, `pdfinfo`).
The workflow in `.github/workflows/pages.yml` runs the same steps on every push to `main`
and publishes the result to the `gh-pages` branch.
