# Brief: retrofit of Chapter 1 "The Landscape" and the front matter for the expanded book

Read first: BRIEF_conventions.md (all, esp. §1 numbering, §7 depth, §8 figures), docs/PLAN.md §1 (the 68-chapter
outline with capstones at the end of every part) and §4, docs/COVERAGE.md, book/book.yml, the existing
book/chapters/ch01_landscape.md (keep all content that is still true), book/solutions/ch01_solutions.md,
book/frontmatter/preface.md and roadmap.md, code/labs/lab01_napkin/*.

Files you own: book/chapters/ch01_landscape.md, book/solutions/ch01_solutions.md, book/frontmatter/preface.md,
book/frontmatter/roadmap.md, book/figures/ch01_landscape-12.svg (the book map; replace it), any new
book/figures/ch01_landscape-<k>.svg, code/labs/lab01_napkin/* (extend only), docs/edition3_briefs/refs/ch01_landscape.md.

Work:
1. The book is now "Big Book of LLMs: A Systems-First Textbook". Rewrite §"A roadmap for this book"
   {#sec:landscape:roadmap} and its "How to read this book" for the new structure: seven parts, each ending with its
   capstone projects (list them by [@ch:projN] reference), the new chapters (numerics, GPU architecture, profiling,
   compilers, memory/interconnects, KV-cache management, the long-haul data-movement study, the RL setup) and
   appendices E (containers) and G (doing research). Give reading paths: systems researcher, inference engineer,
   training engineer, post-training/RL researcher. Use only [@ch:...] references (many targets are not written yet;
   that is expected and allowed by book/forward_refs.txt only for labels listed there — add NO new literal numbers;
   if you reference a not-yet-written chapter label that is not in book/forward_refs.txt, list it in your report
   so it can be added centrally; do not edit forward_refs.txt yourself).
2. Redraw the book map figure as a hand-written or matplotlib-generated SVG following §8 (house palette, white
   boxes, dark ink): parts as columns or bands, chapters as boxes coloured by theme, capstones marked. It must stay
   legible at the site's figure width and in the dark theme (the build converts it; check both renderings with
   cairosvg: `python -c "import cairosvg; ..."` on site_check/figures/ch01_landscape-12.svg and -dark.svg, and look
   at the PNGs).
3. §7 depth retrofit: add `## Research frontier {#sec:landscape:frontier}` (the field as a whole: where LLM systems
   research is heading — reasoning/test-time compute, hybrid and sparse architectures, low precision, disaggregated
   serving, RL systems, hardware co-design — with 5-10 defining works and the "how to place a new paper" axes),
   one "Reading the research" deepdive box (e.g. Kaplan et al. 2020 or "Attention Is All You Need" read for its cost
   model), and at least two research-style exercises (with solutions).
4. Preface and frontmatter/roadmap.md: update to the new title, structure, goal (making the reader able to do and
   follow LLM-systems research) and the per-part capstones; keep the author's first-person voice (see
   BRIEF_conventions §2 "Voice"), no meta-commentary.
5. Check the napkin-math lab still matches the text; extend it only if the new sections need it.

Verification: LAB_IMPL=solution python -m pytest -q code/labs/lab01_napkin; the site check of BRIEF_conventions §2
(no new unresolved labels except chapters not yet written, reported); both figure themes inspected.
