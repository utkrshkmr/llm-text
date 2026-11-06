# Brief: additions to Chapter 19 "Pretraining Data" + lab extension

Read first: BRIEF_conventions.md (all, esp. §7), the existing chapter book/chapters/ch19_pretraining_data.md (keep all
content), its solutions file and code/labs/lab19_datapipe/*.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Data-mixture optimisation: DoReMi (proxy model, group DRO weights), RegMix (regression over small-model mixtures),
  online data mixing; implement a small DoReMi-style loop.
- Synthetic data in depth: self-instruct and Evol-Instruct, textbook-style synthetic corpora (Phi line), rephrasing
  web data, verification-filtered synthetic reasoning data; model collapse (theory and evidence) and how to avoid it
  (mixing real data, provenance tracking).
- Public corpora and their recipes: RedPajama, Dolma, FineWeb/FineWeb-Edu, DCLM, Nemotron-CC (what each filter does).
