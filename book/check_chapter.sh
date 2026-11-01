#!/usr/bin/env bash
# Compile a single chapter/appendix standalone to catch LaTeX errors early.
#   ./check_chapter.sh chapters/ch04_attention.tex
set -u
BOOK="$(cd "$(dirname "$0")" && pwd)"
SRC="$1"; NAME="$(basename "$SRC" .tex)"
OUT="$BOOK/_build_check/$NAME"; mkdir -p "$OUT"
cat > "$OUT/$NAME-wrap.tex" <<TEX
\documentclass[11pt,twoside,openany]{book}
\usepackage{$BOOK/preamble/llmsys}
\graphicspath{{$BOOK/figures/}}
\begin{document}
\setcounter{chapter}{\numexpr ${CHNUM:-0}-1\relax}
\input{$BOOK/$SRC}
\end{document}
TEX
cd "$OUT"
pdflatex -interaction=nonstopmode -halt-on-error "$NAME-wrap.tex" > build.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error "$NAME-wrap.tex" > build.log 2>&1
status=$?
grep -n -A4 "^!" build.log | head -40
grep -c "Overfull \\\\hbox" build.log | xargs -I{} echo "overfull hboxes: {}"
if [ -f "$NAME-wrap.pdf" ]; then pdfinfo "$NAME-wrap.pdf" | grep Pages; fi
echo "exit status: $status   (pdf: $OUT/$NAME-wrap.pdf)"
