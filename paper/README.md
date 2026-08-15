# Compiling the report

Two papers share this folder, the same bibliography, the same figures and the same unmodified
ICLR 2026 template files:

| File | What it is | Length |
|---|---|---|
| `final_report_WaleedAhmed_ICLR2026.tex` | **Primary submission.** Structured to the conference-draft rubric: Introduction and research problem · Related work · Approach · Results · Analysis · Limitations · Conclusion. | ~4,235 words, ≈8.9 pages of content |
| `report_cmpt727.tex` | Course-report variant, organised as Introduction/Background · Methods · Results · Discussion, with a table mapping CMPT 727 concepts to what was implemented. | ≈10.1 pages of content |

Both draw on the identical experiments and report identical numbers.

## Overleaf

1. Upload this `paper/` folder as a new Overleaf project.
2. **Set the main document** (Menu → Main document) — both `.tex` files have a `\documentclass`,
   so Overleaf may pick the wrong one. Choose `final_report_WaleedAhmed_ICLR2026.tex`.
3. Compiler: **pdfLaTeX**.
4. Recompile **twice**: the first pass writes the `.aux`, the second resolves citations and
   figure references. Otherwise you will see `[?]` and `??`.
5. The PDF downloads as `final_report_WaleedAhmed_ICLR2026.pdf` — Overleaf names output after
   the main file.

If you would rather upload only one paper, delete the other `.tex` before uploading and step 2
becomes unnecessary.

## Figures

`\graphicspath{{figures/}}` is set, so `figures/*.pdf` resolves automatically. The figures are
vector PDFs drawn at exactly the ICLR text width (5.5 in) — insert them at `\linewidth` or below
and do not enlarge them, which would push fonts below the template's minimum size. The matching
`.png` files exist only so GitHub can render them in the README; LaTeX prefers the `.pdf`.

Regenerate everything from the raw results with:

```bash
.venv/bin/python -m experiments.make_paper_figures   # writes paper/figures/
```

## Checking the length

No LaTeX is installed in the project environment, so the page count is estimated by modelling
the template geometry:

```bash
.venv/bin/python -m experiments.estimate_pages paper/final_report_WaleedAhmed_ICLR2026.tex
```

It is accurate to roughly half a page — verify against the compiled PDF (content runs from
page 1 to wherever `References` begins). Overleaf's Menu → Word Count gives the word total.

## Assembling a submission archive

The assignment asks for a zip containing the PDF and the TeX sources. After compiling, download
the PDF into this folder and run from the project root:

```bash
cd paper && zip -r ../final_report_WaleedAhmed_ICLR2026.zip . -x '*.DS_Store' && cd ..
```
