# How to assemble the report in Overleaf

## 1. Files to upload

Your Overleaf project already has the template files (`iclr2026_conference.sty`, `.bst`,
`fancyhdr.sty`, `natbib.sty`, `math_commands.tex`). You need to change **two** files and add
**seven** figures.

### Replace these two
| Overleaf file | Replace with |
|---|---|
| `report.tex` | `report/report.tex` from this repo (the complete paper) |
| `iclr2026_conference.bib` | `report/iclr2026_conference.bib` from this repo (22 references) |

### Upload these seven (drag into the project **root**, next to `report.tex`)
```
fig1_schematic.pdf        fig2_low_label.pdf     fig3_sbm_heatmap.pdf
fig4_crossover_beta.pdf   fig5_sensitivity.pdf   fig6_fidelity.pdf
fig7_bias_variance.pdf
```
They are in `report/` (already copied) and in `report_figures/`. They are vector PDFs drawn at
the exact ICLR text width (5.5 in), so use `width=\linewidth` and do **not** rescale them —
rescaling shrinks the fonts below the template's minimum readable size.

If you prefer a subfolder, upload them to `figures/` and add `\graphicspath{{figures/}}` after
`\usepackage{graphicx}`.

## 2. Compile settings

- Compiler: **pdfLaTeX** (Menu → Compiler).
- Main document: `report.tex`.
- Press Recompile **twice** (once to write `.aux`, once to resolve citations and cross-refs), or
  Overleaf will show `[?]` for citations and `??` for figure references.

## 3. Where each result appears

| Section | Content | Asset |
|---|---|---|
| §4 Experimental setup | dataset statistics | Table 1 (inline, already filled) |
| §5.1 | scarcity result, endpoint C1 | Figure 2, Table 2 |
| §5.2 | homophily sweep, endpoints C2/C3 | Figures 3 and 4 |
| §5.3 | mechanism + rewiring control | inline table (already filled) |
| §5.4 | consistency vs accuracy, calibration | Figure 5 |
| §5.5 | inference validation | Figure 6 |
| Appendix | endpoints, amendments, bias/variance | Figure 7 |

Every number in the text and tables is already filled in from the experiments — nothing is a
placeholder.

## 4. If it runs over the page limit

The draft should land at roughly 9–10 pages of main text. ICLR's initial-submission limit is 9
pages; your instructor allows 6–10. Trim in this order, which costs the least evidence:

1. Move Figure 6 (`fig6_fidelity.pdf`) and §5.5's second paragraph to the appendix. Keep one
   sentence in Methods: *"All inference routines reproduce exact marginals on tractable graphs
   (Appendix)."* — saves ~0.6 page.
2. Drop the CiteSeer rows from Table 2 and cite the numbers inline instead — saves ~0.25 page.
3. Shorten §2 Background to two paragraphs (keep the smoothing and homophily-measurement ones).
4. Delete the `\subsubsection*{Reproducibility statement}` (its content is in the appendix).

To check your page count, look at where `\bibliography` starts — everything before it is main
text.

## 5. Regenerating the figures

```bash
.venv/bin/python -m experiments.make_paper_figures   # -> report_figures/*.pdf
cp report_figures/*.pdf report/
```
All figures are generated from `results/raw/*.csv`; nothing is drawn by hand.

## 6. Optional extra tables

`tables/*.tex` holds LaTeX versions of every table produced by the analysis (T1–T7), including
the full CSBM sweep (T4), per-structure inference fidelity (T5), all controls (T6) and the
exploratory β-extension (T7). Add any of them to the appendix with:

```latex
\input{T4_sbm_sweep.tex}
```
after uploading the corresponding `.tex` file. They are wider than the text block, so wrap them
in `\resizebox{\linewidth}{!}{...}` or `\small`.
