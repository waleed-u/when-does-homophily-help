# How to assemble the report in Overleaf

## 1. Files to change

Your Overleaf project already has the template files (`iclr2026_conference.sty`, `.bst`,
`fancyhdr.sty`, `natbib.sty`, `math_commands.tex`). Change **two**, add **seven**.

### Replace
| Overleaf file | Replace with |
|---|---|
| `report.tex` | `report/report.tex` from this repo |
| `iclr2026_conference.bib` | `report/iclr2026_conference.bib` from this repo (22 references) |

### Upload (drag into the project **root**, next to `report.tex`)
```
fig1_schematic.pdf        fig2_low_label.pdf     fig3_sbm_heatmap.pdf
fig4_crossover_beta.pdf   fig5_sensitivity.pdf   fig6_fidelity.pdf
fig7_bias_variance.pdf
```
They are in `report/` (already copied there) and in `report_figures/`. These are vector PDFs
drawn at the exact ICLR text width (5.5 in), so they go in at ~full scale — do not enlarge them,
which would push fonts past the template's style.

## 2. Compile

- Compiler: **pdfLaTeX** · Main document: `report.tex`
- Press Recompile **twice** — the first pass writes the `.aux`, the second resolves citations
  and figure numbers. Otherwise you will see `[?]` and `??`.

## 3. Length

Main text is estimated at **~10.1 pages** (excluding references and appendix), against the
instructor's 6–10 page window. The estimate is accurate to roughly half a page, so check the
compiled PDF: main content runs from page 1 to wherever `References` begins.

**If it exceeds 10 pages,** apply these in order — each is a small, self-contained edit:

1. **Delete the CiteSeer block from Table 2** (the three `\multirow{3}{*}{\rotatebox{90}{CiteS.}}`
   rows). The CiteSeer numbers are already stated in the §5.1 text, so no evidence is lost.
   *Saves ~0.15 page.*
2. **Move Figure 5** (`fig5_sensitivity.pdf`) to the appendix: cut the `figure` environment and
   paste it after `\subsection{Additional material}`. The text keeps its `\ref`.
   *Saves ~0.28 page.*
3. **Delete the `\subsubsection*{Reproducibility}` block** — the same information is in the
   appendix amendments section. *Saves ~0.1 page.*
4. **Shorten §2 Background** to the first two paragraphs. *Saves ~0.15 page.*

**If you have room to spare,** the reverse move is worthwhile: bring
`fig6_fidelity.pdf` back from the appendix into §5.6 (it is currently in the appendix purely for
space). It is the figure that most directly demonstrates the exact-inference / BP / MCMC work.

You can re-check the estimate at any time with:
```bash
.venv/bin/python -m experiments.estimate_pages report/report.tex
```

## 4. How the report is organised

| Section | Content | Assets |
|---|---|---|
| §1 Introduction | where the project started, why the question was sharpened, **Table 1: course concepts → what we implemented → source followed** | Table 1 |
| §2 Background | setting, why a GCN already assumes homophily, relation to prior work | — |
| §3 Methods | the 5-step pipeline, the three impositions, Proposition 1, the four inference algorithms, the synthetic generator, the two prior regimes, correctness gates | Figure 1 |
| §4 Setup | datasets, pre-registered protocol, statistics | Table 2 |
| §5 Results | Q1 scarcity, Q2 homophily, Q3 mechanism, consistency/calibration, inference validation, robustness | Figures 2–5, Table 3 |
| §6 Discussion | what we learned, limitations, future work (EM-learned compatibility) | — |
| Appendix | the three pre-registered comparisons, protocol amendments, Figures 6–7 | Figures 6–7 |

Every number in the text and tables is already filled in from the experiments — nothing is a
placeholder.

## 5. Regenerating figures

```bash
.venv/bin/python -m experiments.make_paper_figures   # -> report_figures/*.pdf
cp report_figures/*.pdf report/
```
All figures are generated from `results/raw/*.csv`; nothing is drawn by hand.

## 6. Optional appendix tables

`tables/*.tex` holds LaTeX versions of every table the analysis produced (T1–T7), including the
full synthetic sweep (T4), per-structure inference accuracy (T5), all controls (T6) and the
exploratory β-extension (T7). Add one with `\input{T4_sbm_sweep.tex}` after uploading it; wrap
wide tables in `\resizebox{\linewidth}{!}{...}`.
