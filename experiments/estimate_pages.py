"""Estimate the rendered page count of the ICLR-format report without a LaTeX installation.

Models the template's geometry directly: a 5.5in x 9in text block at 10pt on 11pt leading gives
about 59 lines per page and roughly 88 characters per line in Times. Text is measured by
wrapping each paragraph; figures, tables, equations and headings are charged their vertical
space in lines. Accurate to roughly half a page, which is enough to decide what to cut.
"""
import re
import sys
from pathlib import Path

CHARS_PER_LINE = 88
LINES_PER_PAGE = 59
COST = {                      # vertical cost in text-lines
    "section": 4, "subsection": 3.2, "paragraph": 1.2,
    "equation": 4.0, "align": 5.5,
    "figure_base": 3.5,       # caption + spacing; image height added separately
    "table_row": 1.15, "table_base": 5.0,
    "itemize_item": 0.4,      # extra beyond the wrapped text itself
}
# rendered height of each figure, in text-lines (from the .pdf aspect at \linewidth = 5.5in)
FIG_LINES = {   # measured from each PDF's aspect ratio at width = 5.5in, 11pt leading
    "fig1_schematic.pdf": 13.8, "fig2_low_label.pdf": 15.1, "fig3_sbm_heatmap.pdf": 14.5,
    "fig4_crossover_beta.pdf": 14.1, "fig5_sensitivity.pdf": 13.2, "fig6_fidelity.pdf": 10.4,
    "fig7_bias_variance.pdf": 25.9,
}


def strip(s):
    s = re.sub(r"\\(cite[tp]?|citealp)\{[^}]*\}", "Author et al. (2020)", s)
    s = re.sub(r"\\(ref|label|eqref)\{[^}]*\}", "1", s)
    s = re.sub(r"\\(emph|textbf|textit|texttt|textsc|mathrm|mathbf)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    return re.sub(r"[{}$\\]", "", s)


def measure(tex):
    lines = 0.0
    body = tex[tex.index(r"\begin{abstract}"):]
    # remove float bodies first, charge them separately
    for env, base in (("figure", "figure_base"), ("table", "table_base")):
        for block in re.findall(rf"\\begin{{{env}}}.*?\\end{{{env}}}", body, re.S):
            if env == "figure":
                fn = re.search(r"includegraphics\[[^\]]*\]\{([^}]+)\}", block)
                w = re.search(r"width=([\d.]*)\\linewidth", block)
                scale = float(w.group(1)) if w and w.group(1) else 1.0
                lines += COST["figure_base"] + FIG_LINES.get(fn.group(1), 10) * scale
                cap = re.search(r"\\caption\{(.*?)\}\s*\\label", block, re.S)
                if cap:
                    lines += len(strip(cap.group(1))) / CHARS_PER_LINE
            else:
                nrows = block.count(r"\\")
                lines += COST["table_base"] + nrows * COST["table_row"]
                cap = re.search(r"\\caption\{(.*?)\}\s*\\label", block, re.S)
                if cap:
                    lines += len(strip(cap.group(1))) / CHARS_PER_LINE
            body = body.replace(block, "")
    # standalone tabulars (the inline rewired table)
    for block in re.findall(r"\\begin\{center\}\s*\\small\s*\\begin\{tabular\}.*?\\end\{center\}",
                            body, re.S):
        lines += COST["table_base"] + block.count(r"\\") * COST["table_row"]
        body = body.replace(block, "")
    for env in ("equation", "align"):
        for block in re.findall(rf"\\begin{{{env}}}.*?\\end{{{env}}}", body, re.S):
            lines += COST[env]
            body = body.replace(block, "")
    for kind, pat in (("section", r"\\section\{"), ("subsection", r"\\subsection\{"),
                      ("paragraph", r"\\paragraph\{")):
        lines += len(re.findall(pat, body)) * COST[kind]
    lines += len(re.findall(r"\\item", body)) * COST["itemize_item"]
    for para in re.split(r"\n\s*\n", body):
        txt = strip(para).strip()
        if txt:
            lines += max(1, len(txt) / CHARS_PER_LINE)
    return lines


if __name__ == "__main__":
    tex = Path(sys.argv[1] if len(sys.argv) > 1 else "report/report.tex").read_text()
    main = tex[:tex.index(r"\appendix")]
    total = measure(main)
    print(f"main-text lines  : {total:.0f}")
    print(f"lines per page   : {LINES_PER_PAGE}")
    print(f"ESTIMATED PAGES  : {total / LINES_PER_PAGE:.1f}   (title+abstract block included)")
    over = total / LINES_PER_PAGE - 10.0
    if over > 0:
        print(f"\nOVER by {over:.1f} pages -> cut ~{int(over * LINES_PER_PAGE * CHARS_PER_LINE / 5.9)} words"
              f" (or move one figure to the appendix, worth ~0.25 page)")
    else:
        print(f"\nWithin the 10-page limit, {-over:.1f} pages of headroom.")
