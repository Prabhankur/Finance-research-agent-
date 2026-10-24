"""
nodes/report_saver.py  –  Node 7: Saves the final report to disk.

Writes:
  1. A clean .md (Markdown) file – always.
  2. A basic .docx (Word) file – always (using python-docx).
     Full styled DOCX with tables is a Phase 2 enhancement
     (use the docx skill for that when ready).
"""

from __future__ import annotations
import os
import re
from datetime import datetime
from pathlib import Path
from graph.state import AgentState


def _md_to_docx(md_text: str, output_path: str, ticker: str, chart_paths: list = None) -> None:
    """
    Converts markdown report text to a .docx file with embedded charts.
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document()

        # ── Page margins ──────────────────────────────────────────────────────
        for section in doc.sections:
            section.top_margin    = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin   = Inches(1.2)
            section.right_margin  = Inches(1.2)

        # ── Title styling ─────────────────────────────────────────────────────
        doc.add_heading(f"Financial Analysis Report – {ticker}", 0)

        lines = md_text.split("\n")
        i     = 0

        while i < len(lines):
            line = lines[i]

            # Skip horizontal rules
            if line.strip().startswith("---"):
                i += 1
                continue

            # Headings
            if line.startswith("#### "):
                doc.add_heading(line[5:].strip(), level=4)
            elif line.startswith("### "):
                doc.add_heading(line[4:].strip(), level=3)
            elif line.startswith("## "):
                doc.add_heading(line[3:].strip(), level=2)
            elif line.startswith("# "):
                doc.add_heading(line[2:].strip(), level=1)

            # Markdown table (look-ahead for full table)
            elif line.startswith("|"):
                table_lines = []
                while i < len(lines) and lines[i].startswith("|"):
                    table_lines.append(lines[i])
                    i += 1

                # Filter out separator rows
                data_rows = [
                    r for r in table_lines
                    if not re.match(r"^\|[\s\-:|]+\|", r)
                ]

                if data_rows:
                    parsed = [
                        [cell.strip() for cell in row.strip("|").split("|")]
                        for row in data_rows
                    ]
                    cols = max(len(r) for r in parsed)

                    tbl = doc.add_table(rows=len(parsed), cols=cols)
                    tbl.style = "Table Grid"

                    for row_idx, row_data in enumerate(parsed):
                        for col_idx, cell_text in enumerate(row_data):
                            if col_idx < cols:
                                cell = tbl.rows[row_idx].cells[col_idx]
                                # Strip inline bold from cell text
                                cell_text = re.sub(r"\*\*(.+?)\*\*", r"\1", cell_text)
                                cell.text = cell_text
                                if row_idx == 0:
                                    for para in cell.paragraphs:
                                        for run in para.runs:
                                            run.bold = True

                    doc.add_paragraph()  # spacing after table
                continue  # already advanced i in while loop above

            # Bullet points
            elif line.startswith("- ") or line.startswith("* "):
                doc.add_paragraph(line[2:].strip(), style="List Bullet")

            # Numbered lists
            elif re.match(r"^\d+\.\s", line):
                doc.add_paragraph(re.sub(r"^\d+\.\s", "", line).strip(), style="List Number")

            # Blockquotes
            elif line.startswith("> "):
                p = doc.add_paragraph(line[2:].strip())
                p.style = "Quote"

            # Inline image reference: ![alt](filename.png)
            elif line.strip().startswith("!["):
                img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
                if img_match:
                    alt_text = img_match.group(1)
                    img_file = img_match.group(2)
                    # Look for the image in output dir or chart_paths
                    img_path = None
                    for cp in (chart_paths or []):
                        if os.path.basename(cp) == img_file and os.path.exists(cp):
                            img_path = cp
                            break
                    if not img_path and os.path.exists(img_file):
                        img_path = img_file
                    if not img_path:
                        # Try output directory
                        candidate = os.path.join("output", img_file)
                        if os.path.exists(candidate):
                            img_path = candidate

                    if img_path:
                        doc.add_picture(img_path, width=Inches(5.5))
                        cap = doc.add_paragraph(
                            alt_text or img_file.replace("_", " ").replace(".png", "")
                        )
                        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        if cap.runs:
                            cap.runs[0].font.size = Pt(9)
                            cap.runs[0].font.color.rgb = RGBColor(0x6c, 0x75, 0x7d)

            # Normal paragraph (with inline bold support)
            elif line.strip():
                p   = doc.add_paragraph()
                # Split on **bold** markers
                parts = re.split(r"(\*\*[^*]+\*\*)", line)
                for part in parts:
                    if part.startswith("**") and part.endswith("**"):
                        run = p.add_run(part[2:-2])
                        run.bold = True
                    else:
                        p.add_run(part)

            i += 1

        doc.save(output_path)
        print(f"[Node 7] DOCX saved: {output_path}")

    except ImportError:
        print("[Node 7] python-docx not installed. Skipping DOCX output.")
    except Exception as exc:
        print(f"[Node 7] DOCX error: {exc}")


def report_saver_node(state: AgentState) -> dict:
    """
    Node 7: Saves the final report as .md and .docx files, with charts.
    """
    report_text = state.get("final_report", "")
    ticker      = state.get("ticker", "UNKNOWN")
    chunks      = state.get("context_chunks", [])

    if not report_text:
        print("[Node 7] No report text to save.")
        return {"report_path": ""}

    # ── Create output directory ───────────────────────────────────────────────
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name  = f"{ticker}_financial_report_{timestamp}"

    # ── Collect chart paths from inline markdown references ─────────────────
    # Charts are now generated by report_generator inline after each section.
    # We just need to find the image references in the markdown to pass to
    # the docx converter for embedding.
    import re as _re
    chart_paths = []
    for match in _re.finditer(r"!\[[^\]]*\]\(([^)]+\.png)\)", report_text):
        img_file = match.group(1)
        for candidate in [
            os.path.join(str(output_dir), img_file),
            img_file,
        ]:
            if os.path.exists(candidate):
                chart_paths.append(candidate)
                break
    if chart_paths:
        print(f"[Node 7] Found {len(chart_paths)} inline chart(s) for docx embedding.")

    # ── Save Markdown ─────────────────────────────────────────────────────────
    md_path = output_dir / f"{base_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[Node 7] Markdown saved: {md_path}")

    # ── Save DOCX (with embedded charts) ──────────────────────────────────────
    docx_path = str(output_dir / f"{base_name}.docx")
    _md_to_docx(report_text, docx_path, ticker, chart_paths=chart_paths)

    return {"report_path": str(md_path)}
