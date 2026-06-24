from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import fitz
import gradio as gr

from pdf_plan_extractor import (
    highlight_keywords_callouts_bytes,
    normalize_keywords,
)

DEFAULT_KEYWORDS = [
    "Concrete",
    "Grout",
    "Vertical and Overhead Patching Material",
    "Joint Filler",
    "Caulking Compound",
    "Joint Sealer",
    "Wire Fabric",
    "Form",
    "Forms",
    "Crack Repairs",
]

APP_CSS = """
.gradio-container { max-width: 1800px !important; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(130px, 1fr));
  gap: 12px;
  margin: 8px 0 14px 0;
}
.kpi-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  padding: 14px 16px;
  background: var(--background-fill-secondary);
  min-height: 84px;
}
.kpi-value { font-size: 26px; font-weight: 700; line-height: 1.1; }
.kpi-label { font-size: 13px; opacity: 0.75; margin-top: 7px; }
.review-note {
  border-left: 4px solid var(--color-accent);
  padding: 8px 12px;
  margin: 4px 0 12px 0;
  background: var(--background-fill-secondary);
  border-radius: 6px;
}
.full-width-table { width: 100% !important; }
@media (max-width: 900px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
}
"""


def _render_pages(
    pdf_bytes: bytes,
    report: dict[str, Any],
    output_dir: Path,
) -> list[tuple[str, str]]:
    """Render every sheet for one full-width document preview."""
    report_by_page = {page["page"]: page for page in report["pages"]}
    rendered: list[tuple[str, str]] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(1.25, 1.25),
                alpha=False,
                annots=True,
            )
            image_path = output_dir / f"page-{page_number}.png"
            pixmap.save(image_path)

            page_report = report_by_page.get(page_number, {})
            flagged_count = len(page_report.get("flagged_callouts", []))
            excluded_count = len(page_report.get("excluded_non_555_mentions", []))
            source = page_report.get("text_source", "native")
            caption = (
                f"Page {page_number} · {flagged_count} flagged · "
                f"{excluded_count} not flagged by rules"
            )
            if source == "ocr":
                caption += " · OCR used"
            elif source == "native_sparse":
                caption += " · limited extraction"
            rendered.append((str(image_path), caption))

    return rendered


def _status_text(item: dict[str, Any]) -> str:
    status = item.get("classification", "confirmed callout")
    qualities = {
        detail.get("match_quality")
        for detail in item.get("match_details", [])
    }
    if "recovered one-character text error" in qualities:
        status += " · text repaired"
    if "recognized abbreviation CONC." in qualities:
        status += " · CONC. = Concrete"
    return status


def _all_finding_rows(report: dict[str, Any]) -> list[list[Any]]:
    """All findings table. Deliberately omits sheet title."""
    rows: list[list[Any]] = []
    for page in report["pages"]:
        for item in page.get("flagged_callouts", []):
            rows.append(
                [
                    page["page"],
                    ", ".join(item.get("matched_keywords", [])),
                    _status_text(item),
                    item.get("text", ""),
                ]
            )
    return rows


def _all_excluded_rows(report: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for page in report["pages"]:
        for item in page.get("excluded_non_555_mentions", []):
            rows.append(
                [
                    page["page"],
                    ", ".join(item.get("exclusion_terms", [])),
                    item.get("text", ""),
                ]
            )
    return rows


def _sheet_rows(report: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            page["page"],
            page.get("sheet_title") or "Untitled sheet",
            page.get("sheet_title_confidence", "unknown"),
            page.get("sheet_title_source", "unknown"),
        ]
        for page in report["pages"]
    ]


def _summary_html(report: dict[str, Any]) -> str:
    findings = [item for page in report["pages"] for item in page.get("flagged_callouts", [])]
    confirmed = sum(1 for item in findings if item.get("classification") == "confirmed callout")
    review = sum(1 for item in findings if item.get("classification") == "review mention")
    excluded = report.get("total_excluded_non_555_mentions", 0)
    no_flag_pages = sum(1 for page in report["pages"] if not page.get("flagged_callouts"))
    pages_analyzed = report.get("pages_analyzed", len(report["pages"]))
    total_pages = report.get("total_pages", len(report["pages"]))
    ocr_pages = report.get("ocr_pages", [])

    ocr_note = f" OCR was used on page(s) {', '.join(map(str, ocr_pages))}." if ocr_pages else ""
    return f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-value">{pages_analyzed}/{total_pages}</div><div class="kpi-label">Pages analyzed</div></div>
      <div class="kpi-card"><div class="kpi-value">{confirmed}</div><div class="kpi-label">Confirmed callouts</div></div>
      <div class="kpi-card"><div class="kpi-value">{review}</div><div class="kpi-label">Review mentions</div></div>
      <div class="kpi-card"><div class="kpi-value">{excluded}</div><div class="kpi-label">Not flagged by rules</div></div>
      <div class="kpi-card"><div class="kpi-value">{no_flag_pages}</div><div class="kpi-label">Pages with no flagged result</div></div>
    </div>
    <div class="review-note"><strong>Review flow:</strong> Scan the full highlighted plan below, then use the tabs in order: <em>Sheet Titles</em>, <em>All Flagged</em>, and <em>Not Flagged by Rules</em>. Every finding already includes its page number.{ocr_note}</div>
    """


def process_pdf(
    pdf_path: str | None,
    include_review_mentions: bool,
):
    if not pdf_path:
        return (
            "<div class='review-note'>Upload a PDF to begin.</div>",
            [],
            [],
            [],
            [],
            None,
            {},
        )

    keywords = normalize_keywords(DEFAULT_KEYWORDS)
    source_path = Path(pdf_path)
    source_bytes = source_path.read_bytes()
    highlighted_pdf, report = highlight_keywords_callouts_bytes(
        source_bytes,
        keywords=keywords,
        filename=source_path.name,
        include_review_mentions=include_review_mentions,
    )

    output_dir = Path(tempfile.mkdtemp(prefix="plan-keyword-review-"))
    output_pdf = output_dir / f"{source_path.stem}_keywords_highlighted.pdf"
    output_report = output_dir / f"{source_path.stem}_keyword_report.json"
    output_pdf.write_bytes(highlighted_pdf)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    gallery = _render_pages(highlighted_pdf, report, output_dir)

    return (
        _summary_html(report),
        gallery,
        _sheet_rows(report),
        _all_finding_rows(report),
        _all_excluded_rows(report),
        str(output_pdf),
        report,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Plan Material Callout Review") as demo:
        gr.Markdown(
            "# Plan Material Callout Review\n"
            "Upload a contract-plan PDF to identify configured material callouts, review uncertain mentions, "
            "and see which concrete references were intentionally excluded by the current Item 555 rules."
        )

        with gr.Row(equal_height=True):
            upload = gr.File(
                label="Upload contract-plan PDF",
                file_types=[".pdf"],
                type="filepath",
                scale=3,
            )
            include_review = gr.Checkbox(
                value=True,
                label="High-recall mode",
                info="Includes uncertain mentions so potential matches are not silently missed.",
                scale=1,
            )
            run_button = gr.Button("Analyze PDF", variant="primary", scale=1)

        summary = gr.HTML("<div class='review-note'>Upload a PDF to begin.</div>")

        plan_preview = gr.Gallery(
            label="Highlighted plan preview",
            columns=1,
            rows=1,
            height="auto",
            object_fit="contain",
            preview=True,
        )

        with gr.Tabs():
            with gr.Tab("Sheet Titles"):
                sheet_table = gr.Dataframe(
                    headers=["Page", "Tagged sheet title", "Confidence", "Detection method"],
                    datatype=["number", "str", "str", "str"],
                    label="Detected sheet titles",
                    interactive=False,
                    wrap=True,
                    max_height=720,
                    elem_classes=["full-width-table"],
                )

            with gr.Tab("All Flagged"):
                all_findings = gr.Dataframe(
                    headers=["Page", "Keyword", "Status", "Flagged text"],
                    datatype=["number", "str", "str", "str"],
                    label="Flagged callouts and mentions",
                    interactive=False,
                    wrap=True,
                    max_height=720,
                    elem_classes=["full-width-table"],
                )

            with gr.Tab("Not Flagged by Rules"):
                gr.Markdown(
                    "Concrete references listed here were found but intentionally not highlighted because "
                    "they matched Carol's non-Item-555 exclusion logic."
                )
                all_excluded = gr.Dataframe(
                    headers=["Page", "Exclusion rule", "Text not flagged"],
                    datatype=["number", "str", "str"],
                    label="Excluded non-555 concrete mentions",
                    interactive=False,
                    wrap=True,
                    max_height=720,
                    elem_classes=["full-width-table"],
                )

            with gr.Tab("Downloads & Raw Report"):
                highlighted_file = gr.File(label="Download highlighted PDF")
                report_json = gr.JSON(label="Full JSON report", open=False)

        run_button.click(
            fn=process_pdf,
            inputs=[upload, include_review],
            outputs=[
                summary,
                plan_preview,
                sheet_table,
                all_findings,
                all_excluded,
                highlighted_file,
                report_json,
            ],
        )

    return demo


app = build_app()

if __name__ == "__main__":
    app.launch(css=APP_CSS)
