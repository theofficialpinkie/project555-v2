from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import fitz
import gradio as gr

from pdf_plan_extractor import highlight_keyword_callouts_bytes


def _render_flagged_pages(pdf_bytes: bytes, report: dict[str, Any], output_dir: Path) -> list[tuple[str, str]]:
    """Render every page so unflagged or OCR-processed sheets never disappear."""
    report_by_page = {page["page"]: page for page in report["pages"]}
    gallery_items: list[tuple[str, str]] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(1.15, 1.15),
                alpha=False,
                annots=True,
            )
            image_path = output_dir / f"page-{page_number}.png"
            pixmap.save(image_path)

            page_report = report_by_page.get(page_number, {})
            flagged_count = len(page_report.get("flagged_callouts", []))
            source = page_report.get("text_source", "native")
            caption = f"Page {page_number} — {flagged_count} highlighted"
            if source == "ocr":
                caption += " — OCR used"
            elif source == "native_sparse":
                caption += " — OCR unavailable"
            gallery_items.append((str(image_path), caption))

    return gallery_items


def process_pdf(pdf_path: str | None, include_review_mentions: bool):
    if not pdf_path:
        return (
            "Upload a PDF to begin.",
            [],
            [],
            None,
            {},
        )

    source_path = Path(pdf_path)
    source_bytes = source_path.read_bytes()
    highlighted_pdf, report = highlight_keyword_callouts_bytes(
        source_bytes,
        keyword="concrete",
        filename=source_path.name,
        include_review_mentions=include_review_mentions,
    )

    output_dir = Path(tempfile.mkdtemp(prefix="concrete-callout-review-"))
    output_pdf = output_dir / f"{source_path.stem}_concrete_highlighted.pdf"
    output_report = output_dir / f"{source_path.stem}_concrete_report.json"
    output_pdf.write_bytes(highlighted_pdf)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows: list[list[Any]] = []
    for page in report["pages"]:
        sheet_title = page.get("sheet_title") or "Untitled sheet"
        for callout in page.get("flagged_callouts", []):
            status = callout.get("classification", "confirmed callout")
            if callout.get("match_quality") == "recovered one-character text error":
                status += " (PDF text repaired)"
            rows.append(
                [
                    page["page"],
                    sheet_title,
                    status,
                    callout["text"],
                ]
            )

    confirmed_total = report["total_flagged_callouts"]
    review_total = report.get("total_review_mentions", 0)
    recovered_total = report.get("total_recovered_matches", 0)
    highlighted_total = sum(
        len(page.get("flagged_callouts", [])) for page in report["pages"]
    )
    flagged_page_count = sum(
        1 for page in report["pages"] if page.get("flagged_callouts")
    )
    ocr_pages = report.get("ocr_pages", [])
    limited_pages = report.get("limited_pages", [])
    total_pages = report.get("total_pages", len(report["pages"]))
    pages_analyzed = report.get("pages_analyzed", total_pages)

    summary = (
        f"### Review complete\n"
        f"- **Pages analyzed:** {pages_analyzed} of {total_pages}\n"
        f"- **Pages requiring OCR:** {', '.join(map(str, ocr_pages)) if ocr_pages else 'None'}\n"
        f"- **Confirmed concrete callouts:** {confirmed_total}\n"
        f"- **Other concrete mentions found:** {review_total}\n"
        f"- **Recovered PDF text errors:** {recovered_total}\n"
        f"- **Mentions highlighted:** {highlighted_total}\n"
        f"- **Pages with highlights:** {flagged_page_count}\n\n"
        + (
            "High-recall mode is on, so uncertain mentions are highlighted and labeled "
            "**review mention**. This prevents silent misses, but may include notes or tables."
            if include_review_mentions
            else "Only confirmed leader-line callouts are highlighted. Other searchable "
            "mentions remain in the JSON report."
        )
    )
    if limited_pages:
        summary += (
            "\n\n⚠️ OCR could not run on pages "
            + ", ".join(map(str, limited_pages))
            + ". Install Tesseract (`brew install tesseract`) and restart the app."
        )

    gallery = _render_flagged_pages(highlighted_pdf, report, output_dir)
    return summary, gallery, rows, str(output_pdf), report


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Concrete Callout Review") as demo:
        gr.Markdown(
            "# Concrete Callout Review\n"
            "Upload a contract-plan PDF. The tool detects leader-line callouts, filters "
            "them to callouts containing **concrete**, and highlights the exact word on the plan. "
            "Flattened/image-only sheets are automatically OCRed."
        )

        upload = gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            type="filepath",
        )
        include_review = gr.Checkbox(
            value=True,
            label="High-recall mode: also highlight uncertain concrete mentions",
            info="Recommended while tuning. Review mentions may include notes or tables, but searchable occurrences will not be silently dropped.",
        )
        run_button = gr.Button("Review concrete callouts", variant="primary")

        summary = gr.Markdown("Upload a PDF to begin.")
        gallery = gr.Gallery(
            label="Highlighted plan preview",
            columns=1,
            rows=1,
            height="auto",
            object_fit="contain",
        )
        table = gr.Dataframe(
            headers=["Page", "Sheet title", "Status", "Flagged text"],
            datatype=["number", "str", "str", "str"],
            label="Flagged callouts",
            interactive=False,
            wrap=True,
        )
        highlighted_file = gr.File(label="Download highlighted PDF")
        report_json = gr.JSON(label="Review report", open=False)

        run_button.click(
            fn=process_pdf,
            inputs=[upload, include_review],
            outputs=[summary, gallery, table, highlighted_file, report_json],
        )

    return demo


app = build_app()

if __name__ == "__main__":
    app.launch()
