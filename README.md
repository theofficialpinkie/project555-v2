# Concrete Callout Review MVP

A small Gradio web app for contract-plan PDFs.

## Workflow

1. Upload a PDF.
2. The app detects callouts associated with leader/detail lines.
3. It filters those callouts to ones containing **concrete**.
4. It highlights the exact word **concrete** on the plan.
5. Review the flagged pages and callout list, then download the highlighted PDF.

## Run locally

```bash
cd contract_plan_concrete_mvp
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open the local URL printed in the terminal.

## Extractor CLI

```bash
python pdf_plan_extractor.py input.pdf \
  -o extracted_plan_data.json \
  --csv extracted_plan_data.csv \
  --highlight-keyword concrete \
  --highlight-dir highlighted
```

## Notes

- Best results come from text-based/vector PDFs.
- Raster-only scans need an OCR fallback.
- The highlighter intentionally ignores ordinary title-block/table text unless it is also detected as a leader-line callout.
