---
title: Plan Material Callout Review
emoji: 📐
colorFrom: blue
colorTo: yellow
sdk: gradio
app_file: gradio_app.py
python_version: "3.12"
pinned: false
---

# Plan Material Callout Review — v19

Deployment-ready version of the contract-plan PDF review MVP.

## Recommended hosting: Hugging Face Spaces

This repository is ready for a **Gradio Space**. Hugging Face will:

- run `gradio_app.py`;
- install Python dependencies from `requirements.txt`;
- install Tesseract OCR from `packages.txt`;
- rebuild automatically after each push.

### Optional password

Set both of these Space secrets to require a login:

- `APP_USERNAME`
- `APP_PASSWORD`

If neither is set, the app is publicly accessible.

## Alternative hosting: Render

The included `Dockerfile` and `render.yaml` make the same repository deployable as a Render Docker web service.

## Local use

```bash
./.venv/bin/python3 gradio_app.py
```

The app listens on `0.0.0.0` and uses the `PORT` environment variable, defaulting to `7860`.

## Regression tests

```bash
./.venv/bin/python3 regression_test.py
```
