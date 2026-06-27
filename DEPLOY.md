# Put the app live

## Option A — Hugging Face Spaces (recommended)

1. Create a Hugging Face account.
2. Create a new Space.
3. Choose **Gradio** as the SDK and choose Public or Private visibility.
4. Clone the new Space repository.
5. Copy every file from this folder into that repository.
6. Commit and push.
7. Open the Space's **Build logs** and wait for the status to become Running.
8. Share the Space URL with your co-founder.

For a public Space with a simple password, add `APP_USERNAME` and `APP_PASSWORD` in **Space Settings → Variables and secrets**.

## Option B — Render

1. Push this folder to GitHub.
2. In Render, choose **New → Blueprint** or **New → Web Service**.
3. Connect the GitHub repository.
4. Render will detect `render.yaml` / `Dockerfile`.
5. Deploy and share the generated `.onrender.com` URL.

## Do not use Vercel for the PDF processor

The app accepts and returns potentially large PDFs, performs CPU-heavy PDF rendering/OCR, and requires the Tesseract system package. A container or Gradio Space is a better fit than a serverless function.
