from fastapi import FastAPI
import gradio as gr

from gradio_app import APP_CSS, app as gradio_blocks

fastapi_app = FastAPI()

app = gr.mount_gradio_app(
    fastapi_app,
    gradio_blocks,
    path="/",
    css=APP_CSS,
)
