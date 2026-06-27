"""Compatibility entrypoint for platforms that expect app.py."""
from gradio_app import launch_app

if __name__ == "__main__":
    launch_app()
