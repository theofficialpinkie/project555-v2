from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Concrete Callout Review API is running"
    }
