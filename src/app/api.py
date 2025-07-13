from fastapi import FastAPI

app = FastAPI()


@app.post(path="/inference/v1")
def inference():
    return 