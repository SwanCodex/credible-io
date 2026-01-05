from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Credible.io backend is running"}