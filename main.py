"""
BestPhoto AI — Image Ranking API
Entry point: uvicorn main:app --host 0.0.0.0 --port 8000
"""
from src.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
