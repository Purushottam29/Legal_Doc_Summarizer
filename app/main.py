from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.query import router as query_router
from app.routers.upload import router as upload_router
from app.routers.ask import router as ask_router
from app.routers.evaluate import router as evaluate_router

# CREATE THE FASTAPI APP
app = FastAPI(
    title="Legal Document Summarizer + RAG",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REGISTER ROUTES
app.include_router(upload_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(ask_router, prefix="/api")
app.inlcude_router(evaluate_router, prefix="/api")

# ROOT ROUTE
@app.get("/")
def home():
    return {"message": "Legal Doc Summarizer Running"}

