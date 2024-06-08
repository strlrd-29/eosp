from fastapi import FastAPI

from src.routers.chat import router as chat_router

app = FastAPI()

app.include_router(chat_router)
