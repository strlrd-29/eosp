import asyncio
from typing import Annotated

import chromadb
from chromadb.config import Settings
from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorClient

from src.config import settings

client = chromadb.PersistentClient(settings=Settings(allow_reset=True))


async def init_db_client():
    """Initialize async motor client for mongodb.

    Returns
    -------
        mongo_client (AsyncIOMotorClient): Async motor client from mongodb.

    """
    mongo_client = AsyncIOMotorClient(
        settings.DB_URL,
        port=settings.DB_PORT,
        io_loop=asyncio.get_running_loop(),
        uuidRepresentation="standard",
    )

    return mongo_client


async def init_db(client: Annotated[AsyncIOMotorClient, Depends(init_db_client)]):
    """Initialize hyko database, creates collections and indexes.

    Returns
    -------
        (AsyncIOMotorDatabase): Async motor client from mongodb.

    """
    db_name = settings.DB_NAME
    try:
        yield client[db_name]
    except Exception:
        client.close()
