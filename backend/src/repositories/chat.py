from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from src.models.chat import Chat
from src.models.core import PyObjectId


class ChatRepository:
    def __init__(self, database: AsyncIOMotorDatabase):
        """Init the chat repository"""
        self.collection: AsyncIOMotorCollection = database.get_collection("chats")

    async def find(self):  # type: ignore
        return await self.collection.find().to_list(None)  # type: ignore

    async def find_one(self, id: str):  # type: ignore
        return await self.collection.find_one({"_id": PyObjectId(id)})  # type: ignore

    async def create(self, chat: Chat):
        res = await self.collection.insert_one(chat.model_dump(exclude_unset=True))  # type: ignore
        return res.inserted_id

    async def update(self, id: PyObjectId, chat: Chat):
        await self.collection.update_one(
            {"_id": id}, {"$set": {**chat.model_dump(exclude={"id"})}}
        )

    async def add_message(self, chat_id: PyObjectId, message: dict[str, Any]):
        docs = message["documents"]
        message.pop("documents")
        message["documents"] = [doc.dict() for doc in docs]
        message["id"] = PyObjectId().__str__()
        await self.collection.update_one(
            {"_id": chat_id}, {"$push": {"messages": message}}
        )
