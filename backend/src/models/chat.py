from enum import Enum
from typing import Any, Optional

from src.models.core import CoreModel, IDModelMixin


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(CoreModel):
    content: str
    role: Role


class Url(CoreModel):
    source: str
    title: Optional[str] = None
    description: Optional[str] = None


class Chat(IDModelMixin):
    name: Optional[str] = "New Chat"
    messages: list[dict[str, Any]] = []
    urls: list[Url] = []
    file: str
