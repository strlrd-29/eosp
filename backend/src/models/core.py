from typing import Any, Callable

from bson import ObjectId
from bson.errors import InvalidId
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import core_schema


class PyObjectId(ObjectId):
    """ObjectId as a pydantic model."""

    @staticmethod
    def validate(val: "str | bytes | PyObjectId | ObjectId") -> ObjectId:
        """Validate the ObjectId."""
        try:
            return ObjectId(val)
        except InvalidId as exc:
            raise ValueError("Invalid ObjectId") from exc

    @staticmethod
    def serialize(obj: "PyObjectId") -> str:
        """Serialize the ObjectId."""
        return f"{obj}"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """Define a custom core schema."""
        return core_schema.no_info_before_validator_function(
            PyObjectId.validate,
            schema=core_schema.union_schema(
                [
                    core_schema.str_schema(),
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.is_subclass_schema(ObjectId),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                function=PyObjectId.serialize,
                info_arg=False,
                return_schema=core_schema.str_schema(),
                when_used="json",
            ),
        )


class CoreModel(BaseModel):
    """Any common logic to be shared by all models goes here."""

    model_config = ConfigDict(populate_by_name=True)


class IDModelMixin(CoreModel):
    """Core model with _id field."""

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
