from pydantic import BaseModel as PydanticBaseModel
from pydantic import TypeAdapter


class BaseModel(PydanticBaseModel):
    """
    Subclassing Pydantic BaseModel so that it
    is easy to modify global behavior within
    the repo.
    """
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @classmethod
    def from_json(cls, json_str: str):
        """
        Load a Pydantic Class from a JSON string

        Parameters
        ----------
        json_str : str
            JSON encoding of the class

        Returns
        -------
        BaseModel
            Decoded Class from JSON
        """
        return TypeAdapter(cls).validate_json(json_str)

    # Changing default behavior of model_dump_json to not output
    # empty fields to the JSON
    def model_dump_json(
            self,
            *,
            indent=None,
            include=None,
            exclude=None,
            by_alias=False,
            exclude_unset=False,
            exclude_defaults=False,
            exclude_none=True,
            round_trip=False,
            warnings=True) -> str:
        return super().model_dump_json(
            indent=indent,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings)
