import pydantic
from pydantic import field_serializer
import datetime
import pandas
import numpy as np
from typing import Union
from haystac_kw.data import BaseModel
from dateutil import parser
import pytz

def serialize_timedelta(duration: datetime.timedelta) -> str:
    return pandas.Timedelta(duration).isoformat()

def validate_timedelta(value) -> datetime.timedelta:
    """
    Validates that an input is a timedelta or a
    ISO formatted timedelta string. If it is a formatted
    string then it converts it to a timedelta object.

    Parameters
    ----------
    value : Any
        Input value to be validated as timedelta.

    Returns
    -------
    datetime.timedelta
        Validated output

    Raises
    ------
    pydantic.ValidationError
        Input is not a timedelta or a valid timedelta string.
    """
    if isinstance(value, datetime.timedelta):
            return value
    elif isinstance(value, str):
        return datetime.timedelta(seconds=pandas.to_timedelta(value).seconds)
    elif value is None:
         return None
    raise pydantic.ValidationError

def validate_datetime(value) -> datetime.datetime:
    """
    Validates that an input is a datetime or a
    ISO formatted datetime string. If it is a formatted
    string then it converts it to a datetime object.

    Parameters
    ----------
    value : Any
        Input value to be validated as datetime.

    Returns
    -------
    datetime.datetime
        Validated output

    Raises
    ------
    pydantic.ValidationError
        Input is not a datetime or a valid datetime string.
    """
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=pytz.UTC)
        return value
    elif isinstance(value, str):
        if "Z" in value:
            value = value.replace("Z", "")
        value = parser.isoparse(value)
        if value.tzinfo is None:
            value = value.replace(tzinfo=pytz.UTC)
        return value
    elif value is None:
         return None
    raise pydantic.ValidationError


class TimeWindow(BaseModel):
    """Time interval for event to occur

    Parameters
    ----------
    begin : str or datetime.datetime
        beginning of the time window
    end : str or datetime.datetime
        end of the time window
    """
    model_config = pydantic.ConfigDict(strict=True)

    begin: Union[datetime.datetime, str]
    end: Union[datetime.datetime, str]

    def duration(self) -> datetime.timedelta:
        """Get the duration between ``begin`` and ``end``

        Returns
        -------
        datetime.timedelta
            Duration between ``begin`` and ``end``
        """
        self.end = validate_datetime(self.end)
        self.begin = validate_datetime(self.begin)
        return self.end - self.begin

    def sample(self) -> datetime.datetime:
        """Uniformly sample a time within the time window.
        """
        return self.begin + self.duration()*np.random.rand()

    @classmethod
    @pydantic.field_validator('begin','end')
    def validate_datetime_type(cls, value):
         return validate_datetime(value)

    @field_serializer('begin')
    def serialize_begin(self, value: datetime.datetime):
        if isinstance(value, datetime.datetime):
            val = value.isoformat().replace('+00:00', 'Z')
            if "Z" not in val:
                val += "Z"
            return  val
        else:
            return value

    @field_serializer('end')
    def serialize_end(self, value: datetime.datetime):
        if isinstance(value, datetime.datetime):
            val = value.isoformat().replace('+00:00', 'Z')
            if "Z" not in val:
                val += "Z"
            return  val
        else:
            return value


class DurationWindow(BaseModel):
    """Possible time intervals between events

    Parameters
    ----------
    minimum : str or datetime.timedelta
        minimum duration between events,
        in ISO duration standard format
    maximum : str or datetime.timedelta
        maximum duration between events,
        in ISO duration standard format
    """
    model_config = pydantic.ConfigDict(strict=True)

    minimum: Union[datetime.timedelta, str]
    maximum: Union[datetime.timedelta, str]

    @field_serializer("minimum", "maximum")
    def serialize_timedelta_type(self, value: datetime.timedelta,
                            info: pydantic.FieldSerializationInfo) -> str:
        return serialize_timedelta(value)

    @classmethod
    @pydantic.field_validator('minimum','maximum')
    def validate_timedelta_type(cls, value):
        return validate_timedelta(value)

