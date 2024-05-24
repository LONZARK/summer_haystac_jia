import pydantic
import datetime
import json

from typing import Union, Any, List, Optional
from uuid import UUID
from shapely.geometry import mapping, Point, Polygon
from pydantic import validator, field_serializer
from enum import Enum

from haystac_kw.data.types.event import (
    validate_shapely_geometry,
    serialize_shapely_geometry,
    validate_shapely_point_poly)
from haystac_kw.data.types.time import TimeWindow, serialize_timedelta, validate_timedelta, validate_datetime
from haystac_kw.data import BaseModel


class TransportationMode(Enum):
    """Mode of transport to take"""
    personal_vehicle = "personal_vehicle"

    def __str__(self):
        return str(self.name.lower())

class Priority(Enum):
    """Indicates whether duration or end_time takes precedence"""
    duration = "duration"
    end_time = "end_time"

    def __str__(self):
        return str(self.name.lower())

class Start(BaseModel):
    """State agent must reach before other instructions can occur

    Parameters
    ----------
    instruction_uid : str or UUID
      Unique identifierfor the instruction
    location : Point or Polygon
      Geographic location to be reached
    time_window : TimeWindow
      time interval the location can be reached within
    """
    instruction_uid: Union[str, UUID]
    location: Any
    time_window: TimeWindow

    @validator("location")
    def validate_location_type(cls, l: Any):
        return validate_shapely_point_poly(l)

    @field_serializer("location")
    def serialize_location(self, value: Any,
                           info: pydantic.FieldSerializationInfo) -> str:
        if isinstance(value, Point) or isinstance(value, Polygon):
            return mapping(value)
        else:
            #print(f"serialize {type(value)} = {value}")
            return value

    @validator("time_window")
    def validate_time_window(cls, t: TimeWindow) -> TimeWindow:
        # Check the type
        tt = type(t)
        if tt is not TimeWindow:
            raise ValueError(f"location must be a TimeWindow, recieved {tt} instead")

        # Check if the time window is at least 1 hour
        duration = t.duration()
        hours = duration.total_seconds() / 3600

        assert hours >= 1, "The time window must be at least 1 hour"

        return t

class InternalStart(Start):
    """Kitware's internal Start format

    Parameters
    ----------
    location : str or UUID
          UUID of the location
    """
    location: Union[str, UUID]

    @validator("location")
    def validate_location_type(cls, l):
        tt = type(l)
        if tt not in [str, UUID]:
            raise ValueError(f"location must be a str or UUID, recieved {tt} instead")
        return l

    @field_serializer("location")
    def serialize_location(self, value: Any,
                           info: pydantic.FieldSerializationInfo) -> str:
        return str(value)

class Move(BaseModel):
    """Location agent must reach and mode of transport to take

    Parameters
    ----------
    instruction_uid : str or UUID
      Unique identifierfor the instruction
    location : Point or Polygon
      Geographic location to be reached
    transportation_mode : TransportationMode
      Mode of transport to take
    """
    instruction_uid: Union[str, UUID]
    location: Any
    transportation_mode: Union[TransportationMode, str]

    @validator("location")
    def validate_location_type(cls, l: Any):
        return validate_shapely_point_poly(l)

    @field_serializer("location")
    def serialize_location(self, value: Any,
                           info: pydantic.FieldSerializationInfo) -> str:
        if isinstance(value, Point) or isinstance(value, Polygon):
            return mapping(value)
        else:
            #print(f"serialize {type(value)} = {value}")
            return value

    @field_serializer("transportation_mode")
    def serialize_transportation_mode(self, value: TransportationMode,
                            info: pydantic.FieldSerializationInfo) -> str:
        return str(value)

    @validator('transportation_mode')
    def validate_transportation_mode_type(cls, value: Any) -> TransportationMode:
        if isinstance(value, TransportationMode):
            return value
        elif isinstance(value, str):
            return TransportationMode[value.lower()]
        raise pydantic.ValidationError

class InternalMove(Move):
    """Kitware's internal Move format

    Parameters
    ----------
    location : str or UUID
          UUID of the location
    """
    location: Union[str, UUID]

    @validator("location")
    def validate_location_type(cls, l):
        tt = type(l)
        if tt not in [str, UUID]:
            raise ValueError(f"location must be a str or UUID, recieved {tt} instead")
        return l

    @field_serializer("location")
    def serialize_location(self, value: Any,
                           info: pydantic.FieldSerializationInfo) -> str:
        return str(value)

class Stay(BaseModel):
    """Period of time to stay at current location

    ``duration``, ``end_time``, or both can be provided;
    if both are provided, ``priority`` should also be provided

    Parameters
    ----------
    instruction_uid : str or UUID
      Unique identifierfor the instruction
    duration : str or datetime.timedelta
      Length of the stay
    end_time : str or datetime.datetime
      Point in time when the stay should end
    priority : Priority
      Indicated whether ``duration`` or ``end_time`` takes precedence
    """
    instruction_uid: Union[str, UUID]
    duration: Optional[Union[str, datetime.timedelta]] = None
    end_time: Optional[Union[str, datetime.datetime]] = None
    priority: Optional[Priority] = None

    @field_serializer("priority")
    def serialize_priority(self, value: Priority,
                           info: pydantic.FieldSerializationInfo) -> str:
        return str(value)

    @field_serializer("duration")
    def serialize_timedelta_type(self, value: datetime.timedelta,
                            info: pydantic.FieldSerializationInfo) -> str:
        return serialize_timedelta(value)

    @validator('duration')
    def validate_timedelta_type(cls, value):
        return validate_timedelta(value)

    @field_serializer('end_time')
    def serialize_end_time(self, value: datetime.datetime):
        if isinstance(value, datetime.datetime):
            val = value.isoformat().replace('+00:00', 'Z')
            if "Z" not in val:
                val += "Z"
            return  val
        else:
            return value

    @classmethod
    @pydantic.field_validator('end_time')
    def validate_datetime_type(cls, value):
        return validate_datetime(value)

    @classmethod
    @pydantic.field_validator('priority')

    def validate_priority_type(cls, value: Any) -> Priority:
        if isinstance(Priority, str):
            return Priority[value.lower()]
        return value

    @pydantic.model_validator(mode='after')
    def validate(self) -> 'Stay':
        duration_provided = self.duration is not None
        end_time_provided = self.end_time is not None
        if duration_provided and end_time_provided:
          assert self.priority is not None, 'Priority not set'
        return self

class Itinerary(BaseModel):
    """All itineraries for the agent

    Parameters
    ----------
    itinerary : Start, Move, or Stay
      Movement instructions for sequential and
      continuous movement of the agent
    """
    itinerary: List[Union[Start, Move, Stay, dict]]

    @validator("itinerary")
    def validate_itinerary(cls, it: List[Union[Start, Move, Stay, dict]]) -> List[Union[Start, Move, Stay]]:
        # Check the type of it
        tt = type(it)
        if tt is not list:
            raise ValueError("An Itinerary must be a list, got {tt} instead")
        
        types = {
            "start": pydantic.TypeAdapter(Start),
            "move": pydantic.TypeAdapter(Move),
            "stay": pydantic.TypeAdapter(Stay)
        }
        for i in range(len(it)):
            if isinstance(it[i], dict):
                value = None
                for k,v in it[i].items():
                    clstype = types[k]
                    value = clstype.validate_json(json.dumps(v))

                it[i] = value
        
        # Make sure we have at least one item 
        assert it, "An Itinerary must have at least a Start object, received an empty list"
        
        # Make sure the first entry is a Start object
        start_t = type(it[0])
        if start_t is not Start:
            raise ValueError(f"The first element of an itinerary must be a Start object, got {start_t} instead")

        # Check that any remaining items are Move or Stay objects
        if len(it) > 1:
            for mv in it[1:]:
              mv_t = type(mv)
              if mv_t is Start:
                  raise ValueError(f"A start object can only be the first element of an itinerary")
              elif mv_t not in [Move, Stay]:
                  raise ValueError(f"Subsequent items in an itinerary must be Move or Stay, got {mv_t} instead")

        return it

    @field_serializer("itinerary")
    def serialize_itineraries(self, value: List[Union[Start, Move, Stay]],
                              info: pydantic.FieldSerializationInfo) -> List[dict]:
        it_json = []

        types = {
            Start: "start",
            InternalStart: "start",
            Move: "move",
            Stay: "stay"
        }
        for mv in value:
            mv_json = json.loads(mv.model_dump_json())
            it_json.append({types[type(mv)]: mv_json})
        return it_json

class InternalItinerary(Itinerary):
    """Kitware's internal Itinerary format

    Parameters
    ----------
    itinerary : InternalStart, InternalMove, or Stay
      Movement instructions for sequential and
      continuous movement of the agent
    """
    itinerary: List[Union[InternalStart, InternalMove, Stay, dict]]

    @validator("itinerary")
    def validate_itinerary(cls,
                           it: List[Union[InternalStart, InternalMove, Stay, dict]]) -> List[Union[InternalStart, InternalMove, Stay]]:
        # Check the type of it
        tt = type(it)
        if tt is not list:
            raise ValueError("An InternalItinerary must be a list, got {tt} instead")

        # Check the type of it
        tt = type(it)
        if tt is not list:
            raise ValueError("An Itinerary must be a list, got {tt} instead")
        
        types = {
            "start": pydantic.TypeAdapter(InternalStart),
            "move": pydantic.TypeAdapter(InternalMove),
            "stay": pydantic.TypeAdapter(Stay)
        }
        for i in range(len(it)):
            if isinstance(it[i], dict):
                value = None
                for k,v in it[i].items():
                    clstype = types[k]
                    value = clstype.validate_json(json.dumps(v))
                it[i] = value

        # Make sure we have at least one item 
        assert it, "An InternalItinerary must have at least an InternalStart object, received an empty list"
        
        # Make sure the first entry is a Start object
        start_t = type(it[0])
        if start_t is not InternalStart:
            raise ValueError(f"The first element of an InternalItinerary must be an InternalStart object, got {start_t} instead")

        # Check that any remaining items are Move or Stay objects
        if len(it) > 1:
            for mv in it[1:]:
              mv_t = type(mv)
              if mv_t is InternalStart:
                  raise ValueError(f"An InternalStart object can only be the first element of an InternalItinerary")
              elif mv_t not in [InternalMove, Stay]:
                  raise ValueError(f"Subsequent items in an InternalItinerary must be InternalMove or Stay, got {mv_t} instead")

        return it
    
    @field_serializer("itinerary")
    def serialize_itineraries(self, value: List[Union[InternalStart, InternalMove, Stay]],
                              info: pydantic.FieldSerializationInfo) -> List[dict]:
        it_json = []

        types = {
            InternalStart: "start",
            InternalMove: "move",
            Stay: "stay"
        }

        for mv in value:
            mv_json = json.loads(mv.model_dump_json())
            it_json.append({types[type(mv)]: mv_json})
        return it_json

class Movements(BaseModel):
    """Movement itineraries for an agent

    Parameters
    ----------
    agent : pydantic.PositiveInt
      Unique identifier for the agent
    itineraries: List[Itinerary]
      All itineraries for the agent
    """
    agent: pydantic.PositiveInt
    itineraries: List[Itinerary]

class InternalMovements(Movements):
    """Kitware's internal Movements format

    Parameters
    ----------
    itineraries: List[InternalItinerary]
        All itineraries for the agent
    """
    itineraries: List[InternalItinerary]
