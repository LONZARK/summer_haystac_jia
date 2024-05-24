import datetime
import pydantic
import pydantic_core
import json

import pandas as pd

from pydantic import validator, field_serializer, field_validator
from typing import Union, List, Any
from uuid import UUID
from enum import Enum
from shapely.geometry import mapping, Point, Polygon
from shapely import GeometryCollection
import geopandas as gpd

from haystac_kw.data.types.time import validate_datetime
from haystac_kw.data.types import time
from haystac_kw.data.types.location import LocationType
from haystac_kw.data import BaseModel


class EventType(Enum):
    """An arrive occurs when a simulation agent goes from not
    being within a location to being within the location. A depart occurs when a simulation agent goes from being
    within a location to not being within the location.
    """
    ARRIVE = 0
    DEPART = 1

    def __str__(self):
        return str(self.name.lower())

def validate_shapely_geometry(l: GeometryCollection) -> GeometryCollection:
    """Validate the given location is a GeometryCollection
    consisting of Points and Polygons

    Parameters
    ----------
    l : GeometryCollection
      Geographic location

    Raises
    ------
    ValueError
      if ``location`` is not a GeometryCollection,
      if ``location`` does not contain Point or Polygon types

    Returns
    -------
    GeometryCollection
      Returns ``l`` unaltered
    """
    tt = type(l)
    if tt is dict:
        l = json.dumps(l)
        tt = type(l)
    if tt is str:
        l = gpd.read_file(l, driver='GeoJSON').geometry.values[0]
        tt = type(l)
    if tt is not GeometryCollection:
        raise ValueError(f"location must be a shapely.GeometryCollection, recieved {tt} instead")
    else:
        for geom in l.geoms:
            t_geom = type(geom)
            if t_geom not in [Point, Polygon]:
                raise ValueError(f"location may only consist of shapely.Points and/or shapely.Polygons, recieved {t_geom} instead")

    return l

def validate_shapely_point_poly(l: Any) -> Union[Point, Polygon]:
    """Validate the given location is a Point or Poly

    Parameters
    ----------
    l : Point or Polygon
      Geographic location

    Raises
    ------
    ValueError
      if ``location`` is not a Point or Polygon,

    Returns
    -------
    Union[Point, Polygon]
      Returns ``l`` unaltered
    """

    ol = l
    tt = type(l)
    if tt is dict:
        ol = l.copy()
        l = json.dumps(l)
        tt = type(l)
    if tt is str:
        l = gpd.read_file(l, driver='GeoJSON').geometry.values[0]
        tt = type(l)
    if tt not in [Point, Polygon]:
        print(l)
        raise ValueError(f"location must be a shapely.geometry.point.Point or Polygon, recieved {tt} instead")

    return ol

def serialize_shapely_geometry(value: GeometryCollection) -> dict:
    """Convert a GeometryCollection to a dict

    Parameters
    ----------
    value : GeometryCollection
      Geographic location

    Returns
    -------
    dict
      ``value`` as a dictionary
    """
    return mapping(value)


class EventObjective(BaseModel):
    """Instants in time where simulation agents must do something somewhere.

    This is used in the HOS format.

    Parameters
    ----------
    event_uid : str or UUID
        unique identifier for the event object
    agents : List[pydantic.PositiveInt] or pydantic.PositiveInt
        unique identifiers of the agents involved in the event
    event_type : str or EventType
        indicates the event type; possible options are arrive and depart .
        An arrive event happens when the agent goes from not being within location to being within
        location . A depart event happens when the agent goes from being within location to not
    location : shapely.GeometryCollection
        geographic location(s) where the event can occur; object restricted to
        GeoJSON GeometryCollection composed of Point and/or Polygon objects
    """
    model_config = pydantic.ConfigDict(strict=True)

    event_uid: Union[str, UUID]
    agents: List[pydantic.PositiveInt]
    event_type: Union[EventType, str]
    location: Any

    @field_serializer("event_type")
    def serialize_event_type(self, value: EventType,
                            info: pydantic.FieldSerializationInfo) -> str:
        return str(value)

    @validator("location")
    def validate_location_type(cls, l: Any):
        return validate_shapely_geometry(l)
    
    @field_serializer("location")
    def serialize_location(self, value: GeometryCollection,
                           info: pydantic.FieldSerializationInfo) -> str:
        return serialize_shapely_geometry(value)
    
    @classmethod
    @field_validator('event_type')
    def validate_event_type(cls, value: Any) -> EventType:
        return EventType[value.upper()]

class InternalEventObjective(EventObjective):
    """Kitware's internal EventObjective format 

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
    
class SimEvent(BaseModel):
    """An arrival or departure event from a stop point or a road segment
    that has already happened in the simulation.

    Parameters
    ----------
    timestamp : datetime.datetime or str
        unix epoch time when event occurred
    agent_id : pydantic.PositiveInt
        id of agent
    event_type : Union[EventType, str]
        arrival or departure
    location_type : Union[LocationType, str]
        stop_point or road_edge
    location_uuid : str or UUID
        UUID of Stop Point (if LocationType=stop_point)
        else uuid in the .shp file for if (LocationType=road_edge)
    """
    model_config = pydantic.ConfigDict(strict=True)

    timestamp: Union[datetime.datetime, str]
    agent_id: pydantic.PositiveInt
    event_type: Union[EventType, str]
    location_type: Union[LocationType, str]
    location_uuid: Union[str, UUID]

    @classmethod
    @pydantic.field_validator('timestamp')
    def validate_datetime_type(cls, value):
        return validate_datetime(value)

    @field_serializer("location_type")
    def serialize_location_type(self, value: Any,
                                info: pydantic.FieldSerializationInfo) -> str:
        return value.name

    @classmethod
    @field_validator('event_type')
    def validate_event_type(cls, value: Any) -> EventType:
        return EventType[value.upper()]

    @classmethod
    @field_validator('location_type')
    def validate_location_type(cls, value: Any) -> LocationType:
        return LocationType[value.upper()]

    @classmethod
    def from_pd_series(cls, event_row: pd.Series):
        """Initialize an event from a row in a pandas dataframe

        Parameters
        ----------
        event_row : pd.Series 
            Row of a pd.DataFrame

        Returns
        -------
        Event
            An agent event
        """
        return cls(
            timestamp=event_row["timestamp"], 
            agent_id=event_row["agent_id"], 
            event_type=EventType[event_row["EventType"].upper()],
            location_type=LocationType[event_row["LocationType"].upper()], 
            location_uuid=event_row["LocationUUID"]
        )

class EventConstraint(BaseModel):
    """Unique identifiers of the events the constraint applies to

    Parameters
    ----------
    first : str or UUID
        unique identifier for the event that occurs first
    second : str or UUID
        unique identifier for the event that occurs second
    """
    model_config = pydantic.ConfigDict(strict=True)

    first: Union[str, UUID]
    second: Union[str, UUID]

class TimeConstraint(BaseModel):
    """The time interval an event can occur within

    Parameters
    ----------
    event : str or UUID
        unique identifier of the event the constraint applies to
    time_window : time.TimeWindow
        time interval for event to occur
    """
    model_config = pydantic.ConfigDict(strict=True)

    event: Union[str, UUID]
    time_window: time.TimeWindow

class DurationConstraint(BaseModel):
    """Binary constraints between a pair of events

    Parameters
    ----------
    events : EventConstraint
        unique identifiers of the events the constraint applies to
    duration_window : time.DurationWindow
        possible time intervals between events
    stay : bool
        indicates whether the agent should remain within the
        location between events
    """
    model_config = pydantic.ConfigDict(strict=True)

    events: EventConstraint
    duration_window: time.DurationWindow
    stay: bool
