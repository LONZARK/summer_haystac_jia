import shapely

from enum import Enum
from collections import namedtuple
from datetime import timedelta
from typing import NamedTuple, List


class LonLat(NamedTuple):
    lon: float
    lat: float

class GraphLocation(NamedTuple):
    edge_id: str
    edge_dist: float

class POI(NamedTuple):
    edge_id: str
    pos: float
    building_centroid: shapely.Point
    building_poly: shapely.Polygon

class DesiredTrips(NamedTuple):
    locations: List[GraphLocation]
    durations: List[timedelta]

class LocationType(Enum):
    """Type of location"""
    STOP_POINT = 0
    ROAD_EDGE = 1
