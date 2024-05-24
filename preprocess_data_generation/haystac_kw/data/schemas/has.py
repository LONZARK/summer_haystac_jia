from __future__ import annotations
"""
Schema for the Hide Activity Specification (HAS) standard
(https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Hide-Activity-Specification)
"""
import pydantic
import json

import geopandas as gpd
import pandas as pd

from typing import Union, List, Iterator, Tuple, Optional
from uuid import UUID

from shapely import GeometryCollection
import json
from shapely.geometry import mapping, Point, Polygon
from haystac_kw.utils.data_utils.road_network import map_buildings

from haystac_kw.data import BaseModel
from haystac_kw.data.types.itinerary import Movements, InternalMovements, Stay
from haystac_kw.data.schemas.hos import HOS, InternalHOS


class HAS(BaseModel):
    """Movement instructions for meeting a trial objective

    Parameters
    ----------
    schema_version : str
        version number of the hide activity
        specification, this can be found in the schema file
    schema_type : str
        specifies the type of schema is HAS
    objective : str or UUID
      Unique identifier for the corresponding trial objective
    movements : List[Movements]
      Movement itineraries for an agent
    """
    schema_version: Optional[str] ="1.1.2"
    schema_type: Optional[str] = "HAS"
    objective: Union[str, UUID]
    movements: List[Movements]

    def convert_to_sumo(self, road_network: gpd.GeoDataFrame) -> str:
        """
        Convert a HAS to a json string referencing the road
        network rather than polygons for SUMO

        Parameters
        ----------
        road_network : gpd.GeoDataFrame
            Road network to reference to

        Returns
        -------
        str
            Converted JSON string
        """

        # Get list of geometries and convert to a GeoDataFrame
        location_ids = []
        locations = []
        for location_id, location in self.iter_locations():
            location_ids.append(location_id)
            locations.append(location.centroid)
        gdf = gpd.GeoDataFrame({'id': location_ids,
                                'geometry': locations},
                               geometry='geometry',
                               crs='EPSG:4326')

        # Map GeoDataFrame to the road network
        location_mapping = map_buildings(
            gdf, road_network, maximum_distance=100)
        assert (len(location_mapping) == len(gdf))
        # Replace locations with the mapping
        has_json = json.loads(self.model_dump_json())
        cnt = 0
        for mvmt_id, movement in enumerate(self.movements):
            has_movement = has_json['movements'][mvmt_id]
            for itr_id, itinerary in enumerate(movement.itineraries):
                has_itinerary = has_movement['itineraries'][itr_id]['itinerary']
                for instr_id, instruction in enumerate(itinerary.itinerary):
                    # Convert each instruction location field
                    # to a road_edge, length along edge

                    if isinstance(instruction, Stay):
                        # The stay instruction doesn't have
                        # any location associated with it
                        # so do nothing
                        continue

                    # At this point we are at an instruction
                    # that actually needs the location swapped
                    # out

                    # Get pointer to data for the instruction
                    # {'start'/'move: {**data}} -> {**data}
                    has_instruction = has_itinerary[instr_id]
                    has_instruction = has_instruction[next(
                        iter(has_instruction.keys()))]

                    # Replace location field
                    row = location_mapping.iloc[cnt]
                    cnt += 1
                    has_instruction['location'] = (
                        row.id_road, row.length_along_edge)

        return json.dumps(has_json)

    def iter_locations(self) -> Iterator[Tuple[int, Union[Point, Polygon]]]:
        """
        Return an iterator through all of the locations in the HOS

        Yields
        ------
        Iterator[Tuple[int, Union[Point, Polygon]]]
            Tuple of location Id and Location
        """
        i = 0
        for movement in self.movements:
            for itinerary in movement.itineraries:
                for instruction in itinerary.itinerary:
                    if isinstance(instruction, Stay):
                        continue
                    i_ = i
                    i += 1
                    yield i_, instruction.location


class InternalHAS(HAS):
    """Kitware's internal HAS format

    Parameters
    ----------
    movements : List[InternalMovements]
        Movement itineraries for an agent
    """
    movements: List[InternalMovements]

    def convert_to_external(
            self,
            hos: HOS,
            ihos: InternalHOS,
            stop_point_table: gpd.GeoDataFrame,
            road_network: gpd.GeoDataFrame) -> HAS:
        """
        Convert a InternalHAS into a HAS referencing
        polygons rather than Location and Road ID's.

        The conversion policy is to replace the road ID's with the
        full road geometry and replacing the stop point ID's with the
        polygons specified in the HOS.

        Parameters
        ----------
        hos : HOS
            Original HOS referencing polygons
        road_network : gpd.GeoDataFrame
            Road network

        Returns
        -------
        HAS
            Converted HAS
        """

        # First build a mapping between the stop points and
        # the polygons.
        stop_point_mapping = {}
        for hos_event in hos.events:
            for ihos_event in ihos.events:
                if ihos_event.event_uid == hos_event.event_uid:
                    stop_point_mapping[ihos_event.location] = hos_event.location
                    break

        # Grab polygons for all of the other POI
        for _, row in stop_point_table.iterrows():
            if row.LocationUUID not in stop_point_mapping:
                stop_point_mapping[row.LocationUUID] = row.geometry

        # Iterate through and convert each location uid into a geometry
        ihas_json = json.loads(self.model_dump_json())
        for mvmt_id, movement in enumerate(self.movements):
            has_movement = ihas_json['movements'][mvmt_id]
            for itr_id, itinerary in enumerate(movement.itineraries):
                has_itinerary = has_movement['itineraries'][itr_id]['itinerary']
                for instr_id, instruction in enumerate(itinerary.itinerary):
                    # Convert each instruction location field
                    # to a polygon

                    if isinstance(instruction, Stay):
                        # The stay instruction doesn't have
                        # any location associated with it
                        # so do nothing
                        continue

                    # At this point we are at an instruction
                    # that actually needs the location swapped
                    # out

                    # Get pointer to data for the instruction
                    # {'start'/'move: {**data}} -> {**data}
                    has_instruction = has_itinerary[instr_id]
                    has_instruction = has_instruction[next(
                        iter(has_instruction.keys()))]

                    # Replace location field
                    if instruction.location in stop_point_mapping:
                        # The location is a stop points
                        geom = stop_point_mapping[instruction.location]
                    else:
                        # The location is a road segment
                        geom = road_network[road_network.id ==
                                            instruction.location].geometry.values[0]

                    #geom = GeometryCollection(geom)
                    # removing GC since the spec says:
                    # location (object/geojson): geographic location to be reached;
                    # object restricted to GeoJSON Point or Polygon

                    # some items come in as GeometryCollections already
                    # replace GC with its contents
                    if type(geom) is GeometryCollection:
                        # TODO: possibly just do the convex_hull operation?
                        # it's not clear where these collections are coming from
                        # and if they maybe need full geometry preserved?
                        inner = None
                        for geo in geom.geoms:
                            t_geom = type(geo)
                            if t_geom in [Point, Polygon]:
                                print(f"extracting point/polygon")
                                inner = geo
                        if inner is not None:
                            geom = inner
                        else:
                            print("Warning: Geometry collection has no Point or Polygon (running convex_hull)")
                            geom = geom.convex_hull.exterior  # get a poly from it
                    #print(f"geom = {type(geom)}")

                    # Update HAS dict
                    has_instruction['location'] = mapping(geom)  # converts to dict
                    #print(f"location type = {type(has_instruction['location'])}")

        # Create HAS object from HAS dict
        has = HAS.from_json(json.dumps(ihas_json))
        return has

    def drop_road_edge_instructions(
            self,
            stop_point_table: pd.DataFrame) -> InternalHAS:
        """
        In place remove all instructions with location that is not
        in the stop point table (all road segment instructions).

        Parameters
        ----------
        stop_point_table : pd.DataFrame
            All stop points in the simulation

        Returns
        -------
        InternalHAS
            HAS with all instructions referencing road network locations
            removed.
        """

        # Iterate through and convert each location uid into a geometry
        for movement in self.movements:
            for itinerary in movement.itineraries:
                for instr_id in range(len(itinerary.itinerary)-1,-1,-1):
                    instruction = itinerary.itinerary[instr_id]
                    if isinstance(instruction, Stay):
                        # The stay instruction doesn't have
                        # any location associated with it
                        # so do nothing
                        continue

                    # At this point we are at an instruction
                    # that actually has a location

                    # Replace location field
                    if instruction.location in stop_point_table.LocationUUID.values:
                        continue
                    del itinerary.itinerary[instr_id]

        # Create HAS object from HAS dict
        return self


if __name__ == "__main__":
    from pathlib import Path

    ihas_file = Path('/home/local/KHQ/cole.hill/Desktop/trial_objective_and_train_data_context_has/hos_12f5023f-221a-42ac-8b0d-239a9eb11c65.json')
    stop_point_table = pd.read_parquet('/home/local/KHQ/cole.hill/Desktop/kitware/stop_points/StopPoints.parquet')

    ihas = InternalHAS.from_json(ihas_file.read_text())
    ihas.drop_road_edge_instructions(stop_point_table)
    print(ihas)
