from __future__ import annotations
"""
Schema for the Hide Objective Specification (HOS) standard
(https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Hide-Objective-Specification)
"""
import pydantic
import json

import geopandas as gpd
import pandas as pd

from typing import Tuple, List, Optional
from uuid import uuid4

from haystac_kw.data.types import event
from haystac_kw.data import BaseModel


class HOS(BaseModel):
    """Trial objective events and timing constraints

    Parameters
    ----------
    schema_version : str
        version number of the hide objective
        specification, this can be found in the schema file
    schema_type : str
        specifies the type of schema as HOS
    objective_uid : str
        unique identifier for the trial objective
    narrative : str
        human-readable version of the objective
    events : List[event.EventObjective]
        objects representing events
    time_constraints :  List[event.TimeConstraint]
        objects representing unary constraints on an event
    duration_constraints : List[event.DurationConstraint]
        objects representing binary constraints between a pair of events
    """
    model_config = pydantic.ConfigDict(strict=True)
    
    schema_version: Optional[str] = "1.1.2"
    schema_type: Optional[str] = "HOS"
    objective_uid: str
    narrative: str
    events: List[event.EventObjective]
    time_constraints: List[event.TimeConstraint]
    duration_constraints: List[event.DurationConstraint]

    def convert_to_internal(self, poi_table: gpd.GeoDataFrame, distance_threshold: float = 37.5) -> Tuple[InternalHOS, gpd.GeoDataFrame]:
        """
        Convert a HOS to an Internal HOS referencing
        Location UUID's rather than polygons, updates
        the table of location UUID's if they do not contain
        sufficient POI to cover all of the HOS polygons.
    

        Parameters
        ----------
        poi_table : gpd.GeoDataFrame
            Table of unique locations
        distance_threshold : float
            Maximum distance to a POI that a polygon can be and
            still be associated with it.

        Returns
        -------
        Tuple[InternalHOS, gpd.GeoDataFrame]
            HOS converted to Internal HOS and updated POI table.
        """
        

        # Grab POI geometries
        loc_id, geometry = [], []
        for i, hos_event in enumerate(self.events):
            loc_id.append(i)
            geometry.append(hos_event.location.centroid)
        building_gdf = gpd.GeoDataFrame({'id': loc_id, 'geometry': geometry},
                                        geometry='geometry',
                                        crs='EPSG:4326')

        # Convert to cartesian coordinate system
        original_crs = poi_table.crs
        utm = building_gdf.estimate_utm_crs()
        building_gdf = building_gdf.to_crs(utm)
        poi_table = poi_table.to_crs(utm)

        # Spatially join POI geometries with known stop points
        building_mapping = building_gdf.sjoin_nearest(
            poi_table[['point', 'global_stop_points']],
            max_distance=distance_threshold,
            how='left',
            distance_col='distance')
        while (len(building_mapping[['global_stop_points']].dropna()) < len(
                building_mapping)):
            # Here we have a case were there are POI geometries that aren't
            # within the distance threshold to be considered one of the known
            # stop points. While this is true we add the first unkown geometry's
            # stop point to the list of known stop points and run spatial join
            # again.

            # Grab first unkown POI geometry
            sub = building_mapping[building_mapping['global_stop_points'].isna()]
            row = sub.iloc[0]

            # Add its centroid to the list of known stop points
            poi_table = pd.concat([poi_table,
                                   gpd.GeoDataFrame({'point': [row.geometry],
                                                     'global_stop_points':[str(uuid4())]},
                                                    geometry='point',
                                                    crs=utm)],
                                  ignore_index=True)

            # Spatially join POI geometries with known stop points
            building_mapping = building_gdf.sjoin_nearest(
                poi_table, max_distance=37.5, how='left', distance_col='distance')

        # Replace geometries with edge/pos along edge in road network
        building_mapping = building_mapping.set_index('id')
        internal_hos_json = json.loads(self.model_dump_json())
        for i, hos_event in enumerate(internal_hos_json['events']):
            row = building_mapping.loc[i]
            hos_event['location'] = row.global_stop_points
        internal_hos = InternalHOS.from_json(json.dumps(internal_hos_json))
        poi_table = poi_table.to_crs(original_crs)

        return internal_hos, poi_table


class InternalHOS(HOS):
    """Kitware's internal HOS format

    Parameters
    ----------
    events : List[event.InternalEventObjective]
        objects representing events
    """
    events: List[event.InternalEventObjective]

