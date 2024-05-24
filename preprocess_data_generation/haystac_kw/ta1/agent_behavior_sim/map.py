#!/usr/bin/env python3

import numpy as np
import uuid
from tqdm import tqdm
import libsumo
import sumolib
import networkx as nx

try:
    from timezonefinder import TimezoneFinder
except ImportError:
    TimezoneFinder = None  # or handle it in another way

import pytz
import pandas as pd
from collections import namedtuple
from haystac_kw.utils.data_utils.road_network import extract_xgraph
from scipy.stats import rv_discrete
import pandana as pdna
import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.stats import gamma
from typing import Optional

POI = namedtuple("POI", "edge_id pos building_centroid building_poly")


class TravelTimeDistribution:
    """
    Class for sampling a distribution of travel times
    along a road network.
    """

    def __init__(
            self,
            k: float = .0025,
            theta: float = 80,
            x_shift: float = 1.0):
        """
        Constructor Method

        Parameters
        ----------
        k : float, optional
            Shape parameter for the Gamma distribution, by default .0025
        theta : float, optional
            Scale parameter for the Gamma distribution, by default 80
        x_shift : float, optional
            Horizontal shift of distribution on x-axis, by default 1.0
        """

        self.rv = gamma(k, scale=theta, loc=x_shift)

    def sample(self, min_travel_time: float, n_samples: int) -> np.ndarray:
        """
        Sample `n_samples` travel times from the distribution
        of travel times.

        Parameters
        ----------
        min_travel_time : float
            Fastest possible travel time
        n_samples : int
            Number of samples to return

        Returns
        -------
        np.ndarray
            Sampled travel times
        """
        return min_travel_time * self.rv.rvs(size=n_samples)


class Map:
    """Object representing the SUMO mobility network and points of interest."""

    def __init__(self, sumo_net, poi=None,
                 edge_allow_filter=['passenger'], save_graph=False):
        """Initialize map.

        Parameters
        ----------
        sumo_net : sumolib.net.Net or str
            Road network or filename from which to load SUMO net.
        :param poi : str, optional
            Path to pickle file containing dataframe of poi information
            (defaults to None)
        edge_allow_filter : list, optional
            List of string of the type of vehicles the road must support
            (defaults to ['passenger'])
        """
        if isinstance(sumo_net, str):
            sumo_net = sumolib.net.readNet(sumo_net)

        # xmin, ymin, xmax, ymax
        xmin, ymin, xmax, ymax = sumo_net.getBoundary()
        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2

        self.center_lon, self.center_lat = sumo_net.convertXY2LonLat(xc, yc)
        time_zone_str = TimezoneFinder().timezone_at(lng=self.center_lon,
                                                     lat=self.center_lat)
        self._timezone = pytz.timezone(time_zone_str)

        # Filter out edges that can't support cars.
        edges = []
        for edge in sumo_net.getEdges():
            if any([not edge.allows(x) for x in edge_allow_filter]):
                continue

            edges.append(edge)

        self._sumo_net = sumo_net

        ret = extract_xgraph(sumo_net, edges)
        travel_time_graph = ret[0]
        self._sumo_edge_to_nx_map = ret[1]
        self._nx_edge_to_sumo_map = ret[2]

        # Identify the largest subgraph.
        largest = max(nx.strongly_connected_components(
            travel_time_graph), key=len)

        # Remove nodes (and adjcanet edges) that fall outside of this subgraph.
        for node in set(travel_time_graph.nodes.keys()) - largest:
            travel_time_graph.remove_node(node)

        # Remove nodes that don't reference any edges.
        for node in nx.isolates(travel_time_graph):
            travel_time_graph.remove_node(node)

        self._nx_edges = list(travel_time_graph.edges.keys())
        self._sumo_edges = [self._nx_edge_to_sumo_map[edge]
                            for edge in self._nx_edges if edge in self._nx_edge_to_sumo_map]

        # node = list(travel_time_graph.nodes.keys())[0]
        # shortest_paths = nx.single_source_shortest_path(travel_time_graph, node)

        # Cache used for random location sampling.
        edge_sampling_weight = np.array([self.edge_length(edge)
                                         for edge in self.edges])
        edge_sampling_weight = edge_sampling_weight / sum(edge_sampling_weight)
        self.edge_sampler = rv_discrete(
            values=(range(len(self.edges)), edge_sampling_weight))

        self.poi = {}
        if poi is not None:
            self.parse_poi(poi)
        if save_graph:
            self.travel_time_graph = travel_time_graph

    @property
    def sumo_net(self):
        """Sumo network instance of type sumolib.net.Net.

        Returns
        -------
        sumolib.net.Net
            Sumo network instance
        """
        return self._sumo_net

    @property
    def gdf(self):
        """Return the road network as a GeoDataFrame

        Returns
        -------
        gpd.GeoDataFrame
            Road Network
        """

        geoms = [LineString([Point(*self._sumo_net.convertXY2LonLat(*x))
                            for x in y.getShape()])
                 for y in self._sumo_net.getEdges()]
        edge_ids = [x.getID() for x in self._sumo_net.getEdges()]
        u = [x.getFromNode().getID() for x in self._sumo_net.getEdges()]
        v = [x.getToNode().getID() for x in self._sumo_net.getEdges()]
        return gpd.GeoDataFrame({'u': u,
                                 'v': v,
                                 'id': edge_ids,
                                 'geometry': geoms},
                                geometry='geometry',
                                crs='EPSG:4326')

    @property
    def edges(self):
        """Sumo edges in the strongly-connected graph of the simulation road
        network as a list of str.

        Returns
        -------
        list(str)
            List of edge ID.
        """
        return self._sumo_edges

    @property
    def timezone(self):
        """Return time zone of this location.

        Returns
        -------
        pytz.timezone
            time zone of this location
        """
        return self._timezone

    def edge_length(self, edge):
        """Return length (meters) of road edge with provided str ID.

        Parameters
        ----------
        edge : str
            Edge ID.

        Returns
        -------
        float
            Length (meters) of road edge
        """
        return self.sumo_net.getEdge(edge).getLength()

    def edge_speed(self, edge):
        """Return speed (meters/s) of road edge with provided str ID.

        Parameters
        ----------
        edge : str
            Edge ID.

        Returns
        -------
        float
            Speed (meters/s) of road edge
        """
        return self.sumo_net.getEdge(edge).getSpeed()

    def get_random_location(self):
        """Get a random location in the SUMO road network.

        Returns
        -------
        tuple(string, float)
            - edge : Edge in road network
            - l : point along length of the edge
        """
        edge = self.edges[self.edge_sampler.rvs()]
        l = self.edge_length(edge) * np.random.rand(1)[0]
        return edge, l

    def parse_poi(self, poi_file):
        """Parse a pickle file containing a pandas dataframe of POI locations
        and parse them into a named tuple object with the following fields

        Parameters
        ----------
        poi_file : string
            pickle file with poi in simulation area
        """
        df = pd.read_pickle(poi_file)
        ids = df.id_building.values
        """
        id_road - SUMO edge id
        length_along_edge - length along edge closest to building centroid
        building_centroid - centroid of building footprint (shapley point)
        building_poly - building footprint (shapley polygon)
        """
        df = pd.read_pickle(poi_file)
        ids = df.id_building.values
        df = df[["id_road", "length_along_edge",
                 "building_centroid", "building_poly"]]
        for i, row in df.iterrows():
            self.poi[ids[i]] = POI(*row.values.flatten().tolist())

    def get_route(self, location1, location2):
        """Get route between two locations.

        Parameters
        ----------
        location1 : (str, float)
            Location within the road network encoded by (edge id,
            distance along edge).
        location2 : (str, float)
            Location within the road network encoded by (edge id,
            distance along edge).

        Returns
        -------
        libsumo.libsumo.TraCIStage
            Route between two locations
        """
        return libsumo.simulation.findRoute(location1[0], location2[0])


class OfflineMap(Map):

    def __init__(
            self,
            xgraph=None,
            sumo_net=None,
            poi=None,
            edge_allow_filter=['passenger'],
            precompute=150,
            travel_time_dist: TravelTimeDistribution = TravelTimeDistribution()):

        if xgraph is not None:
            self.travel_time_graph = xgraph
            self.parse_poi(poi)
        else:
            super(
                OfflineMap,
                self).__init__(
                sumo_net=sumo_net,
                poi=poi,
                edge_allow_filter=edge_allow_filter,
                save_graph=True)

        node_x = []
        node_y = []
        node_map = {}
        for node_id, node_attrs in self.travel_time_graph.nodes.items():
            node_map[node_id] = len(node_x)
            node_x.append(node_attrs['coord'][0])
            node_y.append(node_attrs['coord'][1])
        nodes = pd.DataFrame({'x': node_x, 'y': node_y})
        self.edge_map = {}
        eid, u, v, weight = [], [], [], []
        for edge_id, edge_attrs in self.travel_time_graph.edges.items():
            if 'id' in edge_attrs:
                self.edge_map[edge_attrs['id']] = node_map[edge_id[0]]
            u.append(node_map[edge_id[0]])
            v.append(node_map[edge_id[1]])
            weight.append(edge_attrs['weight'])
            if 'id' in edge_attrs:
                eid.append(edge_attrs['id'])
            else:
                eid.append(None)

        edges = pd.DataFrame(
            {'u': u, 'v': v, 'weight': weight, 'id_road': eid})
        self.travel_time_graph = pdna.Network(
            nodes['x'], nodes['y'], edges['u'], edges['v'], edges[['weight']], twoway=False)
        self.pandana_edges = edges
        self.travel_time_graph.precompute(precompute)

        self.poi = self.poi.loc[pd.Index(
            self.poi.id_road.astype(str)).isin(edges.id_road)]
        sorted_edges = edges.sort_values(['id_road']).dropna()
        road_index = pd.Index(sorted_edges.id_road)
        self.poi['node'] = [edges.u.values[road_index.get_loc(x)]
                            for x in tqdm(self.poi.id_road.values.astype(str))]
        self.poi = self.poi.set_index(self.poi.id_building)

        self.travel_time_dist = travel_time_dist

    def parse_poi(self, poi_file):
        """Loads a GeoDataFrame of poi from a pickle file

        Parameters
        ----------
        poi_file : str
            path to pickle file containing POI
        """
        poi = pd.read_pickle(poi_file)
        self.poi = gpd.GeoDataFrame(
            poi, geometry='building_centroid', crs='EPSG:4326')

    def get_route(self, location1, location2, edges = []):
        """Get route between two locations.

        Parameters
        ----------
        location1 : int
            poi id_building
        location2 : int
            poi id_building
        

        Returns
        -------
        list
            SUMO Road Network Edges
        """

        start_node = self.poi.loc[location1].node
        end_node = self.poi.loc[location2].node

        intermediate_nodes = [self.edge_map[x] for x in edges]
        nodes = [start_node] + intermediate_nodes + [end_node]

        full_route = []
        for i in range(len(nodes)-1):
            node_1 = nodes[i]
            node_2 = nodes[i+1]
            route = self.travel_time_graph.shortest_path(node_1, node_2)
            route = [self.pandana_edges.iloc[x].id_road for x in route]
            route = [x for x in route if x is not None]
            full_route += route
        return full_route

    def get_route_min_travel_time(self, location1, location2, edges = []):
        """Get travel time between two locations.

        Parameters
        ----------
        location1 : int
            poi id_building
        location2 : int
            poi id_building
        edges: list
            edges to force route along

        Returns
        -------
        float
            Travel time between locations in seconds
        """

        start_node = self.poi.loc[location1].node
        end_node = self.poi.loc[location2].node

        intermediate_nodes = [self.edge_map[x] for x in edges]
        nodes = [start_node] + intermediate_nodes + [end_node]

        src = nodes[:-1]
        dst = nodes[1:]
        travel_time = np.sum(self.travel_time_graph.shortest_path_lengths(src, dst))
        
        return travel_time

    def sample_travel_times(
            self,
            location1: int,
            location2: int,
            edges: list,
            n_samples: int) -> np.ndarray:
        """
        Sample `n_samples` travel times from a distribution of travel times
        between **location1** and **location2**

        Parameters
        ----------
        location1 : int
            Starting location poi id_building
        location2 : int
            Ending locaiton poi id_building
        edges: list
            edges to force route along
        n_samples : int
            Number of travel times to sample

        Returns
        -------
        np.ndarray
            Array of sampled travel times in seconds
        """

        min_travel_time = self.get_route_min_travel_time(location1, location2, edges=edges)
        return self.travel_time_dist.sample(min_travel_time, n_samples)

    def get_random_location(self):
        """Select a random poi

        Returns
        -------
        int
            Random POI index
        """
        return np.random.choice(self.poi.index, 1)[0]

    def get_random_locations(self, N):
        """Get N random locations

        Parameters
        ----------
        N : int
            Number of locations to select

        Returns
        -------
        list
            List of POI indices
        """
        return np.random.choice(self.poi.index, N)

    def get_routes(self, locations1, locations2):
        """Get routes between two lists of locations.

        Parameters
        ----------
        locations1 : list
            list of source poi id_building
        locations2 : list
            list of destination poi id_building

        Returns
        -------
        list
            Routes between ``locations1`` and ``locations2``
        """

        node_1 = [self.poi.loc[x].node for x in locations1]
        node_2 = [self.poi.loc[x].node for x in locations2]

        route = self.travel_time_graph.shortest_paths(node_1, node_2)
        route = [[self.pandana_edges.iloc[y].id_road for y in x]
                 for x in route]
        route = [[y for y in x if y is not None] for x in route]
        return route

    @property
    def gdf(self):
        return None
