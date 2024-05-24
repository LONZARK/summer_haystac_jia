import networkx as nx
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely import GeometryCollection

def extract_xgraph(sumo_net, edges):
    """Extract a networkx graph from a SUMO road
    network.
    
    SUMO models a road network as a collection of edges (roads)
    that meet at nodes (junctions). Each road is made up of sub
    components called lanes. When entering a junction the roads
    that the car can drive on to are determined by the lane that
    it is in.
    
    Since we would like to model the road network as a graph this
    function will convert the SUMO road network into a networkx
    DiGraph. To do this we first initialize a graph where every
    road is a directed edge and every junction is a node. However
    the connectivity of each junction may be complex and have an
    arbitrary set of rules. For instance if the junction disallows
    u-turns vs allowing u-turns. To handle this complexity we
    break up each junction into a node for each road adjacent to
    the junction and add directed edges between these nodes
    representing the true connectivity of the junction.
    
    We return the resulting DiGraph as well as mappings between
    the DiGraph edges and the edges in the SUMO network. The
    DiGraph edges are also populated with useful properties
    such as the length, speed limit, travel time, etc.
    
    See Here for information about how SUMO road network is
    structured:
    https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html

    Parameters
    ----------
    sumo_net : sumolib.Net
        SUMO road network object
    edges : sumolib.Edge
        List of edges in the road network

    Returns
    -------
    tuple(nx.DiGraph, dict, dict)
        DiGraph representing the road network, mapping from SUMO edges
        to graph edges, mapping from graph edges to SUMO edges
    """
    travel_time_graph = nx.DiGraph()

    # First build our directed graph

    # Get all the nodes (these are Junctions in SUMO)
    for node in sumo_net.getNodes():
        travel_time_graph.add_node(node.getID(),
                                   coord=node.getCoord())

    # Now gather all of the edges
    sumo_edge_to_nx_map = {}
    nx_edge_to_sumo_map = {}
    for edge in edges:
        # Travel time is length divided by speed.
        travel_time = edge.getLength() / edge.getSpeed()
        key1 = edge.getID()  # SUMO Edge ID
        key2 = (edge.getFromNode().getID(),  # (u,v)
                edge.getToNode().getID())
        # Add edge to the graph w/ metadata attributes
        travel_time_graph.add_edge(edge.getFromNode().getID(),
                                   edge.getToNode().getID(),
                                   weight=travel_time,
                                   sumo_id=key1,
                                   speed=edge.getSpeed(),
                                   length=edge.getLength(),
                                   id=key1
                                   )

        # Keep mapping between networkx and SUMO
        sumo_edge_to_nx_map[key1] = key2
        nx_edge_to_sumo_map[key2] = key1

    # SUMO collapes all junctions into a single node, however
    # each incomming road, does not necessarily have access to
    # each outgoing road, SUMO handles this using lane connections.
    # So we will iterate through the lane connections and modify
    # the graph to reflect the true connectivity.
    node_list = list(travel_time_graph.nodes.keys())
    for node in node_list:

        # Copy node attributes in case we split it
        node_attrs = travel_time_graph.nodes[node]

        # Inbound roads
        in_edges = list(travel_time_graph.in_edges([node]))
        # Outbound roads
        out_edges = list(travel_time_graph.out_edges([node]))

        if (len(in_edges) <= 1) and (len(out_edges) <= 1):
            # This is a dead end or junction with no complexity
            # just skip it
            continue

        # Split up the node into one node for each road in
        # the junction
        # TODO refactor this to handle as in_nodes and out_nodes so that
        # we can handle the case where a road is both and in-edge and an
        # out-edge for the junction (e.g. jughandle)
        new_nodes = {}  # Mapping from edges to new node
        for i, edge in enumerate(in_edges + out_edges):

            # Add new node with same attributes as original
            node_id = node + f'_KIT{i}'
            travel_time_graph.add_node(node_id, **node_attrs)
            new_nodes[edge] = node_id  # Keep track of edge connected

            # Get source, dest for new edge
            if edge[0] == node:
                # outbound road case
                nedge = (node_id, edge[1])
            else:
                # inbound road case
                nedge = (edge[0], node_id)
            # Add new edge with same attributes connected to the
            # split node
            travel_time_graph.add_edge(nedge[0], nedge[1],
                                       **travel_time_graph.edges[edge])
            nx_edge_to_sumo_map[nedge] = nx_edge_to_sumo_map[edge]

        # Now we add connections between all of the subnodes
        # that we created by splitting the junction node up
        for from_edge in in_edges:

            # Source node
            u = new_nodes[from_edge]

            # Get edge info from SUMO
            from_edge_id = nx_edge_to_sumo_map[from_edge]
            from_edge = sumo_net.getEdge(from_edge_id)

            for to_edge in out_edges:
                # Iterating through each outbound road and
                # checking if it is accessbile from from_edge

                # Dest node
                v = new_nodes[to_edge]
                to_edge_id = nx_edge_to_sumo_map[to_edge]

                # Get the connections between from_edge and to_edge
                # this returns a list, if it has any elements then
                # there is connectivity so we can add it as an edge
                conns = from_edge.getConnections(sumo_net.getEdge(to_edge_id))
                if len(conns) > 0:
                    # There is connectivity, add it as an edge with
                    # 0 travel time
                    travel_time_graph.add_edge(u, v, weight=0)

        # Delete all of the nodes and edges that we
        # split up / replaced.
        if len(new_nodes) > 0:
            # Deletes node and the old edges that
            # we replaced
            travel_time_graph.remove_node(node)
        for edge in in_edges:
            del nx_edge_to_sumo_map[edge]
        for edge in out_edges:
            del nx_edge_to_sumo_map[edge]

    sumo_edge_to_nx_map = {v: k for k, v in nx_edge_to_sumo_map.items()}

    return travel_time_graph, sumo_edge_to_nx_map, nx_edge_to_sumo_map


def map_buildings(building_gdf: gpd.GeoDataFrame,
                  road_network_gdf: gpd.GeoDataFrame,
                  maximum_distance: float = 50) -> gpd.GeoDataFrame:
    """Map building footprints to a edge on a road network
    and a length along the edge.

    Parameters
    ----------
    building_gdf : gpd.GeoDataFrame
        GeoDataFrame of building footprints
        (must have an `id` and `geometry` column)
    road_network_gdf : gpd.GeoDataFrame
        GeoDataFrame of the road network
        (must have an `id` and `geometry` column)
    maximum_distance : float, optional
        maximum tolerable distance (defaults to 50)

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with road mapping
    """

    # Convert to a cartesian coordinate system
    utm = building_gdf.estimate_utm_crs()
    aoi = GeometryCollection(building_gdf.geometry.values).convex_hull
    road_network_gdf = road_network_gdf[[x.intersects(aoi) for x in road_network_gdf.geometry.values]]
    road_network_gdf = road_network_gdf.to_crs(utm)
    building_gdf = building_gdf.to_crs(utm)

    # Spatially join the buildings and the roads
    pairs = building_gdf[['id',
                          'geometry']].sjoin_nearest(road_network_gdf[['id',
                                                                       'geometry']],
                                                     lsuffix='building',
                                                     rsuffix='road',
                                                     max_distance=maximum_distance)
    pairs = pairs.drop_duplicates(subset=['id_building'])

    # Reindex to speed up queries
    road_network_gdf = road_network_gdf.reindex()
    road_network_gdf = road_network_gdf.set_index('id')

    # Compute centroid of each building and project on the road
    # network. Save the edge and the length along the edge.
    centroids = []
    lengths = []
    for i in tqdm(range(len(pairs))):
        row = pairs.iloc[i]
        road_geom = road_network_gdf.loc[row.id_road].geometry
        centroid = row.geometry.centroid
        length_along_edge = road_geom.project(centroid)
        centroids.append(centroid)
        lengths.append(length_along_edge)
    pairs['geometry'] = centroids
    pairs['length_along_edge'] = lengths

    # Convert back to Lat/Long
    pairs = pairs.to_crs('EPSG:4326')
    building_gdf = building_gdf.to_crs('EPSG:4326')

    # merge to retain footprints
    building_gdf['id_building'] = building_gdf['id']
    poi = pairs.merge(building_gdf[['id_building', 'geometry']],
                      how='inner', on='id_building')
    return poi
