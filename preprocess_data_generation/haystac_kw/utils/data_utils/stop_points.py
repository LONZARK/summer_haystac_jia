import numpy as np
import numba
from numba import int32, float64
from numba.typed import List
from scipy.ndimage import binary_dilation
import pandas as pd
import geopandas
import geopandas as gpd
from shapely import GeometryCollection
from collections import defaultdict
from uuid import uuid4


def brute_force(df, time_window=120, distance_threshold=75):
    """Brute force method of calculating stopping points defined
    as the agent points being in a `distance_threshold` radius
    of their centroid in at least a `time_window` seconds
    timeframe.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        GeoDataFrame with a `timestamp_unix` (epoch time) column
        and a `points` column containing shapley Points in a carteisan
        coordinate system
    time_window : int, optional
        minimum time window in seconds (defaults to 120)
    distance_threshold : int, optional
        maximum distance points in a valid window can
        be from their centroid in meters (defaults to 75)

    Returns
    -------
    list
        Computed stop points
    """
    time_start_idx = 0  # Index of start time of potential stop-period
    time_stop_idx = 1  # Index of stop time of potential stop-period
    df = df.sort_values(['timestamp_unix'])
    times = df.timestamp_unix.values  # Timestamps
    points = df.geometry.values  # GPS Points in Euclidean Coordinates
    stop_points = []  # Array to store time indices for valid stop-periods
    coords = []
    coords = np.vstack([[x.x for x in points], [x.y for x in points]]).T

    # Function to return maximum distance of a set of points x
    # to another point y
    def max_dist(x, y): return np.max(np.linalg.norm(x - y[None, :], axis=1))

    # Find time_stop so that we have at least time_window seconds
    while (times[time_stop_idx - 1] - times[time_start_idx]) < time_window:
        time_stop_idx += 1
        if time_stop_idx > len(times):
            return stop_points

    # We have not yet varified if this window
    # is valid
    valid_window = None

    # Slide window until we are out of time
    while True:
        # At this point time_start_idx:time_stop_idx covers at least
        # time_window seconds
        #
        # valid_window is either None if the previous version of
        # start_idx:stop_idx did not constitute a valid window
        # or will be equal to (time_start_idx,time_stop_idx)
        # for the previous best valid window

        if (time_stop_idx > len(times)) or (time_start_idx >= len(times)):
            # Reached end of sequence
            if valid_window:
                stop_points.append(valid_window)
            break

        print(f'{time_stop_idx/len(times):.3f}', end='\r')
        # Calculate centroid with new points
        centroid = np.mean(coords[time_start_idx:time_stop_idx], axis=0)
        if max_dist(coords[time_start_idx:time_stop_idx],
                    centroid) < distance_threshold:
            # time_start_idx:time_stop_idx encodes a valid window, but
            # not necesarily the largest one
            valid_window = time_start_idx, time_stop_idx - 1
            time_stop_idx += 1
            continue
        elif valid_window:
            # Previous valid window breaks when adding
            # time_stop_idx-1, so we save valid window
            # and restart.
            stop_points.append(valid_window)
            time_start_idx = time_stop_idx - 1
            valid_window = None
        else:
            # time_start_idx:time_stop_idx did not yield a
            # valid window so we will increment time_start_idx
            # to shift window right
            time_start_idx += 1

        # Increment time_stop_idx until time_start_idx:time_stop_idx is
        # at least time_window seconds
        if time_start_idx >= len(times):
            break
        while (times[time_stop_idx - 1] - times[time_start_idx]) < time_window:
            if time_stop_idx >= len(times):
                break
            time_stop_idx += 1
    return stop_points


spec = [
    ('N', int32),
    ('centroid', float64[:]),
]


@numba.experimental.jitclass(spec)
class RollingCentroid(object):
    """Datastructure for maintaining a weighted
    rolling centroid
    """

    def __init__(self, N, average):
        """
        Constructor Method

        Parameters
        ----------
        N : int
            Number of observations
        average : list
            Rolling centroid
        """
        self.N = N  # Current Number of Observations
        self.centroid = average  # Current centroid

    def add_value(self, x, w):
        """Add value x to centroid with weight w

        Parameters
        ----------
        x : list
            Coordinate
        w : float
            Weight
        """
        self.centroid = self.centroid * self.N / \
            (self.N + w) + w * x / (self.N + w)
        self.N = self.N + w

    def remove_value(self, x, w):
        """Remove value x from centroid with weight w

        Parameters
        ----------
        x : list
            Coordinate
        w : float
            Weight
        """
        self.centroid = (self.centroid * self.N - w * x) / \
            (self.N - w)  # - x/(self.N-1)
        self.N = self.N - w

    def refresh(self, N, x):
        """Re-initialize centroid

        Parameters
        ----------
        N : int
            Number of observations
        x : list
            Coordinate
        """
        self.N = N
        self.centroid = x


@numba.njit(nogil=True)
def max_dist(x, y):
    """Calculate the distance between a set of points
    and one other point.

    Parameters
    ----------
    x : np.ndarray
        First set of points
    y : np.ndarray
        Second set of points

    Returns
    -------
    float
        Maximum point-wise distance
    """
    diff = x - y
    return np.max(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1])


@numba.njit(nogil=True)
def calculate_stop_points_indices_compiled(times, w, coords,
                                           time_window, distance_threshold):
    """Method for calculating stop points from a collection of gps trajectories

    Parameters
    ----------
    times : list
        array representing the end time of each gps observation (unix epoch)
    w : list
        duration of each gps observation
    coords : list
        cartesian coordinates of the gps observations
    time_window : float
        minimum time frame for a stop
    distance_threshold : float
        maximum distance for a stop

    Returns
    -------
    list
        List of of tuples describing start_time,stop_time of each stop point
    """
    time_start_idx = 0  # Index of start time of potential stop-period
    time_stop_idx = 1  # Index of stop time of potential stop-period
    stop_points = List()  # Array to store time indices for valid stop-periods

    # Initialize data structure for rolling centroid
    centroid = RollingCentroid(1, coords[0])
    start_delta = 0

    # Find time_stop so that we have at least time_window seconds
    while (times[time_stop_idx - 1] - times[time_start_idx]) < time_window:
        centroid.add_value(coords[time_stop_idx - 1], 1)
        time_stop_idx += 1
        if time_stop_idx > len(times):
            return stop_points

    # We have not yet varified if this window
    # is valid
    valid_window = [-1, -1, -1, -1]

    # Slide window until we are out of time
    while True:
        # At this point time_start_idx:time_stop_idx covers at least
        # time_window seconds
        #
        # valid_window is either -1,-1,-1,-1 if the previous version of
        # start_idx:stop_idx did not constitute a valid window
        # or will be equal to
        # (time_start_idx,time_stop_idx, start_delta, stop_delta)
        # for the previous best valid window

        if time_stop_idx > len(times):
            # Reached end of sequence
            if valid_window[0] != -1:
                stop_points.append(valid_window[0])
                stop_points.append(valid_window[1])
                stop_points.append(valid_window[2])
                stop_points.append(valid_window[3])
            break

        # Calculate centroid with new points
        if max_dist(coords[time_start_idx:time_stop_idx],
                    centroid.centroid) < distance_threshold:
            # time_start_idx:time_stop_idx encodes a valid window, but
            # not necesarily the largest one
            if time_stop_idx < len(times):
                centroid.add_value(coords[time_stop_idx], w[time_stop_idx])
            valid_window = [time_start_idx, time_stop_idx - 1, start_delta, 0]
            time_stop_idx += 1
            continue
        elif valid_window[0] > -1:
            # Previous valid window breaks when adding
            # time_stop_idx-1, so we save valid window
            # and restart.
            start_delta = 0
            if w[time_stop_idx - 1] > 1:
                # It is possible that the time window could be extended until
                # this observation shifts the centroid too much. Add 1s of this
                # observation until the window becomes invalid
                centroid.remove_value(
                    coords[time_stop_idx - 1], w[time_stop_idx - 1])
                for i in range(w[time_stop_idx - 1]):
                    centroid.add_value(coords[time_stop_idx - 1], 1)
                    if max_dist(coords[time_start_idx:time_stop_idx],
                                centroid.centroid) >= distance_threshold:
                        # Window is invalid save, how much of the observation
                        # the previous
                        # window can tolerate
                        start_delta = -w[time_stop_idx - 1] + i + 1
                        valid_window[-1] = i
                        break

            # Save window
            stop_points.append(valid_window[0])
            stop_points.append(valid_window[1])
            stop_points.append(valid_window[2])
            stop_points.append(valid_window[3])
            # Update centroid with the remaining portion of the last
            # observation
            centroid.refresh(abs(start_delta) if start_delta !=
                             0 else w[time_stop_idx - 1],
                             coords[time_stop_idx - 1])
            time_start_idx = time_stop_idx - 1
            valid_window = [-1, -1, -1, -1]
        else:
            # time_start_idx:time_stop_idx did not yield a
            # valid window so we will increment time_start_idx
            # to shift window right
            if start_delta > 0:
                centroid.remove_value(coords[time_stop_idx - 1], start_delta)
            start_delta = 0
            centroid.remove_value(coords[time_start_idx], w[time_start_idx])
            time_start_idx += 1

        # Increment time_stop_idx until time_start_idx:time_stop_idx is
        # at least time_window seconds
        while (times[time_stop_idx - 1] - times[time_start_idx]) < time_window:
            if time_stop_idx >= len(times):
                break
            centroid.add_value(coords[time_stop_idx], w[time_stop_idx])
            time_stop_idx += 1

    return stop_points


def calculate_stop_points_indices(df, time_window=120, distance_threshold=75):
    """Brute force method of calculating stopping points defined
    as the agent points being in a `distance_threshold` radius
    of their centroid in at least a `time_window` seconds
    timeframe

    Parameters
    ----------
    df : pd.GeoDataFrame
        GeoDataFrame with a `timestamp_unix` (epoch time) column
        and a `points` column containing shapley Points in a carteisan
        coordinate system
    time_window : int, optional
        minimum time window in seconds, defaults to 120
    distance_threshold : int, optional
        maximum distance points in a valid window can
        be from their centroid in meters, defaults to 75

    Returns
    -------
    list
        List of time indices of stop points.
    """
    df = df.sort_values(['timestamp_unix'])
    times = df.timestamp_unix.values  # Timestamps
    points = df.geometry.values  # GPS Points in Euclidean Coordinates
    # Get time weight of each point
    w = np.concatenate([[1], times[1:] - times[:-1]])

    # Convert coordinates to numpy array
    coords = []
    coords = np.vstack([[x.x for x in points], [x.y for x in points]]).T
    coords = coords.astype(np.float64)

    stop_points = calculate_stop_points_indices_compiled(
        times, w, coords, time_window, distance_threshold**2)
    stop_points = [(stop_points[i],
                    stop_points[i + 1],
                    stop_points[i + 2],
                    stop_points[i + 3]) for i in range(0,
                                                       len(stop_points),
                                                       4)]

    return stop_points


def precondition_table(df):
    """Prepare dataframe for calculating stop points and consolidates
    repeated points

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with columns: timestamp, longitude, latitude

    Returns
    -------
    tuple(pd.DataFrame, gpd.GeoDataFrame, np.ndarray)
        Return original DataFrame, GeoDataFrame with repeated points removed, mask
    """
    # Convert timetamps to unix epoch
    df['timestamp_unix'] = df['timestamp'].values.astype(
        'datetime64[s]').astype('int')

    # Get lat/long
    coords_full = df[['longitude', 'latitude']].to_numpy()

    # Remove repeated points
    mask = np.concatenate(
        [[True], np.all(coords_full[1:] != coords_full[:-1], axis=1)])
    mask = mask | np.concatenate(
        [np.all(coords_full[:-1] != coords_full[1:], axis=1), [True]])
    mask = binary_dilation(mask, iterations=120)
    sub_df = df[mask]

    # Create geodatagrame for conversion to utm
    sub_gdf = gpd.GeoDataFrame(
        sub_df,
        geometry=gpd.points_from_xy(
            sub_df.longitude,
            sub_df.latitude),
        crs="EPSG:4326")
    crs = sub_gdf.estimate_utm_crs()
    sub_gdf = sub_gdf.to_crs(crs)
    return df, sub_gdf, mask


def calculate_stop_points(agent_id, df, distance_threshold=75):
    """Calculates the stop points from a GPS trajectory set in ULLT format.

    Parameters
    ----------
    agent_id : str
        id of current agent
    df : pd.DataFrame
        Pandas DataFrame created from reading a parquet file or from querying
        postgres with columns: timestamp, latitute, longitude
    distance_threshold : float
        Distance threshold defining a stop (Default value = 75)

    Returns
    -------
    pd.DataFrame
        Returns a dataframe with columns: agent_id, time_start, time_stop, geometry
    """
    gdf, sub_gdf, mask = precondition_table(df)
    ts_full = gdf.timestamp_unix.values
    ts = sub_gdf.timestamp_unix.values
    tso_full = gdf.timestamp.values
    coords_full = gdf[['longitude', 'latitude']].to_numpy()

    stop_times = calculate_stop_points_indices(sub_gdf, distance_threshold)
    data = {
        'agent_id': [],
        'time_start': [],
        'time_stop': [],
        'x': [],
        'y': []}
    for stop_instance in stop_times:
        start, stop, start_delta, stop_delta = stop_instance
        mask = (
            ts_full >= (
                ts[start] +
                start_delta)) & (
            ts_full <= (
                ts[stop] +
                stop_delta))
        tstart = tso_full[ts_full == (ts[start] + start_delta)][0]
        tstop = tso_full[ts_full == (ts[stop] + stop_delta)][0]
        if start == stop:
            continue
        data['agent_id'].append(agent_id)
        data['time_start'].append(tstart)
        data['time_stop'].append(tstop)
        c = np.average(coords_full[mask], axis=0)
        data['x'].append(c[0])
        data['y'].append(c[1])
    df = pd.DataFrame(data)
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(
            df.x, df.y), crs='EPSG:4326')
    gdf = gdf.to_crs("EPSG:4326").dropna()[
        ['agent_id', 'time_start', 'time_stop', 'geometry']]

    return gdf


def calculate_agent_unique_stop_points(agent_id, gdf, dist_thresh=37.5):
    """Calculates and agent's unique stop points from a stop point table

    Parameters
    ----------
    agent_id : str
        id of the agent in question
    gdf : gpd.GeoDataFrame
        GeoDataFrame of stop points with geometry column
    dist_thresh : float, optional
        Distance threshold for consolidating points, defaults to 37.5

    Returns
    -------
    tuple(gpd.GeoDataFrame, gpd.GeoDataFrame)
        input gdf with a column indicating the stop point id, GeoDataFrame with mapping of stop points
        to lat/long coords
    """
    # Get columns for downfiltering results later
    cols = list(gdf.columns)
    # Assign each stop point and id
    gdf = gdf.sort_values('time_start')
    gdf['pid'] = list(range(len(gdf)))

    # Convert table to local UTM (quasi-cartesian)
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)

    # Dilate each point to the cluster radius
    dilated_gdf = gdf.copy()
    dilated_gdf['geometry'] = [x.buffer(dist_thresh)
                               for x in gdf['geometry'].values]

    # Spatially join the dialated points with the points
    clustered = dilated_gdf.sjoin(gdf, how='left', predicate='contains')
    cluster_map = {}  # Maps each point to its cluster id
    # List of stop points in each cluster
    cluster_points = defaultdict(list)

    # Points available to be clustered
    available_pid = list(gdf['pid'].values)

    # While points are still available, greedily
    # build clusters by picking a point taking all
    # points within dist_thresh of that point
    # and declaring it a cluster, remove those
    # points from available points
    cid = uuid4()  # cluster id
    while len(available_pid) > 0:
        # We still have points not clustered

        # Get all points within dist_thresh of this point
        sub = clustered[clustered.pid_left == available_pid[0]]
        sub = sub[[x in available_pid for x in sub.pid_right]]

        # Create and log the cluster
        vals = sub.pid_right.values
        geoms = sub.geometry.values
        for geom, val in zip(geoms, vals):
            available_pid.remove(val)
            cluster_map[val] = cid
            cluster_points[cid].append(geom)
        cid = uuid4()

    gdf['unique_stop_point'] = [cluster_map[x] for x in gdf.pid.values]
    gdf = gdf.to_crs('EPSG:4326')[cols + ['unique_stop_point']]
    gdf_updated = gdf

    # Calculate the cluster centroids
    data = {'agent_id': [], 'unique_stop_point': [], 'point': []}
    for cluster_id, points in cluster_points.items():
        data['agent_id'].append(agent_id)
        data['unique_stop_point'].append(cluster_id)
        data['point'].append(GeometryCollection(points).centroid)
    unique_stop_points = gpd.GeoDataFrame(data, geometry='point',
                                          crs=utm).to_crs('EPSG:4326')
    return gdf_updated, unique_stop_points


def calculate_set_unique_stop_points(gdf, dist_thresh=37.5):
    """Calculate Set-Level Unique Stop Points (global unique stop points)

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame of all agent unique stop points with columns
        agent_id, unique_stop_point, point
    dist_thresh : float, optional
        Distance threshold for consolidating points, defaults to 37.5

    Returns
    -------
    tuple(gpd.GeoDataFrame, gpd.GeoDataFrame)
        input gdf with global_stop_point column with global stop point,
        GeoDataFrame mapping global stop points UUID
        to Lat/Lon
    """

    print('Estimating UTM')
    # Convert to UTM (quasi-cartesian)
    utm = gdf.iloc[:100].estimate_utm_crs()
    gdf = gdf.to_crs(utm)

    print('Buffering Points')
    # Spatially join the dialated points with the points
    dilated_gdf = gdf.copy()
    dilated_gdf['point'] = [x.buffer(dist_thresh) for x in gdf['point'].values]
    cols = list(gdf.columns)

    print('Spatially Joining')
    
    cluster_map = {}  # Maps each point to its cluster id
    # List of stop points in each cluster
    cluster_points = defaultdict(list)
    available_pid = {x: True for x in gdf['unique_stop_point'].values}
    available_pid_ = {x: True for x in gdf['unique_stop_point'].values}
    tot=len(available_pid)
    avail = tot
    # While points are still available, greedily
    # build clusters by picking a point taking all
    # points within dist_thresh of that point
    # and declaring it a cluster, remove those
    # points from available points
    cid = uuid4()  # cluster id
    while avail > 0:
        # We still have points not clustered
        print(f'{avail/tot:.3f}', end='\r')
        pid = next(iter(available_pid_.keys()))
        # Get all points within dist_thresh of this point
        # Spatially join the dialated points with the points
        clustered = dilated_gdf[dilated_gdf.unique_stop_point==pid].sjoin(gdf, how='left', predicate='contains')
        sub = clustered[clustered.unique_stop_point_left == pid]
        sub = sub[[available_pid[x] for x in sub.unique_stop_point_right]]

        # Create and log the cluster
        vals = sub.unique_stop_point_right.values
        geoms = sub.geometry.values
        for geom, val in zip(geoms, vals):
            available_pid[val] = False
            del available_pid_[val]
            avail -= 1
            cluster_map[val] = cid
            cluster_points[cid].append(geom)
        cid = uuid4()

    gdf['global_stop_points'] = [cluster_map[x] for x in gdf.unique_stop_point.values]
    gdf_sp = gdf.to_crs('EPSG:4326')[cols + ['global_stop_points']]

    # Calculate the cluster centroids
    data = {'global_stop_points': [], 'point': []}
    for cluster_id, points in cluster_points.items():
        data['global_stop_points'].append(cluster_id)
        data['point'].append(GeometryCollection(points).centroid)
    gdf_global = gpd.GeoDataFrame(
        data, geometry='point', crs=utm).to_crs('EPSG:4326')
    return gdf_sp, gdf_global
