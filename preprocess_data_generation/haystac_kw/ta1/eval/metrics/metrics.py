"""
Add new metrics to this file and subclass the Metric class

Then add the metric to ALL_METRICS in the __init__.py
"""

import math

import pandas as pd
import numpy as np

from pandas import DataFrame
from datetime import timedelta
from collections import defaultdict
from shapely import wkt
from scipy.sparse import csr_array, dok_array

from haystac_kw.ta1.eval.metrics.metric_base import HistogramMetric, MatrixMetric
from haystac_kw.utils.data_utils.enu import get_enu_from_ll


class EncounterFrequencyMetric(HistogramMetric):
    """Class for computing Enounter Frequency metric."""
    def __init__(self):

        super(EncounterFrequencyMetric, self).__init__('Encounter Frequency')
        self.plot_params['Metric Name'] = 'Encounter Frequncy (encounters)'
        self.plot_params['plot_log10_x'] = True
        self.plot_params['plot_log10_y'] = True
        self.plot_params['y_axis_label'] = 'Count'
        self.plot_params['is_density'] = False

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        # Compute day/hour bins for calculating encounters
        df = event_table
        df['time_start'] = pd.to_datetime(df['time_start'])
        df['time_stop'] = pd.to_datetime(df['time_stop'])
        df['day_start'] = [x.day for x in df.time_start.dt.date.values]
        df['hour_start'] = [x.hour for x in df.time_start.dt.time.values]
        df['day_stop'] = [x.day for x in df.time_stop.dt.date.values]
        df['hour_stop'] = [x.hour for x in df.time_stop.dt.time.values]

        data = {'agent_id': [], 'day': [], 'hour': [], 'location': []}
        for i in range(len(df)):
            row = df.iloc[i]
            cur_time = row.time_start
            while cur_time < row.time_stop:
                data['agent_id'].append(row.agent_id)
                data['day'].append(cur_time.date().day)
                data['hour'].append(cur_time.time().hour)
                data['location'].append(row.global_stop_points)
                cur_time += timedelta(hours=1)
        df = pd.DataFrame(data)

        # Calculate potential encounters
        encounters = df.merge(df, on=['day', 'hour', 'location'])
        encounters = encounters[encounters.agent_id_x != encounters.agent_id_y]
        encounters = encounters[['agent_id_x', 'agent_id_y']].groupby(
            ['agent_id_x', 'agent_id_y']).value_counts().to_frame()
        encounters = encounters[[x[0] < x[1] for x in encounters.index]]
        output = defaultdict(list)
        if len(encounters) > 0:
            for x, v in zip(encounters.index, encounters['count'].values):
                output[(x[0], x[1])].append(v)
        return output


class TotalDistancePerDayMetric(HistogramMetric):
    """Class for computing Total Distance per Day metric"""
    def __init__(self):

        super(TotalDistancePerDayMetric, self).__init__(
            'Total Distance Per Day')
        self.plot_params['metric_name'] = "Distance per day (km)"
        self.plot_params['minx'] = 1
        self.plot_params['plot_log10_x'] = True
        self.plot_params['plot_log10_y'] = True

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        agent_stops = self.get_stop_points(event_table)
        # calc the distances
        distances = self.get_distances(agent_stops)
        return distances

    def get_distances(self, agent_stops: dict) -> list:
        """Accumulates the distances from a dictionary into a distribution (list)

        Parameters
        ----------
        agent_stops : Dict
            Dictionary of stops 

        Returns
        -------
        list
            List of distances traveled each day
        """
        distances = defaultdict(list)
        for date in agent_stops:
            for agent in agent_stops[date]:
                # get all the stop points
                coords = agent_stops[date][agent]
                # get the points grouped by 2s (e.g. [[0,1], [1,2], [2,3]] )
                subList = [coords[n:n + 2] for n in range(0, len(coords))]
                distance = 0
                for points in subList:
                    if len(points) < 2:
                        # ignore the last item since it won't have a pair
                        continue
                    # get the distance between the 2 stop points
                    # euclidean since we are in EN coordinates
                    dist = ((points[0][0] - points[1][0])**2 +
                            (points[0][1] - points[1][1])**2)**0.5
                    distance += dist
                # add this distance per the day
                distances[agent].append(distance)
        return distances

    def get_stop_points(self, event_table: pd.DataFrame) -> dict:
        """Get stop points per agent per day in a cartesian coordinate
        system.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop Points

        Returns
        -------
        dict
            Stop points organized by agent,date in ENU.
        """

        stops_by_date_by_agent = {}
        event_table['time_start'] = pd.to_datetime(event_table['time_start'])
        for index, row in event_table.iterrows():
            agent = row['agent_id']
            date = row.time_start.date()
            if date not in stops_by_date_by_agent.keys():
                stops_by_date_by_agent[date] = {}
            if agent not in stops_by_date_by_agent[date].keys():
                stops_by_date_by_agent[date][agent] = []
            coords = row['geometry']
            temp = coords.split("(")[-1]
            temp = temp.split(")")[0]
            lon, lat = temp.split(" ")
            en = get_enu_from_ll(float(lat), float(lon))
            stops_by_date_by_agent[date][agent].append([x / 1000 for x in en])
        return stops_by_date_by_agent


class RadiusOfGyrationMetric(HistogramMetric):
    """Class for computing Radius of Gyration Metric"""
    def __init__(self):
        """
        Constructor method
        """
        super(RadiusOfGyrationMetric, self).__init__('Radius of Gyration')
        self.plot_params['metric_name'] = "Radius of Gyr. (km)"
        self.plot_params['plot_log10_x'] = True
        self.plot_params['plot_log10_y'] = True

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        dist_stops = self.get_stop_points(event_table)
        agent_ids = list(dist_stops.keys())

        rog = defaultdict(list)
        for agent in agent_ids:
            # calculate ROG
            xy = dist_stops[agent]
            rog[agent].append(math.sqrt(sum(np.std(xy, axis=0)**2)))
        return rog

    def get_stop_points(self, event_table: DataFrame) -> dict:
        """Get stop points per agent in a cartesian coordinate
        space.

        Parameters
        ----------
        event_table : DataFrame
            Stop Points

        Returns
        -------
        dict
            Stop points converted to ENU
        """
        dist_of_stops = defaultdict(list)

        # Convert geometry to ENU
        geom = [wkt.loads(x) for x in event_table.geometry.values]
        enu = [get_enu_from_ll(x.x, x.y) for x in geom]
        enu = np.array([(x[0] / 1000, x[1] / 1000) for x in enu])

        # Iterate over array (faster than pandas)
        agent_list = event_table.agent_id.values
        for i in range(len(agent_list)):
            dist_of_stops[agent_list[i]].append(enu[i])
        return dist_of_stops


class NumberLocationsVisitedMetric(HistogramMetric):
    """Class for computing the Number of Locations Visited Metric"""
    def __init__(self):

        super(NumberLocationsVisitedMetric, self).__init__(
            'Number of Locations Visited')

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        unique_dist = self.get_unique_stops(event_table)

        # now pull the numbers from the dictionary
        distribution = defaultdict(list)
        for date in unique_dist:
            for agent in unique_dist[date]:
                # add up the total number of unique items
                distribution[agent].append(len(unique_dist[date][agent]))
        return distribution

    def add_uid(self, dist: dict, uid: str, date: str, agent: str) -> None:
        """Add a location to the list of places an agent
        visited on a particular date.

        Parameters
        ----------
        dist : dict
            Dictionary to be populated
        uid : str
            UUID of location
        date : str
            Date of visit
        agent : str
            UUID of agent.
        """
        if date not in dist.keys():
            dist[date] = {}
        if agent not in dist[date].keys():
            dist[date][agent] = []

        if uid not in dist[date][agent]:
            # add the uid
            dist[date][agent].append(uid)

    def get_unique_stops(self, event_table: DataFrame) -> dict:
        """Create a dictionary containing a list of places
        an agent visited each day.

        Parameters
        ----------
        event_table : DataFrame
            Stop Points

        Returns
        -------
        dict
            Dictionary mapping agent id and date to a
            list of locations visited by that agent.
        """
        unique_dist = {}
        agent_ids = event_table['agent_id'].values
        dates = pd.to_datetime(event_table['time_start'].values)
        uids = event_table['unique_stop_point'].values
        for i in range(len(uids)):
            agent = agent_ids[i]
            uid = uids[i]
            full_date = dates[i]
            date = str(full_date.date())
            self.add_uid(unique_dist, uid, date, agent)
        return unique_dist


class TemporalVariabilityMetric(HistogramMetric):
    """Class for computing Temporal Variability Metric"""
    def __init__(self, rank=2):
        """
        Constructor method.

        Parameters
        ----------
        rank : int, optional
            Highest rank location to consider (defaults to 2)
        """

        super(TemporalVariabilityMetric, self).__init__('Temporal Variability')
        self.rank = rank

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        # Load df of unique stop points
        df = event_table.copy()
        df['time_start'] = pd.to_datetime(df['time_start'])
        df['time_stop'] = pd.to_datetime(df['time_stop'])
        df['arrival_time_s'] = [
            x.hour *
            60 *
            60 +
            x.minute *
            60 +
            x.second for x in df.time_start.dt.time.values]
        df['departure_time_s'] = [
            x.hour *
            60 *
            60 +
            x.minute *
            60 +
            x.second for x in df.time_stop.dt.time.values]
        # Get List of Agent Ids
        agent_ids = sorted(df.agent_id.unique())

        # Get visit count for each stop point
        loc_counts = df.groupby(['agent_id', 'unique_stop_point'])[
            'unique_stop_point'].value_counts().sort_values(ascending=False)

        data = defaultdict(list)
        for agent in agent_ids:
            locations = loc_counts.loc[agent, :].head(self.rank).index.values
            agent_sub = df[df.agent_id == agent]
            for i, location in enumerate(locations):
                location_sub = \
                    agent_sub[agent_sub.unique_stop_point == location]
                std_a = location_sub.arrival_time_s.std()
                std_d = location_sub.departure_time_s.std()

                # data[f'std-arr-{i}'].append(std_a)
                # data[f'std-dep-{i}'].append(std_d)
                if not np.isnan(std_a):
                    data[agent].append(std_a)
                if not np.isnan(std_d):
                    data[agent].append(std_d)
        return data


class LevelOfExplorationMetric(HistogramMetric):
    """Class for computing the Level of Exploration Metric"""
    def __init__(self):
        """
        Constructor method.
        """
        super(LevelOfExplorationMetric, self).__init__('Level of Exploration')

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        # get the sets split by dates
        # these dictionaries will now have the data arranged as:
        # dict[agent][uid] = (count of visits per simulation)
        # and dict[agent]["total_stops"] = (total stops for simulation)
        unique_dist = self.get_unique_stops(event_table)

        # pick an agent from the reference set to be the comparison
        # TODO: how does this get decided?  For now we are getting the first
        # one
        print('LEVEL OF EXPLORATION METRIC IMPLEMENTED BUT NEEDS DISCUSSION')
        agents = list(unique_dist.keys())
        # store this distribution for plotting
        distribution = defaultdict(list)
        for agent in agents:
            distribution[agent] = self.get_topk_list_per_agent(
                unique_dist, agent)
        return distribution

    def add_uid(self, dist: dict, uid: str, agent: str) -> None:
        """Increment the count of the number of times the agent
        visited a particular location.

        Parameters
        ----------
        dist : dict
            Dictionary of arrival counts per location
        uid : str
            UUID of the location
        agent : str
            UUID of the agent
        """

        if agent not in dist.keys():
            dist[agent] = {}
            dist[agent]["total_stops"] = 0

        if uid not in dist[agent]:
            # add the uid
            dist[agent][uid] = 0

        # count a visit to this uid
        dist[agent][uid] += 1
        # get the total stops per agent
        dist[agent]["total_stops"] += 1

    def get_unique_stops(self, event_table: pd.DataFrame) -> dict:
        """Compute the number of times each agent visited each
        location.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop Points

        Returns
        -------
        dict
            Dictionary mapping agent and location to visit count
        """
        unique_dist = {}
        agents = event_table.agent_id.values
        uids = event_table.unique_stop_point.values
        for i in range(len(uids)):
            agent = agents[i]
            uid = uids[i]
            self.add_uid(unique_dist, uid, agent)

        return unique_dist

    def get_topk_list_per_agent(self, dist: dict, agent: str) -> list:
        """Return sorted list of number of visits to each location
        the agent visited.

        Parameters
        ----------
        dist : dict
            Dictionary mapping agent and location to visit count
        agent : str
            UUID of agent

        Returns
        -------
        list
            Sorted list of visit counts
        """
        topk = [dist[agent][u] for u in dist[agent] if u != "total_stops"]
        topk = sorted(topk, reverse=True)
        out = []
        for i in range(len(topk)):
            for j in range(topk[i]):
                out.append(i)
        return out


class InterEncounterTimeMetric(HistogramMetric):
    """Class for computing Inter Encounter Time Metric"""
    def __init__(self):
        """
        Constructor method
        """
        super(InterEncounterTimeMetric, self).__init__('Inter-Encounter Time')
        self.plot_params['plot_log10_x'] = False
        self.plot_params['plot_log10_y'] = False
        self.plot_params['minx'] = 3600  # 2 minutes

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        df = event_table.copy()

        # Compute day/hour bins for calculating encounters
        df['time_start'] = pd.to_datetime(df['time_start'])
        df['time_stop'] = pd.to_datetime(df['time_stop'])
        df['day_start'] = [x.day for x in df.time_start.dt.date.values]
        df['hour_start'] = [x.hour for x in df.time_start.dt.time.values]
        df['day_stop'] = [x.day for x in df.time_stop.dt.date.values]
        df['hour_stop'] = [x.hour for x in df.time_stop.dt.time.values]

        data = {
            'agent_id': [],
            'day': [],
            'hour': [],
            'location': [],
            'time_start': [],
            'time_stop': []}
        for i in range(len(df)):
            row = df.iloc[i]
            cur_time = row.time_start
            while cur_time < row.time_stop:
                data['agent_id'].append(row.agent_id)
                data['day'].append(cur_time.date().day)
                data['hour'].append(cur_time.time().hour)
                data['location'].append(row.global_stop_points)
                data['time_start'].append(row.time_start)
                data['time_stop'].append(row.time_stop)
                cur_time += timedelta(hours=1)
        df = pd.DataFrame(data)

        # Calculate potential encounters
        encounters = df.merge(df, on=['day', 'hour', 'location'])
        encounters = encounters[encounters.agent_id_x != encounters.agent_id_y]
        encounters = encounters.drop_duplicates(
            subset=['agent_id_x', 'agent_id_y', 'day', 'hour'])
        encounters = encounters.drop_duplicates(
            subset=[
                'agent_id_x',
                'agent_id_y',
                'time_start_x',
                'time_start_y'])
        encounters['encounter_start'] = pd.to_datetime(
            [max(x, y) for x, y in
             zip(encounters.time_start_x, encounters.time_start_y)])
        encounters['encounter_stop'] = pd.to_datetime(
            [min(x, y) for x, y in
             zip(encounters.time_stop_x, encounters.time_stop_y)])
        encounters = encounters.set_index(['agent_id_x', 'agent_id_y'])
        encounters = encounters[[x[0] < x[1] for x in encounters.index]]
        encounter_deltas_s = defaultdict(list)
        unique_agent_pairs = set(encounters.index)
        encounters = encounters.sort_index()
        for pair in unique_agent_pairs:
            pair_encounters = encounters.loc[pair[0], pair[1]][[
                'day', 'hour', 'encounter_start',
                'encounter_stop']].sort_values('encounter_start')
            if len(pair_encounters) == 1:
                continue
            deltas = (
                pair_encounters.encounter_start.values[1:] -
                pair_encounters.encounter_stop.values[:-1]
            ) / np.timedelta64(1, 's')
            deltas = deltas[deltas > 0]
            encounter_deltas_s[pair] += list(deltas)
        return encounter_deltas_s


class OriginDestinationProbabilityMetric(MatrixMetric):
    """Class for computing Origin Destination Probability Metric"""
    def __init__(self, res: int = 7):
        """
        Constructor Method

        Parameters
        ----------
        res : int, optional
            H3 resolution (defaults to 7)
        """
        metric_name = 'Origin Destination Probability'
        super(OriginDestinationProbabilityMetric, self).__init__(metric_name)
        self.resolution_col = f'res_{res}'

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        aoi = event_table[self.resolution_col].unique()
        agents = event_table['agent_id'].unique()

        df = event_table.copy().sort_values(['agent_id', 'time_start'])
        agents.sort()
        aoi = np.sort(aoi)
        aoi_map = {aoi[i]: i for i in range(len(aoi))}

        agent_matrices = {}
        agent_values = df.agent_id.values
        loc_values = np.array([aoi_map[x]
                              for x in df[self.resolution_col].values])
        agent_ind = pd.Index(agent_values)
        for agent in agents:
            mask = agent_ind.get_loc(agent)
            # mask = agent_values==agent
            loc_array = loc_values[mask]
            agent_matrix = dok_array((len(aoi), len(aoi)), np.int64)

            for i in range(len(loc_array) - 1):
                agent_matrix[(loc_array[i], loc_array[i + 1])] += 1
            agent_matrices[agent] = csr_array(agent_matrix)
        return agent_matrices, aoi


class LocationConnectivityMetric(MatrixMetric):
    """Class for computing Location Connectivity Metric"""
    def __init__(self, res: int = 7):
        """
        Constructor method

        Parameters
        ----------
        res : int, optional
            H3 Resolution (defaults to 7)
        """
        metric_name = 'Location Connectivity'
        super(LocationConnectivityMetric, self).__init__(metric_name)
        self.resolution_col = f'res_{res}'

    def compute_metric(self, event_table: DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        aoi = event_table[self.resolution_col].unique()
        agents = event_table['agent_id'].unique()

        df = event_table.copy().sort_values(['agent_id', 'time_start'])
        agents.sort()
        aoi = np.sort(aoi)
        aoi_map = {aoi[i]: i for i in range(len(aoi))}

        agent_matrices = {}
        agent_values = df.agent_id.values
        loc_values = np.array([aoi_map[x]
                              for x in df[self.resolution_col].values])
        agent_ind = pd.Index(agent_values)
        for agent in agents:
            mask = agent_ind.get_loc(agent)
            loc_array = loc_values[mask]
            agent_matrix = dok_array((len(aoi), len(aoi)), np.int64)

            for i in range(len(loc_array) - 1):
                agent_matrix[(loc_array[i], loc_array[i + 1])] = 1
            agent_matrices[agent] = csr_array(agent_matrix)
        return agent_matrices, aoi
