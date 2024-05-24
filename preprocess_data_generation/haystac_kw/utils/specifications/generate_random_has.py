""" This file contains functions that build HAS files randomly. 
For a CLI to generate HAS files (to actually generate HAS files), 
please see haystac_kw.ta2.cli.generate_random_has.
"""

import json
import os
import argparse

os.environ['USE_PYGEOS'] = '0'

import uuid
from shapely import GeometryCollection
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from haystac_kw.utils.data_utils.conversions import has2internalhas

EARLIEST_DATE = datetime(2023, 3, 1, 0, 0, 0, 0)

def create_n_random_has(
        poi_pickle_path,
        output_dir="random_has_outputs",
        n=10,
        hos_file=None,
        internal_format=False,
        stop_point_file=None):
    """Creates N random HAS specifications. Not based on any HOS specifications; this is purely for generating examples.
    Does not return anything. Instead, it saves output to a defined directory.

    Parameters
    ----------
    output_dir : path-like str, optional
        directory to store random outputs. If it doesn't exist, the directory path is made. Default is "./random_has_outputs"
    n : int, optional 
        number of specifications to generate at random, default is 10
    hos_file : path-like str, optional
        path of HOS file to constrain HAS generation. Defaults to None. If none, HAS files are generated without constraints and event IDs will be randomly generated.
    """

    # modify the output path to indicate which hos file the has generation
    # corresponds to
    if hos_file is not None:
        output_dir = output_dir + "_%s" % os.path.basename(hos_file)[:-5]

    os.makedirs(output_dir, exist_ok=True)

    for i in range(n):
        filename = os.path.join(output_dir, "random_has_%d.json" % i)
        create_random_has(poi_pickle_path, filename, hos_file=hos_file)

    # convert to internal format if --internal flag is specified
    if internal_format:
        has2internalhas(
            input_spec_file_folder=output_dir,
            sim_stop_points_pq_path=stop_point_file,
            output_spec_file_folder=output_dir,
            distance_threshold=75)

    print(
        "Done generating %d random HAS outputs. \nSavedir: %s" %
        (n, os.path.abspath(output_dir)))


def create_random_has(poi_pickle_path, savepath=None, hos_file=None):
    """Creates a randomly initialized HAS dictionary object based on the HAS Schema specification:
    https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Hide-Activity-Specification

    Parameters
    ----------
    savepath : str, optional
        path to save to, defaults to None. Saves the HAS output if savepath is defined
    hos_file : path-like str, optional
        path of HOS file to constrain HAS generation. Defaults to None. If none, HAS files are
        generated without constraints and event IDs will be randomly generated.
        
    Returns
    -------
    dict
        random HAS data
    """

    poi_df = pd.read_pickle(poi_pickle_path)
    poi_df = gpd.GeoDataFrame(poi_df,
                              geometry="building_centroid",
                              crs='EPSG:4326')

    utm = poi_df.estimate_utm_crs()

    poi_df = gpd.GeoDataFrame(poi_df,
                              geometry="building_poly",
                              crs=utm).to_crs('EPSG:4326')

    objective_uid = str(uuid.uuid4())
    empty_has = initialize_empty_has(objective_uid)

    rng = np.random.default_rng()
    num_agents = rng.integers(low=1, high=10, size=1)[0]
    random_agent_list = rng.integers(low=1, high=1000, size=num_agents)

    if hos_file is not None:
        event_data, time_constraints, duration_constraints = read_hos_file(
            hos_file)
                
        hos_agents = [] 
        for x in event_data["agents"].values:
            for agent_id in x: 
                hos_agents.append(agent_id)

        hos_agents = list(set(hos_agents))
        
        random_agent_list = [x for x in random_agent_list if x not in hos_agents]

    if hos_file is not None:
        for hos_agent in hos_agents: 
            hos_agent = int(hos_agent)
            movement = create_movement(hos_agent)
            num_itineraries = rng.integers(low=1, high=10, size=1).item()

            # for _ in range(num_itineraries):

            itinerary_data = _create_hos_itinerary(
                hos_agent, event_data, poi_df, rng, time_constraints, duration_constraints)

            movement["itineraries"].append(itinerary_data)

            empty_has["movements"].append(movement)
    else:
        # generate movements for each agent
        for agent in random_agent_list:

            agent = int(agent)
            movement = create_movement(agent)
            num_itineraries = rng.integers(low=1, high=10, size=1).item()

            for _ in range(num_itineraries):

                itinerary_data = _create_random_itinerary(poi_df, rng)

                movement["itineraries"].append(itinerary_data)

            empty_has["movements"].append(movement)

    assert len(empty_has["movements"]) > 0, "movements is empty for agents %s" % str(
        random_agent_list)

    # if savepath is specified, save the HAS file to disk
    if savepath:

        assert savepath.endswith(
            ('.json')), "savepath must be appended with *.json extension"

        with open(savepath, 'w') as fp:
            json.dump(empty_has, fp, indent=2)

        return 'Exported HAS random JSON to %s' % savepath

    return empty_has

def read_hos_file(path):
    """Reads HOS file and returns both a list of time constraints 
    and a list of duration constraints.

    Parameters
    ----------
    path : path-like str, optional
        path of hos file to read

    Returns
    -------
    Tuple
        tuple consisting of events, time constraints, and duration constraints [List, List]
    """

    assert path.endswith(
        ".json"), "HOS file must have *.json extension and be a valid json file"

    with open(path) as f:
        hos_data = json.load(f)

    # process time constraint data
    new_time_constraint_data = []

    for data in hos_data["time_constraints"]:

        new_data = {}
        new_data["event_uid"] = data["event"]
        new_data["time_window_begin"] = data["time_window"]["begin"]
        new_data["time_window_end"] = data["time_window"]["end"]
        new_time_constraint_data.append(new_data)

    time_constraint_data = pd.DataFrame.from_dict(new_time_constraint_data)

    global EARLIEST_DATE
    time_constraint_data["time_window_begin_dt"] = time_constraint_data["time_window_begin"].apply(lambda x: datetime.fromisoformat(x.replace("Z", ""))) 
    time_constraint_data["time_window_end_dt"] = time_constraint_data["time_window_end"].apply(lambda x: datetime.fromisoformat(x.replace("Z", ""))) 

    EARLIEST_DATE = time_constraint_data["time_window_begin_dt"].min() 

    global LATEST_DATE 
    LATEST_DATE = time_constraint_data["time_window_end_dt"].max() 

    # process duration constraint data
    new_duration_constraint_data = []

    for data in hos_data["duration_constraints"]:

        new_data = {}
        new_data["first_event_uid"] = data["events"]["first"]
        new_data["second_event_uid"] = data["events"]["second"]
        new_data["duration_window_minimum"] = data["duration_window"]["minimum"]
        new_data["duration_window_maximum"] = data["duration_window"]["maximum"]
        new_data["stay"] = data["stay"]
        new_duration_constraint_data.append(new_data)

    duration_constraint_data = pd.DataFrame.from_dict(
        new_duration_constraint_data)

    event_data = pd.DataFrame.from_dict(hos_data["events"])

    return event_data, time_constraint_data, duration_constraint_data


def create_hos_movements(path: str, poi_pickle_path: str):
    """Uses event data and constraint data to construct movements 
    directly fulfilling the constraints. Does not generate other movements.

    Parameters
    ----------
    path : path-like str, optional
        path of hos file to read

    Returns
    -------
    List[dict]
        list of movements which fulfill the HOS objectives
    """

    event_data, time_constraints, duration_constraints = read_hos_file(path)

    poi_df = pd.read_pickle(poi_pickle_path)
    poi_df = gpd.GeoDataFrame(poi_df,
                              geometry="building_centroid",
                              crs='EPSG:4326')

    utm = poi_df.estimate_utm_crs()

    poi_df = gpd.GeoDataFrame(poi_df,
                              geometry="building_poly",
                              crs=utm).to_crs('EPSG:4326')

    rng = np.random.default_rng()
    _create_hos_itinerary(
        event_data,
        poi_df,
        rng,
        time_constraints,
        duration_constraints)


def _create_hos_itinerary(agent_id: int, 
                          events: pd.DataFrame,
                          poi_gdf: gpd.GeoDataFrame,
                          rng: np.random.BitGenerator,
                          time_constraint: pd.DataFrame = None,
                          duration_constraint: pd.DataFrame = None):
    """
    Helper function for generating HOS itineraries. 
    Please do not call directly.

    agent_id : int
        agent id to generate itinerary for
    events : pandas.DataFrame
        pandas dataframe of event data linking event UIDs to location.
    poi_gdf : geopandas.Dataframe
        places-of-interest geodataframe containing all the the polygons to sample from
    rng : np.random.BitGenerator
        random number generator object
    time_constraint : pandas.Dataframe, optional
        time constraint dataframe containing all time constraints, defaults to None
    duration_constraint : pandas.Dataframe, optional
        duration constraint dataframe containing all duration constraints, defaults to None

    Returns
    -------
    dict
        one itinerary data dict
    """

    assert time_constraint is not None or duration_constraint is not None, "either time or duration constraint must be provided."

    itinerary = []

    geojson_dict = sample_random_geojson(poi_gdf)

    # First, we generate the HOS-fulfilling movements
    # assume that the start data can range from 60 to 360 minutes here
    global EARLIEST_DATE
    time_window_size_minutes = rng.integers(low=60, high=360, size=1).item()
    start_time = EARLIEST_DATE - timedelta(minutes=time_window_size_minutes)
    end_time = EARLIEST_DATE - timedelta(minutes=time_window_size_minutes - 60)

    EARLIEST_DATE = end_time

    # since each itinerary must start with a start, we create one first no matter what
    start_data = create_itinerary_start(
        geojson_dict,
        start_time.isoformat() + 'Z',
        end_time.isoformat() + 'Z')
    itinerary.append(start_data)

    # now, let's create movements fulfilling time constraints
    hos_itinerary_data_time = []

    existing_locations = []

    for _, row in time_constraint.iterrows():

        event_type = events.loc[events['event_uid']
                                   == row["event_uid"]]["event_type"].values[0]
        event_agents = events.loc[events['event_uid']
                                   == row["event_uid"]]["agents"].values[0]
        
        if agent_id not in event_agents: 
            continue 
        
        # if time constraint is applied on departure, we make sure the move 
        # just moves the agent away from the current location and sample 
        # a random location to move to
        if event_type == "depart": 
            move_location = sample_random_geojson(poi_gdf) 
        # otherwise, if we need to arrive at this location, then move location 
        # should be the arrival location
        else: 
            move_location = events.loc[events['event_uid']
                                   == row["event_uid"]]["location"].values[0]
        
        move_data = create_itinerary_move(move_location, "personal_vehicle")

        existing_locations.append(str(move_location))

        dt_begin = datetime.fromisoformat(row["time_window_begin"].replace("Z", ""))
        dt_end = datetime.fromisoformat(row["time_window_end"].replace("Z", ""))
        dt_duration_max = dt_end - dt_begin
        dt_duration_max_minutes = dt_duration_max.total_seconds() // 60

        time_window_size_minutes = rng.integers(
            low=0, high=dt_duration_max_minutes, size=1).item()
        end_time = dt_begin + \
            timedelta(minutes=time_window_size_minutes)
        end_time = end_time.isoformat() + 'Z'
        priority = "end_time"

        stay_data = create_itinerary_stay(None, end_time, priority)

        hos_itinerary_data_time.append(move_data)
        hos_itinerary_data_time.append(stay_data)

    # next, we create movements fulfilling duration constraints

    hos_itinerary_data_duration = []

    for i, row in duration_constraint.iterrows():
        
        # Commented; left here for future development reference on retrieving event type
        # event_type_1 = events.loc[events['event_uid']
        #                            == row["first_event_uid"]]["event_type"].values[0]
        # event_type_2 = events.loc[events['event_uid']
        #                            == row["second_event_uid"]]["event_type"].values[0]
        
        event_agents_1 = events.loc[events['event_uid']
                                   == row["first_event_uid"]]["agents"].values[0]
        event_agents_2 = events.loc[events['event_uid']
                                   == row["second_event_uid"]]["agents"].values[0]
        
        if agent_id not in event_agents_1 and agent_id not in event_agents_2: 
            continue 

        # TODO: check if first event is depart and second is arrive 

        move_location = events.loc[events['event_uid'] ==
                                row["first_event_uid"]]["location"].values[0]
        
        existing_locations.append(str(move_location))

        move_data = create_itinerary_move(move_location, "personal_vehicle")

        duration_min = pd.Timedelta(
            row["duration_window_minimum"]).total_seconds() // 60 + 1
        duration_max = pd.Timedelta(
            row["duration_window_maximum"]).total_seconds() // 60
        duration_minutes_random = rng.integers(
            low=duration_min, high=duration_max, size=1).item()

        duration = pd.Timedelta(
            minutes=float(duration_minutes_random)).isoformat()
        priority = "duration"

        stay_data = create_itinerary_stay(duration, None, priority)

        hos_itinerary_data_duration.append(move_data)
        hos_itinerary_data_duration.append(stay_data)

    # now, we build the final itinerary which combines all movements
    final_itinerary = itinerary[:1]
    
    # we want to keep track of the end time of the last movement to make sure
    # everything happens chronologically
    last_end_time = datetime.fromisoformat(final_itinerary[0]["start"]["time_window"]["end"].replace("Z", ""))

    # we iterate through the generated time constraint movements, since they 
    # have a concrete time assigned to them 
    for _ in range(len(hos_itinerary_data_time) // 2):
        move_data = hos_itinerary_data_time.pop(0) 
        stay_data = hos_itinerary_data_time.pop(0)

        current_end_time = datetime.fromisoformat(stay_data["stay"]["end_time"].replace("Z", ""))
        time_gap = current_end_time - last_end_time 

        # add random trip here if the time gap is larger than 30 minutes
        if time_gap > timedelta(minutes=30):
            move_location = None 

            while move_location is None or str(move_location) in existing_locations:
                move_location = sample_random_geojson(poi_gdf)

            move_data_random = create_itinerary_move(move_location, "personal_vehicle")

            # flip a coin for duration or end time, or both
            # 0 = duration, 1 = end time, 2 = both
            coin_flip = rng.integers(3, size=1).item()

            duration = None
            end_time = None
            priority = None

            if coin_flip == 0:
                duration_minutes_random = rng.integers(low=15, high=time_gap.total_seconds() // 60, size=1).item()
                duration = pd.Timedelta(
                    minutes=float(duration_minutes_random)).isoformat()
                last_end_time = last_end_time + \
                    timedelta(minutes=duration_minutes_random)
            elif coin_flip == 1:
                time_window_size_minutes = rng.integers(low=15, high=time_gap.total_seconds() // 60, size=1).item()
                sampled_start_time = last_end_time + \
                    timedelta(minutes=10)
                end_time_dt = sampled_start_time + \
                    timedelta(minutes=time_window_size_minutes)
                end_time = end_time_dt.isoformat() + 'Z'
                last_end_time = end_time_dt
            else:
                duration_minutes_random = rng.integers(low=15, high=time_gap.total_seconds() // 60, size=1).item()
                duration = pd.Timedelta(
                    minutes=float(duration_minutes_random)).isoformat()

                time_window_size_minutes = rng.integers(low=15, high=time_gap.total_seconds() // 60, size=1).item()
                sampled_start_time = last_end_time + \
                    timedelta(minutes=10)
                end_time_dt = sampled_start_time + \
                    timedelta(minutes=time_window_size_minutes)
                end_time = end_time_dt.isoformat() + 'Z'
                start_time_max_duration = end_time_dt + \
                    pd.Timedelta(minutes=float(duration_minutes_random))
                last_end_time = max(sampled_start_time, start_time_max_duration)

                # another coin flip for priority
                priority = "duration" if rng.integers(
                    2, size=1).item() == 0 else "end_time"
                
            stay_data_random = create_itinerary_stay(duration, end_time, priority)

            # add the random non-HOS related movements to the itinerary
            final_itinerary.append(move_data_random)
            final_itinerary.append(stay_data_random)

        # add the time constraint movements themselves
        final_itinerary.append(move_data)
        final_itinerary.append(stay_data)

        last_end_time = current_end_time
    
    for i in range(len(hos_itinerary_data_duration) // 2):
        final_itinerary.append(hos_itinerary_data_duration.pop(0))
        final_itinerary.append(hos_itinerary_data_duration.pop(0))

    # Add a final random move / stay combo to end the itinerary
    move_location = None 
    while move_location is None or str(move_location) in existing_locations:
        move_location = sample_random_geojson(poi_gdf)
    move_data = create_itinerary_move(move_location, "personal_vehicle")
    stay_data = create_itinerary_stay(120, None, "duration")
    final_itinerary.append(move_data)
    final_itinerary.append(stay_data)

    # do some checks to make sure all constraints are addressed in HAS
    assert len(hos_itinerary_data_duration) == 0, "duration constraints still left"
    assert len(hos_itinerary_data_time) == 0, "time constraints still left"

    itinerary_dict = {
        "itinerary": final_itinerary
    }

    return itinerary_dict


def _create_random_itinerary(poi_gdf, rng):
    """Helper function for generating random itineraries. Please do not call directly.

    Parameters
    ----------
    poi_gdf : geopandas.Dataframe
        places-of-interest geodataframe containing all the the polygons to sample from
    rng : np.random.BitGenerator
        random number generator object

    Returns
    -------
    dict
        one itinerary data dict
    """

    itinerary = []

    geojson_dict = sample_random_geojson(poi_gdf)

    time_window_size_minutes = rng.integers(low=60, high=120, size=1).item()
    sampled_start_time = EARLIEST_DATE
    sampled_end_time = sampled_start_time + \
        timedelta(minutes=time_window_size_minutes)

    start_data = create_itinerary_start(
        geojson_dict,
        sampled_start_time.isoformat() + 'Z',
        sampled_end_time.isoformat() + 'Z')
    itinerary.append(start_data)

    # randomly select how many stops we make in the itinerary; must be at
    # least 1
    stops_in_itinerary = rng.integers(low=1, high=5, size=1).item()

    for i in range(stops_in_itinerary):

        move_location = sample_random_geojson(poi_gdf)
        move_data = create_itinerary_move(move_location, "personal_vehicle")

        # flip a coin for duration or end time, or both
        # 0 = duration, 1 = end time, 2 = both
        coin_flip = rng.integers(3, size=1).item()

        duration = None
        end_time = None
        priority = None

        if coin_flip == 0:  # we only use duration
            duration_minutes_random = rng.integers(120, size=1).item()
            duration = pd.Timedelta(
                minutes=float(duration_minutes_random)).isoformat()
        elif coin_flip == 1:  # we only use time constraints
            time_window_size_minutes = rng.integers(480, size=1).item()
            sampled_start_time = EARLIEST_DATE
            end_time = sampled_start_time + \
                timedelta(minutes=time_window_size_minutes)
            end_time = end_time.isoformat() + 'Z'
        else:  # both duration and time constraints are used
            duration_minutes_random = rng.integers(120, size=1).item()
            duration = pd.Timedelta(
                minutes=float(duration_minutes_random)).isoformat()

            time_window_size_minutes = rng.integers(480, size=1).item()
            sampled_start_time = EARLIEST_DATE
            end_time = sampled_start_time + \
                timedelta(minutes=time_window_size_minutes)
            end_time = end_time.isoformat() + 'Z'

            # another coin flip for priority
            priority = "duration" if rng.integers(
                2, size=1).item() == 0 else "end_time"

        stay_data = create_itinerary_stay(duration, end_time, priority)

        itinerary.append(move_data)
        itinerary.append(stay_data)

    itinerary_dict = {
        "itinerary": itinerary
    }
    return itinerary_dict


def sample_random_geojson(poi_gdf: gpd.GeoDataFrame):
    """Sample a random polygon from the GeoJSON dataframe, and return 
    the GeometryCollection JSON needed for the HAS Schema.

    Parameters
    ----------
    poi_gdf : geopandas.Dataframe
        places-of-interest geodataframe containing all the the 
        polygons to sample from

    Returns
    -------
    dict
        dict representing shapely.GeometryCollection/polygon 
        information of a sampled location
    """

    sampled_location = poi_gdf.sample(n=1)  # pick a random location from gdf
    # get the polygon info
    location_polygon = sampled_location["building_poly"]

    geojson_collection = GeometryCollection([location_polygon])
    geojson_dict = json.loads(gpd.GeoSeries(
        geojson_collection).set_crs('EPSG:4326').to_json())
    # extract the geojson
    geojson_dict = geojson_dict['features'][0]['geometry']

    return geojson_dict


def initialize_empty_has(object_uid: str):
    """Initializes an empty HAS dictionary based on the HAS Schema specification:
    https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Hide-Activity-Specification

    Parameters
    ----------
    object_uid : str
        objective UID of HOS specification that fits this activity

    Returns
    -------
    dict
        HAS data
    """

    has_data = {
        "schema_version": "1.1.2",
        "schema_type": "HAS",
        "objective": str(object_uid),
        "movements": []
    }

    return has_data


def create_movement(agent_id: int):
    """Creates movement dictionary information based on the HAS Schema specification:
    https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Hide-Activity-Specification

    Parameters
    ----------
    agent_id : int
        agent id

    Returns
    -------
    dict
        movement data
    """

    movement_data = {
        "agent": agent_id,
        "itineraries": []
    }

    return movement_data


def create_itinerary_start(location_geojson: dict, begin: str, end: str):
    """Produces "start" dictionary corresponding to "itinerary (array)" label in the HAS Schema documentation:
    https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Hide-Activity-Specification

    Parameters
    ----------
    location_geojson : dict
        geojson dictionary format, represents location of the instruction
    begin : str, date-time format (ISO)
        beginning of time window
    end : str, date-time format (ISO)
        end of time window

    Returns
    -------
    dict
        start itinerary data
    """

    it_data = {
        "start": {
            "instruction_uid": str(uuid.uuid4()),
            "location": location_geojson,
            "time_window": {
                "begin": begin,
                "end": end
            }
        }
    }

    return it_data


def create_itinerary_move(location_geojson: dict, transport_mode: str):
    """Produces "move" dictionary corresponding to "itinerary (array)" label in the HAS Schema documentation:
    https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Hide-Activity-Specification

    Parameters
    ----------
    location_geojson : dict
        geojson dictionary format, represents location of the instruction
    transport_mode : str
        transportation mode. Can only be one of: ["personal_vehicle"] so far

    Returns
    -------
    dict
        move itinerary data
    """

    it_data = {
        "move": {
            "instruction_uid": str(uuid.uuid4()),
            "location": location_geojson,
            "transportation_mode": transport_mode,
        }
    }

    return it_data


def create_itinerary_stay(duration: str, end_time: str, priority: str = None):
    """Produces "stay" dictionary corresponding to "itinerary (array)" label in the HAS Schema documentation:
    https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Hide-Activity-Specification

    Parameters
    ----------
    duration : str
        length of stay; should be in ISO duration/timedelta format
    end_time : str
        time when the stay ends; should be in date-time ISO format
    priority : str
        indicates whether duration or end-time takes precedence

    Returns
    -------
    dict
        stay itinerary data
    """

    assert ((duration is not None) or (end_time is not None)), \
        "please provide either a duration or an end time"
    assert priority is not None if (duration is not None and end_time is not None) else True, \
        "priority must be provided if both duration and end time are provided"

    if priority is None and not (duration is None and end_time is None):
        priority = "duration" if duration is not None else "end_time"

    it_data = {
        "stay": {
            "instruction_uid": str(uuid.uuid4()),

        }
    }

    if duration is not None:
        it_data["stay"]["duration"] = duration
    if end_time is not None:
        it_data["stay"]["end_time"] = end_time
    if priority is not None:
        it_data["stay"]["priority"] = priority

    return it_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Random HAS generator',
        description='Generates random HAS files into a given directory. May or may not include a specified HOS file.')

    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help="Directory to output HAS files to. Does not need to exist.")
    parser.add_argument(
        '--hos',
        default=None,
        help="Path of HOS file, optional. If given, generated HAS will include trajectories which are compliant to the given HOS.")
    parser.add_argument(
        '-p',
        '--poi-pickle-path',
        default=None,
        help="Path of POI pickle file file, required if HOS file is given. This file (typically poi.pkl) holds the geometry information of buildings to sample from.")
    parser.add_argument(
        '-s',
        '--stop-point-path',
        default=None,
        help="Path of stop point parquet files, required if HOS file is given. This file (typically StopPoints.parquet) holds the existing table of simulation stop points.")
    parser.add_argument('-n', '--num-generated',
                        type=int,
                        default=10,
                        help="Number of random HAS to generate. Default: 10")
    parser.add_argument(
        '--internal',
        action="store_true",
        default=False,
        help="Include flag to generate with internal HAS format.")

    args = parser.parse_args()

    # python haystac_ta1/haystac_kw/utils/specifications/generate_random_has.py \
    # --poi-pickle-path ~/citydata/hos_app_files/poi.pkl \
    # --stop-point-path ~/haystac/behaviorsim/stop_points/KitwareStopPoints.parquet \
    # --output baseline_iHOS \
    # --hos ~/gov_data/baseline_hos/hos_17b479c2-3f2b-15f3-8ca9-cc00ca44daa2.json \
    # -n 3 \
    # --internal
    
    create_n_random_has(
        args.poi_pickle_path,
        output_dir=args.output,
        n=args.num_generated,
        hos_file=args.hos,
        stop_point_file=args.stop_point_path,
        internal_format=args.internal)
