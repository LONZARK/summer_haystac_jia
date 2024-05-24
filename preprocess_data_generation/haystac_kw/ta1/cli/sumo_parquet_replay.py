#!/usr/bin/env python
"""
Replay parquet files as moving vehicles on top of a simulation (or empty network)
(Note: this was adapted from fcdReplay.py)
"""

import os
import sys
from glob import glob
import pandas as pd
from os.path import basename
from tqdm import tqdm
from collections import defaultdict
import libsumo
import time
from multiprocessing import Pool
sys.path.append(os.path.join(os.environ["SUMO_HOME"], 'tools'))
import sumolib  # noqa

# TODO: make these options passed into the script in the future
_skip_to_second = 0  # 22100.0  # 0
_render_vehicles = True
_render_pois = False
_max_records_to_read = 3600  # set to -1 for all

# this will be used to count the tick time of the simulation vs. the
# unix time that the parquet file has in it
_first_time = 0  # beginning time (unix time) of the parquet data


class SimObject:
    """This class mimics an internal sumo class object - made for convenience
    in passing info around
    """
    def __init__(self, id, x, y) -> None:
        self.id = id
        self.x = x
        self.y = y


def parse_options():
    """Get command line options

    Returns
    -------
    sumolib.options.ArgumentParser
        parsed arguments
    """
    # TODO: switch the format of this to click in the future?
    # for now we are just adapting another existing script
    parser = sumolib.options.ArgumentParser()
    parser.add_argument("-k", "--sumo-config",
                        category="input",
                        default="sumo.sumocfg",
                        help="sumo config file")
    parser.add_argument("-f", "--p-files",
                        category="processing",
                        dest="parquet_files",
                        help="the folder with the parquet files to replay")
    parser.add_argument("-v", "--verbose",
                        category="processing",
                        action="store_true",
                        default=False,
                        help="tell me what you are doing")
    parser.add_argument("sumo_args",
                        nargs="*",
                        catch_all=True,
                        help="additional sumo arguments")
    options = parser.parse_args()

    # add the option to disable teleports
    options.sumo_args.append("--time-to-teleport")
    options.sumo_args.append("-1")
    print(f"options.sumo_args: {options.sumo_args}")

    return options


def load_sumo():
    """Loads the sumo library and starts the gui

    Returns
    -------
    tuple(int, int)
        - deltaT : the delta step in time from the simulation
        - sim_time : the current time from the simulation
    """
    print("__Starting SUMO___")
    sumoBinary = sumolib.checkBinary("sumo-gui")
    libsumo.start([sumoBinary, "-c", options.sumo_config] + options.sumo_args)
    sim_time = libsumo.simulation.getTime()
    deltaT = libsumo.simulation.getDeltaT()
    return deltaT, sim_time


def get_parquet_data(params):
    """Handles loading parquet data for playback in the simulation.

    Parameters
    ----------
    params : tuple(str, float)
        filename and simulation start time

    Returns
    -------
    tuple(str, int, dict)
        - obj_id : the uid of the agent
        - end_time : the last time slot from this object
        - obj_dict : dictionary of object lat lon per time slot
    """
    global _max_records_to_read
    global _first_time
    fname, sim_start = params
    end_time = 0
    obj_id = 0
    obj_dict = {}

    # read in the parquet as a dataframe
    df = pd.read_parquet(fname)
    # get the max val if they set to -1
    if _max_records_to_read == -1:
        _max_records_to_read = len(df.index)

    id = basename(fname).split('.')[0]
    if options.verbose:
        print(f"Loading parquet data from {fname}")
    # for ts in sumolib.xml.parse(fname, 'timestep'):
    for index, row in df.iterrows():
        if index > _max_records_to_read:
            break
        # time = sumolib.miscutils.parseTime(ts.time)
        stime = row['timestamp'].timestamp()
        # grab the starting unix time
        if _first_time == 0:
            _first_time = stime
            print(f"  parquet starting time = {stime}")
        # convert unix time to timestep
        stime = (stime - _first_time) - _skip_to_second
        # print(f"saving timestep: {stime}")
        if stime < sim_start:
            continue
        x = row['longitude']
        y = row['latitude']
        # convert from lat lon to x y
        x, y = libsumo.simulation.convertGeo(x, y, True)
        obj = SimObject(id, x, y)
        obj_id = id
        end_time = stime
        if stime not in obj_dict.keys():
            obj_dict[stime] = []
        obj_dict[stime].append(obj)

    return (obj_id, end_time, obj_dict)


def load_data(options, deltaT, sim_start=0.0):
    """Loads data in parallel and prepares it for the simulation.

    Parameters
    ----------
    options : sumolib.options.ArgumentParser
        cli arguments TODO: change this to a path
    deltaT : int
        the delta time step for the simulation
    sim_start : float
        the time when the simulation should start from zero (Default value = 0.0)

    Returns
    -------
    tuple(dict, float, dict)
        - parquet_data : dictionary of lat lon data per agent
        - last_time : the last time slot in the simulation
        - removeAtTime : when agents are to be removed from the simulation
    """

    files = glob(options.parquet_files + "/*.parquet")  # options.parquet_files.split(',')
    files += glob(options.parquet_files + "/*/*.parquet")
    # print(f"found parquet files:\n{files}")

    print("__Loading parquet files__")

    items = [(x, sim_start) for x in files]
    with Pool(20) as p:
        return_data = list(tqdm(p.imap(get_parquet_data, items),
                                total=len(files)))

    # reassemble the data
    parquet_data = {}
    last_time = {}
    for x in return_data:
        obj_id, end_time, obj_dict = x
        for t in obj_dict:
            if t not in parquet_data.keys():
                parquet_data[t] = []
            parquet_data[t] += obj_dict[t]
        last_time[obj_id] = end_time

    print("loaded all parquet data")

    removeAtTime = defaultdict(list)  # time -> objects to remove
    for oID, t in last_time.items():
        removeAtTime[t + deltaT].append(oID)
        print(f"removing {oID} at {t + deltaT}")

    print(f"parquet_data keys: {len(parquet_data.keys())}")
    return (parquet_data, last_time, removeAtTime)


def run_simulation(lastTime: dict, removeAtTime, parquet_data):
    """Runs the simulation loop.

    Parameters
    ----------
    lastTime : dict
        dict of times per agent
    removeAtTime : dict
        dict for when agents are to be removed
    parquet_data : dict
        dict of lat lon info per agent
    """

    end = max(max(lastTime.values()), libsumo.simulation.getEndTime())
    created = set()

    attributes = {
        'type': 'DEFAULT_VEHTYPE',
        'speed': '0.0',
        'angle': '90.0',
        'slope': '0.00',
        'pos': "5.10",
        'lane': "639422447#0_0"
    }

    # settings for the vehicle moveToXY
    # https://sumo.dlr.de/pydoc/traci._vehicle.html
    # If keepRoute is set to 2 the vehicle has all the freedom of keepRoute=0
    # but in addition to that may even move outside the road network.
    # edgeID and lane are optional placement hints to resolve ambiguities
    keepRoute = 2
    edgeID = ''  # '639422447#0'
    lane = 0
    # the route has to be valid for it to start with something
    fake_route = [
        '641549552',
        '481879064',
        '481879065'
    ]

    print("__Loading simulation__")
    t = libsumo.simulation.getTime()
    while t <= end:
        stime = time.time()
        if t in parquet_data.keys():
            print(f"  simulating timestep {t + _skip_to_second}")
            for obj in parquet_data[t]:
                if obj.id in created:
                    if _render_pois:
                        libsumo.poi.setPosition(obj.id, obj.x, obj.y)
                    # print(f"id {obj.id} set position: {obj.x}, {obj.y}")
                    if _render_vehicles:
                        libsumo.vehicle.moveToXY(obj.id, edgeID=edgeID,
                                                 laneIndex=lane,
                                                 x=obj.x, y=obj.y,
                                                 keepRoute=keepRoute)
                else:
                    created.add(obj.id)
                    if _render_pois:
                        libsumo.poi.add(obj.id, obj.x, obj.y, (255, 0, 0, 255))
                    if _render_vehicles:
                        libsumo.route.add(obj.id, fake_route)
                        libsumo.vehicle.add(obj.id, obj.id)
                        # print(f"____ creating id: {obj.id}")
                        libsumo.vehicle.moveToXY(obj.id, edgeID=edgeID,
                                                 laneIndex=lane,
                                                 x=obj.x, y=obj.y,
                                                 keepRoute=keepRoute)
                if _render_pois:
                    # update the attribute of the item
                    for a in attributes.keys():
                        v = attributes[a]
                        libsumo.poi.setParameter(obj.id, a, v)
            for objID in removeAtTime.get(t, []):
                libsumo.poi.remove(objID)
                print(f"====== removing {objID} ========")

        libsumo.simulationStep()
        t = libsumo.simulation.getTime()
        # print(f"libsumo.poi ids: {libsumo.poi.getIDCount()}")
        print(f"  loop time: {time.time() - stime}")

    libsumo.close()


if __name__ == "__main__":
    options = parse_options()
    # load sumo - so we can use the conversion from lat/lon to x/y
    deltaT, sim_time = load_sumo()
    parquet_data, lastTime, removeAtTime = load_data(options, deltaT)
    # print(f"parquet_data keys: {parquet_data.keys()}")
    run_simulation(lastTime, removeAtTime, parquet_data)
