"""
Takes a HAS file and a HOS file and validates the HAS meets the HOS
"""
import datetime
import pandas as pd
from rich import print
from haystac_kw.data.schemas.hos import HOS, InternalHOS
from haystac_kw.data.schemas.has import HAS, InternalHAS
from haystac_kw.data.types.event import EventType
from haystac_kw.data.types.itinerary import Start, Stay, Move
from geopandas import GeoDataFrame, GeoSeries
from typing import NamedTuple, Dict, List, Union, Optional
from shapely import Point
import numpy as np
from itertools import product
from haystac_kw.ta1.agent_behavior_sim.map import OfflineMap
from haystac_kw.utils.data_utils.road_network import map_buildings
import pickle
from collections import defaultdict

# this is the format that is used in the specification
__time_format = "%Y-%m-%dT%H:%M:%SZ"
__ignore_z = False


class CandidateHOSEvent(NamedTuple):
    event_type: EventType
    timestamp: datetime.datetime
    event_uuid: str
    stay_duration: datetime.timedelta

###############

def extract_trajectory_hos_events(trajectory: GeoDataFrame, hos: HOS) -> Dict[str, List[CandidateHOSEvent]]:
    """
    Given a trajectory, grab all Arrival/Departure Events
    that are candidates for fulfilling the HOS. Checks
    if event_type is correct but does not check time constraints.

    Parameters
    ----------
    trajectory : GeoDataFrame
        GeoDataFrame with timestamp and geomtery columns
    hos : HOS
        HOS we are trying to satisfy

    Returns
    -------
    Dict[str, List[HOSEvent]]
        All arrival/departure events in the trajectory relevant to
        each event in the HOS. The key to the dictionary is the
        event_uid.
    """


    # output data structure
    hos_events = {}
    for event in hos.events:
        # Iterate through each event in the HOS

        # Get the location for the event
        geom = GeoSeries(event.location, crs=trajectory.crs).values[0].geoms[0]
        if isinstance(geom, Point):
            # location is a point, buffer it
            # TODO: do this in cartesian coords
            geom = geom.buffer(0.001) # ~2m

        # Grab objective id and the event type
        event_uuid = event.event_uid
        event_type = event.event_type
        hos_events[event_uuid] = []

        # Get every point in time where trajectory is within
        # the HOS polygon, the resulting array will have a
        # value of 1 for every point that is in the polygon
        # and 0 when not within the polygon.
        points_within = trajectory.within(geom).astype(int)

        # Find the transitions from outside polygon to within
        # polygon. The result will be an array with values
        # 1 when trajectory entered the polygon and -1 when
        # the trajectory exited the polygon.
        # NOTE: np.diff shrinks array by 1 so increment to
        # get index in the timestamp array.
        transitions = np.diff(points_within)
        arrive_indices = np.argwhere(transitions>0) + 1
        depart_indices = np.argwhere(transitions<0) + 1
        durations = []

        if event_type.lower() == 'arrive':
            # Get indices of arrival events
            indices = arrive_indices

            # Grab duration of each stay after `arrive` event
            # this is necessary because some duration constraints
            # can have a `stay` flag.
            for arrive_index in indices:
                departs = depart_indices[depart_indices>arrive_index]
                if len(departs) == 0:
                    durations.append(None)
                else:
                    t0 = trajectory.timestamp.values[arrive_index]
                    t1 = trajectory.timestamp.values[np.min(departs)]
                    durations.append(t1-t0)

        else:
            # Get indices of departure events
            indices = depart_indices
            durations = [None]*len(indices)
        # `indices` contains all indices of timestamps
        # with candidate events for the this event in HOS.
        indices = indices.flatten()

        for i, index in enumerate(indices):
            # Record HOS event
            hos_events[event_uuid].append(
                CandidateHOSEvent(
                    event_type=event_type,
                    event_uuid=event_uuid,
                    timestamp=trajectory.timestamp.values[index],
                    stay_duration=durations[i]
                )
            )

    return hos_events


def is_hos_satisifed(hos: Union[HOS, InternalHOS], candidate_hos_events: Dict[str, List[CandidateHOSEvent]]) -> bool:
    """
    Checks if a HOS is satisified by a collection of candidate events.

    Parameters
    ----------
    hos : HOS
        HOS being validated against
    hos_events : Dict[str, List[CandidateHOSEvent]]
        Collection of candidate hos events. The key to the dictionary is
        an event_uid. The value in the dictionary is a list of
        CandidateHOSEvent. The only assumptions are that the
        CandidateHOSEvent's in the list occured at the correct location for the
        event specified by event_uid and are the correct type (i.e. arrive or
        depart). Elements may have been included the do not satisfy time window
        or duration constraints, as these will be check herein.

    Returns
    -------
    bool
        Whether or not HOS is satisfied by **candidate_hos_events**
    """

    # First pass just check if each event in the HOS
    # is even present in hos_events
    for event in hos.events:
        event_uuid = event.event_uid
        if event_uuid not in candidate_hos_events:
            # Event not in hos_events, so no candidates
            # HOS can't be satisfied
            return False
        if len(candidate_hos_events[event_uuid])==0:
            # Candidate list is empty
            # HOS can't be satisfied
            return False

    # At this point there is at least one candidate
    # event in the trajectory for each required event
    # in the HOS

    # Next filter each candidate list for events within
    # time constraints.
    for time_contraint in hos.time_constraints:

        # Get time window for event
        time_start = time_contraint.time_window.begin
        time_end = time_contraint.time_window.end

        # Filter candidate events by time window
        event_uuid = time_contraint.event
        candidate_hos_events[event_uuid] = \
        [x for x in candidate_hos_events[event_uuid] if x.timestamp>=time_start]
        candidate_hos_events[event_uuid] = \
        [x for x in candidate_hos_events[event_uuid] if x.timestamp<=time_end]

        if len(candidate_hos_events[event_uuid]) == 0:
            # Candidate list is now empty
            # HOS can't be satisfied
            return False

    # At this point we have candidate events for every
    # event in the HOS, and they are in the correct time
    # window. Now we need to check every combindation for
    # a version that satisfies the duraion constraints

    # Build data structure for iterating through candidates
    events = list(candidate_hos_events.keys())
    candidate_options = [list(range(len(candidate_hos_events[x]))) for x in events]

    # Iterate through every combination of candidates
    for combination in product(*candidate_options):

        # Candidate events dict[event_uuid] = candidate_HOS_event
        candidate = {events[i]:candidate_hos_events[events[i]][x] for i,x in enumerate(combination)}

        # Flag indicating that the duration constraints
        # are satisfied, we will set it to false if this
        # combination doesn't work
        durations_satisfied = True
        for constraint in hos.duration_constraints:
            if not durations_satisfied:
                # One of the previous constraints wasn't
                # satisfied so we can skip checking this one
                continue

            # Get event_uuid for constraint
            event_1 = constraint.events.first
            event_2 = constraint.events.second

            # Get time period between the two events
            t1 = candidate[event_1].timestamp
            t2 = candidate[event_2].timestamp
            duration = t2-t1
            agent_stayed_at_first_location = \
                duration == candidate[event_1].stay_duration

            if duration > constraint.duration_window.maximum:
                # Duration was longer than allowed
                durations_satisfied = False
                continue
            if duration < constraint.duration_window.minimum:
                # Duration was shorter than allowed
                durations_satisfied = False
                continue
            if constraint.stay and not agent_stayed_at_first_location:
                # Agent didn't stay at event_1's location
                # but was required to
                durations_satisfied = False
                continue

        if durations_satisfied:
            # We found a combination of candidate events
            # that satisfied the HOS
            return True

    # At this point no combindation of candidate events
    # satisifed the HOS so return False.
    return False


def robust_validate_ihas(hos: InternalHOS, has: InternalHAS,
                         road_network: Optional[str]=None,
                         poi: Optional[str]=None,
                         num_iter: int=1000) -> float:
    """
    Checks how robustly a HOS is satisified by a HAS.

    Parameters
    ----------
    hos : HOS
        HOS being validated against.
    has : HAS
        HAS being validated.
    road_network : str | None
        Path to a pickled version of . This allows accurate sampling of travel
        time between points on the map. If None, travel times will be randomly
        sampled between 0-20 seconds.
    poi : str | None
        Path to a pickled version of  used to load in a model of the map and
        road networks.
    num_iter : int
        Number of Monte Carlo simulations. Each simulation represents one
        possible manifestion of HAS instructions being followed given random
        variations in start time and travel time.

    Returns
    -------
    float
        Robustness score between 0-1.
    """

    # Each HAS is dedicated to one HOS. In the general case, one HAS can have
    # many different movement instructions across many different agents, so we
    # have to check across the entire sequence whether the HOS is satisfied.

    # TODO waiting for answer from gov about whether we can assume that HAS
    # with multiple 'itineraries' entries for one agent will occur in the order
    # they appear in 'itineraries'.

    if road_network is not None:
        road_network = pickle.load(open(road_network, 'rb'))
        road_map = OfflineMap(xgraph=road_network, poi=poi)
        # Mapping from HAS locations to index in map poi
        is_road_edge = defaultdict(lambda: True)
        for poi_id in road_map.poi.index.values:
            is_road_edge[poi_id] = False
    else:
        print('No road network provided, assuming random travel times.')

    # TODO this only works for single-agent movements.
    agents = set()
    for event in hos.events:
        agents = agents.union(set(event.agents))

    assert len(agents) == 1
    agent = list(agents)[0]

    # Simulate a time series of events that could occur given the HAS.
    successes = []
    for _ in range(num_iter):
        # Each loop here is one incarnation of possible outcome of how a TA-1
        # might turn a HAS into a time series of events.
        candidate_hos_events = {hos_event.event_uid:[]
                                for hos_event in hos.events}

        def process_event(timestamp, location, event_type, stay_duration=None):
            """
            Check this event against all required HOS events, and if at the
            right location and of the right type, added to the candidate list.
            """
            # We have to check every required hos event to see if this even
            # could but what solves it.
            for hos_event in hos.events:
                if (hos_event.event_type == event_type and
                    hos_event.location == location):
                    candidate_hos_events[hos_event.event_uid].append(
                        CandidateHOSEvent(
                        event_type=event_type,
                        event_uuid=hos_event.event_uid,
                        timestamp=timestamp,
                        stay_duration=stay_duration
                        )
                    )

        # Loop over all movements specified by this HAS. There really should
        # only be one movement instance per agent.
        for movement in has.movements:
            # TODO: This all currently only works for one agent matching the
            # hos agent.
            assert agent == movement.agent, f"agent {agent} not matching movement {movement.agent}"

            # Loop over all itinerary for this agent.
            for itinerary in movement.itineraries:
                for i, instruction in enumerate(itinerary.itinerary):
                    if i == 0:
                        # First instruction must be a Start.
                        assert isinstance(instruction, Start), \
                            'First instruction of an itinerary must be a start'

                        # Uniformly sample the start time.
                        curr_time = instruction.time_window.sample()
                        curr_loc = instruction.location

                        # We just arrived at the start location. Cache this to
                        # be dealt with on the next move when we know what the
                        # stay duration becomes.
                        last_arrive = [curr_time, curr_loc]

                        # Set this to empty. This gets built up to encode
                        # intermediate road network edges that need to be used
                        # to eventually arrive at a destination.
                        route_edges = []

                        continue

                    if isinstance(instruction, Stay):
                        # We are going hang out at curr_loc for a bit longer.

                        # The following gets violated if a route was built up
                        # but never reached a move instruction including a real
                        # POI, which would have cleared 'route_edges'.
                        assert len(route_edges) == 0, 'Move instructions that ' \
                            'include intermediate road edges to define a ' \
                            'route must be end with a move to a POI location'

                        if (instruction.end_time is None or
                            instruction.priority.name == 'duration'):
                            # Enforce duration.
                            curr_time = curr_time + instruction.duration
                        else:
                            # Enforce time.
                            assert instruction.end_time is not None
                            if curr_time < instruction.end_time:
                                curr_time = instruction.end_time
                    elif isinstance(instruction, Move):
                        if (road_network is not None and
                            (is_road_edge[instruction.location] or
                             is_road_edge[curr_loc])):
                            # This is just an edge, we are building the route.
                            route_edges.append(instruction.location)
                        else:
                            # How long has this agent stated at this location
                            # before this impending move.
                            stay_duration = curr_time - last_arrive[0]
                            process_event(last_arrive[0], last_arrive[1], 'arrive',
                                        stay_duration)

                            process_event(curr_time, curr_loc, 'depart')

                            if road_network is not None:
                                # Get travel time in seconds.
                                travel_time = road_map.get_route_min_travel_time(curr_loc,
                                                                                 instruction.location,
                                                                                 edges=route_edges)
                            else:
                                # We don't have a road network, so we can't
                                # make a meaningful estimate of travel time.
                                # So, just choose a random time (seconds).
                                travel_time = float(np.random.rand(1)*20)

                            curr_time = curr_time + datetime.timedelta(seconds=travel_time)

                            # We have used the route_edges, if they were
                            # populated, to get to our destination, so we can
                            # clear them.
                            route_edges = []
                            curr_loc = instruction.location

                        # Cache this to be dealt with on the next move when we
                        # know what the stay duration becomes.
                        last_arrive = [curr_time, instruction.location]
                    else:
                        raise Exception(f'Should not have a '
                                        '{type(instruction)} object at this '
                                        'point')

                # We shouldn't have any dangling, not-dealt-with route edges by
                # this point.
                assert len(route_edges) == 0, 'Move instructions that ' \
                            'include intermediate road edges to define a ' \
                            'route must be end with a move to a POI location'

                # We had arrived at a location but we never left before giving
                # up control back to TA-1. Therefore, we commit the arrive but
                # with stay_duration None. If the HOS has duration constraint
                # with a stay between this arrival and a departure, which
                # hasn't happened yet but could be forced in a subsequent
                # itinerary, then the check will fail.
                process_event(last_arrive[0], last_arrive[1], 'arrive',
                              stay_duration=None)

        successes.append(is_hos_satisifed(hos, candidate_hos_events))

    return np.mean(successes)

################


def get_itinerary_id(itinerary: dict) -> str:
    """
    extracts the id of an itinerary item

    Parameters
    ----------
    itinerary : dict
        The dictionary representing a single itinerary item

    Returns
    -------
    str
        id for the itinerary
    """
    name = list(itinerary.keys())[0]
    id = itinerary[name]['instruction_uid']
    return id


def check_for_start(itinerary: dict) -> (bool, str):
    """
    Checks that the HAS itinerary begins with a start command as required.
    Per spec: movements.itineraries.itinerary.start must be fulfilled before
    other corresponding instructions

    Parameters
    ----------
    itinerary : dict
        The dictionary representing a set of itinerary items

    Returns
    -------
    tuple(bool, str)
        Pass/Fail of the check and string explanation
    """
    # get the first item info
    id = get_itinerary_id(itinerary[0])
    item_name = list(itinerary[0].keys())[0]
    if "start" != item_name:
        msg = f"ERROR: missing 'start' command for itinerary at '{id}', "
        msg += f"instead it starts with '{item_name}'"
        return (False, msg)
    else:
        return (True, "")


def get_spec_time(time_string: str, format=__time_format) -> datetime.datetime:
    """
    Converts the specific timestring to datetime format

    Parameters
    ----------
    time_string : str
        The string representing the time in the format from the spec.
    format : str
        The conversion format for the time strings

    Returns
    -------
    datetime.datetime
        The datetime object
    """

    # handle issues with internal has not having the Z
    global __ignore_z

    if time_string == "":
        print("   WARNING: bad time string")
        time_string = "1900-01-01T00:00:00Z"
    if "Z" not in time_string and __ignore_z:
        time_string += "Z"
    return datetime.datetime.strptime(time_string, format)


def check_handoff_time(itinerary: dict, min_minutes=60) -> (bool, str):
    """
    Checks that the HAS itinerary start begin is at least 1 hour before the end
    Per spec: To help alleviate handoff issues with agents reaching the
    movements.itineraries.itinerary.start.location, each movements.itineraries.
    itinerary.start.time_window.begin should be at least 1 hour before the
    corresponding movements.itineraries.itinerary.start.time_window.end.
    This interval may change in future trial periods.

    Parameters
    ----------
    itinerary : dict
        The dictionary representing a set of itinerary items
    min_minutes : int
        The number of minutes that are required between begin and end

    Returns
    -------
    tuple(bool, str)
        Pass/Fail of the check and string explanation
    """

    # check the first item's times since this only applies to start
    id = get_itinerary_id(itinerary[0])
    name = list(itinerary[0].keys())[0]
    begin = get_spec_time(itinerary[0][name]['time_window']['begin'])
    end = get_spec_time(itinerary[0][name]['time_window']['end'])
    mins = (end - begin).total_seconds() / 60.0
    if mins < min_minutes:
        msg = f"ERROR: start command '{id}' has a start.begin that is less "
        msg += f"than the required {min_minutes} minutes (it was {mins}m)"
        return (False, msg)
    else:
        return (True, "")


def is_location_entered(location: str, itinerary: dict) -> (bool, str, str):
    """
    Tests if a location is entered with this itinerary

    Parameters
    ----------
    location : str
        The location to check on
    itinerary : dict
        The dictionary representing a set of itinerary items

    Returns
    -------
    tuple(bool, str, str)
        Did we enter and string for time entered and time left
    """
    entered = False
    enter_time1 = ""
    enter_time2 = ""
    last_time1 = ""
    last_time2 = ""
    leave_time1 = ""
    leave_time2 = ""
    for di in itinerary:
        cmd = list(di.keys())[0]
        # keep track of the time for the next item
        if "time_window" in di[cmd].keys():
            # grab the beginning and end of the window
            last_time1 = di[cmd]['time_window']['begin']
            last_time2 = di[cmd]['time_window']['end']
        if "end_time" in di[cmd].keys() and \
                "duration" in di[cmd].keys():
            # if both are present then we need to check for
            # the priority item
            end_time = True  # assume end_time in case they left it out
            if "priority" not in di[cmd].keys():
                id = di[cmd]['instruction_uid']
                msg = "ERROR: both 'end_time' and 'duration' were found in "
                msg += f"a 'stay' instruction without a 'priority' ({id})."
                print(msg)
            else:
                if di[cmd]['priority'].lower() == "duration":
                    end_time = False
            if end_time:
                # set both of these since the time is specific
                last_time1 = last_time2 = di[cmd]['end_time']
            else:
                # grab the seconds in the duration
                secs = get_duration_seconds(di[cmd]['duration'])
                # handle both time slots since this could be affecting
                # two different starts
                ltime1 = get_spec_time(last_time1)
                ltime1 += datetime.timedelta(seconds=secs)
                last_time1 = ltime1.strftime(__time_format)
                ltime2 = get_spec_time(last_time2)
                ltime2 += datetime.timedelta(seconds=secs)
                last_time2 = ltime2.strftime(__time_format)
        elif "end_time" in di[cmd].keys():
            last_time1 = last_time2 = di[cmd]['end_time']
        # use a duration if it is present
        elif "duration" in di[cmd].keys():
            secs = get_duration_seconds(di[cmd]['duration'])
            ltime1 = get_spec_time(last_time1)
            ltime1 += datetime.timedelta(seconds=secs)
            last_time1 = ltime1.strftime(__time_format)
            ltime2 = get_spec_time(last_time2)
            ltime2 += datetime.timedelta(seconds=secs)
            last_time2 = ltime2.strftime(__time_format)
        # skip commands with not location
        if "location" not in di[cmd].keys():
            continue
        # check on the location
        if di[cmd]["location"] == location:
            entered = True
            if cmd == "start":
                # if it was from a start command then get
                # the beginning of the window
                enter_time1 = di[cmd]["time_window"]["begin"]
                enter_time2 = di[cmd]["time_window"]["end"]
            else:
                enter_time1 = last_time1
                enter_time2 = last_time2
        # if it was entered, check to see when we left
        if entered:
            if di[cmd]["location"] != location:
                leave_time1 = last_time1
                leave_time2 = last_time2
                break  # no longer need to check other items

    return (entered, enter_time1, enter_time2, leave_time1, leave_time2)


def get_duration_seconds(duration: str) -> int:
    """
    Takes a duration from the specification in the format 'P0DT0H0M0S'
    and converts it to seconds

    Parameters
    ----------
    duration : str
        The string representing the duration

    Returns
    -------
    int
        number of seconds
    """

    td = pd.to_timedelta(duration)
    #print(f"duration = {td}")
    #print(f"sec = {td.seconds}")

    return td.seconds


def validate_has(has: dict, hos: dict, verbose=False, ignorez=False) -> bool:
    """
    Tests a HAS (Hide Activity Specification) to see if it matches a
    HOS (Hide Objective Specification)

    Parameters
    ----------
    has : dict
        The HAS to check

    hos: dict
        The HOS to check against

    verbose : bool
        Whether to print error messages on the checks failures

    ignorez : bool
        ignores the need for the Z at the end of the time

    Returns
    -------
    bool
        Pass/Fail of the check
    """

    # handle the case where the HAS has no Z on timestamps
    global __ignore_z
    __ignore_z = ignorez

    # create a mapping for events from the HOS
    hos_map = {}
    required_agents = []
    for event in hos['events']:
        uid = event['event_uid']
        hos_map[uid] = {}
        hos_map[uid]['type'] = event['event_type']
        hos_map[uid]['location'] = event['location']
        hos_map[uid]['agents'] = event['agents']
        hos_map[uid]['entered'] = False  # track wether we have entered the location
        hos_map[uid]['left'] = False  # track if we should be done with this event
        hos_map[uid]['errors'] = []  # keep track of errors with this event
        hos_map[uid]['enter_time1'] = ""
        hos_map[uid]['left_time1'] = ""
        hos_map[uid]['enter_time2'] = ""
        hos_map[uid]['left_time2'] = ""
        for agent in event['agents']:
            if agent not in required_agents:
                required_agents.append(agent)

    # add the time constraint info to the event map
    for tc in hos['time_constraints']:
        uid = tc['event']
        if uid not in hos_map.keys():
            print(f"ERROR: missing event uid = {uid}")
            continue
        hos_map[uid]['time_window'] = tc['time_window']

    # check the HAS movements
    all_passed = True
    agent_set = []
    for m_event in has['movements']:
        # get the current agent for this set of itins
        agent = m_event['agent']
        if agent not in agent_set:
            agent_set.append(agent)

        # check some general items
        for iten in m_event['itineraries']:
            # check that there is a start at the beginning of each
            passed, err = check_for_start(iten['itinerary'])
            if not passed:
                print(err)
                all_passed = False

            # check the handoff time for all starts
            passed, err = check_handoff_time(iten['itinerary'])
            if not passed:
                print(err)
                all_passed = False

            # check the agent matches those in the hos
            found_agent = False
            for uid in hos_map:
                if agent in hos_map[uid]['agents']:
                    found_agent = True
            if not found_agent:
                print(f"unable to match agent '{agent}' to HOS")
                all_passed = False

        # walk through itirearies and fill in the hos map
        for uid in hos_map:
            # first match based on agent
            if agent in hos_map[uid]['agents']:
                # has the location been entered?
                for iten in m_event['itineraries']:
                    # find out if this location is entered
                    entered, et1, et2, lt1, lt2 = \
                        is_location_entered(hos_map[uid]['location'],
                                            iten['itinerary'])
                    if entered:
                        hos_map[uid]['entered'] = True
                        hos_map[uid]['enter_time1'] = et1
                        hos_map[uid]['enter_time2'] = et2
                        # save the leave time if it was there
                        if lt1 != "":
                            hos_map[uid]['left'] = True
                            hos_map[uid]['left_time1'] = lt1
                            hos_map[uid]['left_time2'] = lt2
    if verbose:
        print("Internal mapping of HOS events:")
        print(hos_map)

    # work through the checks of the HOS map
    for uid in hos_map:
        etype = hos_map[uid]['type']
        # checks specific to arrive type
        if etype == 'arrive':
            # see if all the hos_map items were entered
            if hos_map[uid]['entered'] is False:
                msg = f"ERROR: HOS '{uid}' was never entered by "
                msg += f"agents {hos_map[uid]['agents']}."
                print(msg)
                all_passed = False

        # checks for depart types
        if etype == 'depart':
            # see if the item was ever exited
            if hos_map[uid]['left'] is False:
                msg = f"ERROR: HOS '{uid}' was never exited, a "
                msg += "depart is required by the HOS."
                all_passed = False
                print(msg)

        # check the duration requirements
        for tc in hos['time_constraints']:
            # match based on uid
            if tc['event'] == uid:
                # see if we should check arrive or depart
                if etype == 'arrive':
                    check_time1 = hos_map[uid]['enter_time1']
                    check_time2 = hos_map[uid]['enter_time2']
                else:
                    check_time1 = hos_map[uid]['left_time1']
                    check_time2 = hos_map[uid]['left_time2']
                # now see if this time was in between the required
                bt = get_spec_time(tc['time_window']['begin'])
                et = get_spec_time(tc['time_window']['end'])
                if check_time1 == "" and check_time2 == "":
                    continue
                ct1 = get_spec_time(check_time1)
                ct2 = get_spec_time(check_time2)
                if (ct1 < bt or ct1 > et) and (ct2 < bt or ct2 < et):
                    msg = f"ERROR: time constraints for '{uid}' were "
                    msg += f"not met:\n  HOS requires '{etype}' to happen "
                    msg += f" between {bt} and {et} (vs {ct1}, {ct2})"
                    print(msg)
                    all_passed = False

    # step through the duration constraints
    for dc in hos['duration_constraints']:
        # get the events
        fe = dc['events']['first']
        se = dc['events']['second']
        # get the window of time
        wmin = get_duration_seconds(dc['duration_window']['minimum'])
        wmax = get_duration_seconds(dc['duration_window']['maximum'])

        # stay means we should hit the time window before leaving
        duration1 = 0
        duration2 = 0
        if dc['stay']:
            # find the movements that match
            for uid in hos_map:
                if uid == fe:
                    # get the time window to check
                    et1 = get_spec_time(hos_map[uid]['enter_time1'])
                    lt1 = get_spec_time(hos_map[uid]['left_time1'])
                    duration1 = (lt1 - et1).total_seconds()
                    et2 = get_spec_time(hos_map[uid]['enter_time2'])
                    lt2 = get_spec_time(hos_map[uid]['left_time2'])
                    duration2 = (lt2 - et2).total_seconds()
        else:
            # when not in stay we need to check the beginning of the first
            # add the beginning of the second movement
            start_time1 = ""
            end_time1 = ""
            start_time2 = ""
            end_time2 = ""
            for uid in hos_map:
                if uid == fe:
                    if hos_map[uid]['type'].upper() == "ARRIVE":
                        start_time1 = hos_map[uid]['enter_time1']
                        start_time2 = hos_map[uid]['enter_time2']
                    else:
                        start_time1 = hos_map[uid]['left_time1']
                        start_time2 = hos_map[uid]['left_time2']
                if uid == se:
                    end_time1 = hos_map[uid]['enter_time1']
                    end_time2 = hos_map[uid]['enter_time2']
            # get the duration in seconds
            et1 = get_spec_time(start_time1)
            lt1 = get_spec_time(end_time1)
            duration1 = (lt1 - et1).total_seconds()
            et2 = get_spec_time(start_time2)
            lt2 = get_spec_time(end_time2)
            duration2 = (lt2 - et2).total_seconds()
            # print(duration)

        # catch the case that we have a negative
        # so we can print a NaN in the duration instead of
        # abad value
        na_duration = False
        if duration1 < 1 and duration2 < 1:
            na_duration = True

        # now see if we matched the duration constraint
        if (duration1 < wmin or duration1 > wmax) and (
                duration2 < wmin or duration2 > wmax):
            msg = f"ERROR: Duration for {fe}+{se} not met:\n"
            msg += f"  should be between {wmin/60}m and {wmax/60}m "
            if na_duration:
                msg += "(vs NaN - possibly never departed?)"
            else:
                msg += f"(vs {duration1/60}m, {duration2/60})"
            print(msg)
            all_passed = False

    # return if all passed
    return all_passed


if __name__ == "__main__":
    """
    This section of code has tests for the validation module
    """

    """
    import json
    # import files for test
    hasf = "./data/test_has.json"
    hosf = "./data/baseline_hos_internal/hos_3f124e1b-c119-d433-b400-e189b1eee331.json"
    has = {}
    with open(hasf, 'r') as hf:
        has = json.load(hf)
    hos = {}
    with open(hosf, 'r') as hf:
        hos = json.load(hf)
    """

    # bare minimum HOS for testing
    hos = {
        "events": []
    }
    ev1 = {
        "event_uid": "0ec46957-7e72-c9e9-b0f3-df571367140e",
        "agents": [
            1729
        ],
        "event_type": "arrive",
        "location": "7d583716-bc44-4fec-a643-053f6c020db1"
    }
    hos["events"].append(ev1)
    ev2 = {
        "event_uid": "e6458200-cea4-0b8b-33b6-e2457fcdf526",
        "agents": [
            1729
        ],
        "event_type": "depart",
        "location": "7d583716-bc44-4fec-a643-053f6c020db1"
    }
    hos["events"].append(ev2)

    # leave these blank for now
    hos['time_constraints'] = []
    hos['duration_constraints'] = []

    # test 1 a HAS with a missing start command
    has = {
        "movements": [
            {
                "agent": 123,
                "itineraries": [
                    {
                        "itinerary": [
                            {
                                "move": {
                                    "location": "X",
                                    "instruction_uid" : "0a900366-should-fail-id"
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }

    # this should fail
    #print("fail based on start command missing")
    #res = validate_has(has, hos)
    #assert res is False

    # add the correct start for the has
    itins1 = {
        "itinerary": [
        {
            "start": {
                "instruction_uid": "71e2556f-f024-46e0-aee1-45019b569e39",
                "location": "7d583716-bc44-4fec-a643-053f6c020db1",
                "time_window": {
                    "begin": "2023-01-17T21:00:00Z",
                    "end": "2023-01-17T21:30:00Z"
                }
            }
        },
        {
            "stay": {
                "instruction_uid": "abb6a10b-9ece-4e47-96e0-f320cf22478d",
                "end_time": "2023-01-18T03:00:00Z"
            }
        },
        ]
    }
    has['movements'][0]['itineraries'][0] = itins1
    #print(has)

    # this should throw an error on the start hour constraint and
    # agent not matching
    print("====== fails based on start hour constraint and agent ======")
    res = validate_has(has, hos)
    assert res is False

    # fix the agent in this test HAS and the start time
    has['movements'][0]['agent'] = 1729
    itins1['itinerary'][0]['start']['time_window']['end'] = \
        "2023-01-17T22:30:00Z"
    # this should fail because of the departure is missing
    print("====== fails based departure ======")
    res = validate_has(has, hos)
    assert res is False

    # add the departure in
    it1 = {
        "move": {
            "instruction_uid": "5312bf33-411c-47eb-875e-e3a4933e7122",
            "transportation_mode": "personal_vehicle",
            "location": "e6458200-cea4-0b8b-33b6-e2457fcdf526"
        }
    }
    itins1['itinerary'].append(it1)

    # add the time constraint in
    time_c1 = [
        {
            "event": "0ec46957-7e72-c9e9-b0f3-df571367140e",
            "time_window": {
                "begin": "2023-05-15T06:00:00Z",
                "end": "2023-05-15T18:00:00Z"
            }
        }
    ]
    hos['time_constraints'] = time_c1

    # this should error out on the time constraints
    print("====== fails on time constraints ======")
    res = validate_has(has, hos)
    assert res is False

    # fix the time constraint
    hos['time_constraints'][0]['time_window']['begin'] = "2023-01-17T20:00:00Z"
    hos['time_constraints'][0]['time_window']['end'] = "2023-01-17T22:00:00Z"

    # add a new event requirement to HOS
    ev3 = {
        "event_uid": "7971da61-d85c-264b-7cb5-698f7ab4b2d7",
        "agents": [
            1729
        ],
        "event_type": "arrive",
        "location": "7a3cbb63-55ca-40dc-8756-6fe620ca192c"
    }
    hos["events"].append(ev3)

    # new time constraint to check with new location
    tc1 = {
        "event": "7971da61-d85c-264b-7cb5-698f7ab4b2d7",
        "time_window": {
            "begin": "2023-01-18T08:00:00Z",
            "end": "2023-01-18T09:00:00Z"
        }
    }
    hos['time_constraints'].append(tc1)

    # add some duration constraints
    dc1 = {
        "events": {
            "first": "0ec46957-7e72-c9e9-b0f3-df571367140e",
            "second": "e6458200-cea4-0b8b-33b6-e2457fcdf526"
        },
        "duration_window": {
            "minimum": "P0DT1H0M0S",
            "maximum": "P0DT3H0M0S"
        },
        "stay": True
    }
    hos['duration_constraints'].append(dc1)

    # add an itinerary item to arrive at the missing location
    it2 = {
        "stay": {
            "instruction_uid": "123456788899911111",
            "end_time": "2023-01-18T09:00:00Z"
        }
    }
    itins1['itinerary'].append(it2)
    it3 = {
        "move": {
            "instruction_uid": "1234567888999",
            "transportation_mode": "personal_vehicle",
            "location": "7a3cbb63-55ca-40dc-8756-6fe620ca192c"
        }
    }
    itins1['itinerary'].append(it3)

    # this should fail based on the duration constraint
    print("====== fails on duration constraint ======")
    res = validate_has(has, hos)
    assert res is False

    # fix that contraint failure
    hos['duration_constraints'][0]['duration_window']["maximum"] = "P0DT6H0M0S"

    # add an additional constraint without the stay option
    dc2 = {
        "events": {
            "first": "0ec46957-7e72-c9e9-b0f3-df571367140e",
            "second": "7971da61-d85c-264b-7cb5-698f7ab4b2d7"
        },
        "duration_window": {
            "minimum": "P0DT1H0M0S",
            "maximum": "P0DT3H0M0S"
        },
        "stay": False
    }
    hos['duration_constraints'].append(dc2)

    # this should fail based on the new constraint
    print("====== fails on duration constraint ======")
    res = validate_has(has, hos, verbose=False)
    assert res is False

    # fix that contraint failure
    hos['duration_constraints'][1]['duration_window']["maximum"] = "P0DT12H0M0S"
    # this should fail based on the new constraint
    print("====== should be passing ======")
    res = validate_has(has, hos, verbose=False)
    if res:
        print("passed!")
    assert res is True
