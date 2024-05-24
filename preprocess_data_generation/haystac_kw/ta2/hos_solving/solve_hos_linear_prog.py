from haystac_kw.data.schemas.hos import HOS
from haystac_kw.data.schemas.has import HAS
from haystac_kw.data.types.itinerary import Stay, Start, Move, Itinerary, Movements
from haystac_kw.data.types.event import EventObjective
from haystac_kw.data.types.time import TimeWindow
from uuid import uuid4
import numpy as np
from scipy.optimize import linprog
import datetime
import pytz
from typing import Dict, List
from haystac_kw.ta2.validation.validate_has import CandidateHOSEvent, is_hos_satisifed


def get_linear_prog_sol(hos: HOS):


    # Get time boundaries of the HOS
    min_time = np.inf
    max_time = -min_time
    for time_constraint in hos.time_constraints:
        min_time = min(time_constraint.time_window.begin.timestamp(), min_time)
        max_time = max(time_constraint.time_window.end.timestamp(), max_time)

    # Mapping from event_uid to events in the HOS
    emap = {x.event_uid: i  for i,x in enumerate(hos.events)}

    # Get number of events to satisfy
    n = len(hos.events)

    # Initialize array of timestamps to solve for
    t = np.zeros((n))

    # Initialize Inequality Constraints Matrix and Vector
    # A_ub@x <= b_bu
    A_ub = np.zeros((2*len(hos.time_constraints),n))
    b_ub = np.zeros((2*len(hos.time_constraints)))

    # Intialize Lower and upper bounds of timestamps
    l = np.zeros((n))
    l.fill(min_time)
    u = np.zeros((n))
    u.fill(max_time)

    for time_constraint in hos.time_constraints:
        # Grab the lower and upper bounds for all
        # of the events.
        min_time = time_constraint.time_window.begin.timestamp()
        max_time = time_constraint.time_window.end.timestamp()
        eid = emap[time_constraint.event]
        l[eid] = min_time
        u[eid] = max_time
    
    for i, duration in enumerate(hos.duration_constraints):
        
        i = 2*i
        
        # Getting the indices of the events in the constraint
        a = emap[duration.events.first]
        b = emap[duration.events.second]

        # Adding the upper bound of the duration constraint
        c = duration.duration_window.maximum.total_seconds()
        A_ub[i,b] = 1
        A_ub[i,a] = -1
        b_ub[i] = c

        i+=1 
        # Adding the lower bound of the duration constraint
        c = duration.duration_window.minimum.total_seconds()
        A_ub[i,b] = -1
        A_ub[i,a] = 1
        b_ub[i] = -c


    # Bounds of the times
    bounds = [(l[i], u[i]) for i in range(n)]

    # Run through the solver
    res = linprog(c=-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    return res


if __name__ == "__main__":


    from pathlib import Path


    hos_path = Path('/home/local/KHQ/cole.hill/Desktop/injection_viz/External HOS/hos_12f5023f-221a-42ac-8b0d-239a9eb11c65.json')
    hos = HOS.from_json(hos_path.read_text())

    res = get_linear_prog_sol(hos)
    
    candidate_events = {}
    for i, event in enumerate(hos.events):

        candidate_events[event.event_uid] = [
            CandidateHOSEvent(
                event_type = event.event_type,
                timestamp = datetime.datetime.fromtimestamp(res.x[i], tz=pytz.UTC),
                event_uuid = event.event_uid,
                stay_duration=None
            )
        ]
    
    events = [x[0] for x in candidate_events.values()]
    events = sorted(events, key=lambda x : x.timestamp)
    emap = {x.event_uid:x.location for x in hos.events}
    for i, event in enumerate(events[:-1]):
        if event.event_type=='arrive' and events[i+1].event_type=='depart':
            if emap[event.event_uuid] == emap[events[i+1].event_uuid]:
                stay_duration = events[i+1].timestamp - event.timestamp
                candidate_events[event.event_uuid] = [
                    CandidateHOSEvent(
                        event_type = event.event_type,
                        timestamp=event.timestamp,
                        event_uuid=event.event_uuid,
                        stay_duration=stay_duration
                    )
                ]
    

    print(is_hos_satisifed(hos, candidate_events))