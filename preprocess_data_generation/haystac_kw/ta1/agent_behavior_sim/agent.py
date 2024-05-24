#!/usr/bin/env python3

import time
import numpy as np
import uuid
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from scipy.stats import truncnorm, norm, rv_histogram, rv_discrete
from scipy.special import factorial
from collections import namedtuple

from haystac_kw.ta1.agent_behavior_sim.map import Map


class IrregularEvents:
    """Class for sampling spontaneous events for an Agents schedule. This class
    encapuslates a set of Rank-N locations representing the places that agent
    visits N'th most often (outside of their scheduled routed (e.g. home/work)).
    """
    def __init__(self, city_map: Map, top_k: int=4):
        """
        Constructor method.

        Parameters
        ----------
        city_map : Map
            Map containing all of the POI available in the simulation
        top_k : int, optional
            Number of locations to use (defaults to 4)
        """
        # A list of locations where the first element is the most-visited
        # location (outside of home and work), the second element is the
        # second-most-visited, etc.
        self.top_k_locs = [city_map.get_random_location()
                           for _ in range(top_k)]

        # Heuristic log-normal distribution for number of places visited per
        # day.
        mu = 1
        sigma = 0.5

        # Add 2 since we assume we are going to work and home already.
        # TODO figured out how we should be dealing with this.
        N = np.arange(top_k) + 2
        prob = np.exp(-(np.log(N)-mu)**2/(2*sigma))/N
        prob = prob/sum(prob)
        self.prob_num_places_per_day = rv_discrete(values=(N, prob))

        # Zipf’s law.
        prob = 1/N
        prob = prob/sum(prob)
        self.prob_rank_loc_per_day = rv_discrete(values=(np.arange(top_k),
                                                         prob))

        # Duration
        duration = np.linspace(20, 60*60*0.5, 100)
        prob = np.exp(-((duration-10*60)/(60*30))**2)
        prob = prob/sum(prob)
        self.duration = rv_discrete(values=(duration, prob))

    def sample(self):
        """Sample a random number of non-routine locations for an agent to
        visit in their schedule.

        Returns
        -------
        tuple(list, list)
            - locs : list of locations
                Specification of the location (edge id, meters distance along
                edge) to travel to.
            - durations : list of timedelta
                Duration to stay
        """
        N = self.prob_num_places_per_day.rvs()

        if N  == 0:
            return [], []

        loc_ind = set(self.prob_rank_loc_per_day.rvs(size=N))
        locs = [self.top_k_locs[i] for i in loc_ind]

        durations = [timedelta(seconds=s) for s in self.duration.rvs(size=N)]

        return locs, durations


class RegularScheduledEvent:
    """Encodes an event that tends to occur repeatedly on a schedule.
    
    In this context, a regularly scheduled event is a location that an agent
    goes to and departs from with some regular frequency (e.g., daily or
    certain days of the week at very similar times). Think of it as a class of
    events that an agent would (at least conceptually) have in their calendar
    as a reoccuring entry. An extreme case would be a strict 9-5 job, 5 days a
    week. However, we expect the agent will sometimes arrive early or late,
    possibly as a function of traffic.
    
    The timing of the schedule is in seconds since midnight.
    
    If the event occurs multiple times in a day, one instance of this object
    should be created for each of the blocks.
    
    If ``mean_arrival`` > ``mean_departure``, it is assumed that departure occurs
    on the next day. Therefore, this object does not support specification of
    events that occur for more than 24 hours.
    """
    def __init__(self, mean_arrival, mean_departure, arrival_std,
                 departure_std, days_of_week, location):
        """
        Constructor method

        Parameters
        ----------
        mean_arrival : float
            Mean arrival time (seconds since midnight).
        mean_departure : float
            Mean departure time (seconds since midnight). If value is less than
            ``mean_arrival``, it is assumed to be on the next day.
        arrival_std : float
            Standard deviation of the arrival time (seconds).
        departure_std : float
            Standard deviation of the departure time (seconds).
        days_of_week : list(bool)
            Days of the week in which the schedule is active. Should be a bool
            array of length 7 encoding whether the schedule is active for
            [Monday, Tuesday, ..., Sunday].
        location : tuple
            (edge id, distance along edge)
            Define the location of the event within the mobility network.
        """
        self.mean_arrival = mean_arrival
        self.mean_departure = mean_departure
        self.arrival_std = arrival_std
        self.departure_std = departure_std
        self.days_of_week = days_of_week
        self.location = location

    def sample(self):
        """Sample a day's schedule.

        Returns
        -------
        tuple(float, float)
            - arrival : Arrival time (seconds since midnight before the day being
                considered).
            - departure : Departure time (seconds since midnight before the day being
                considered). If greater than 86400, it means that the event
                wraps around from the day being scheduled to the next day.
        """
        #print('self.arrival_std', self.arrival_std)
        #print('self.departure_std', self.departure_std)
        arrival = np.random.normal(loc=self.mean_arrival, scale=self.arrival_std)[0]
        departure = np.random.normal(loc=self.mean_departure, scale=self.departure_std)[0]

        if self.mean_departure < self.mean_arrival:
            # It is the next day
            departure += 86400

        return arrival, departure


def schedule_packer(anchor_trips, desired_trips):
    """Schedule in desired trips around the core anchor trips.

    Parameters
    ----------
    anchor_trips : list(Trip)
        Foundation components of the schedule that desired_trips must work
        around.
    desired_trips : tuple(list, list)
        First element is the list of locations (edge id, distance along edge in
        meters) and the second element is a list of timedelta objects encoding
        the duration of the desired stay at each location.

    Returns
    -------
    list(Trip)
        Agent schedule with spontaneous and anchor trips.
    """
    if len(desired_trips) == 0:
        return anchor_trips

    daily_schedule = []

    if np.random.rand() > 0.75:
        # Add a stop before work.
        daily_schedule.append(desired_trips.pop())

    # Send them to work.
    daily_schedule.append(anchor_trips[0])

    # Deal with the desired trips.
    for trip in desired_trips:
        daily_schedule.append(trip)

    daily_schedule.append(anchor_trips[1])
    return daily_schedule


class Trip:
    """Representation of an agent's scheduled plans to visit a location.
    
    The context of a trip becomes active once the context of a previous
    trip is released by the current time satisfying the combination of
    ``depart_from_time``, ``min_stay_duration``, and ``max_stay_duration``.
    Therefore, the context of a trip becomes active while the agent is
    still at the previous trip's destination. If the current time when this
    trip becomes active exceeds ``max_depart_for_time``, then this trip
    becomes invalid and will be skipped. Otherwise, the agent will wait
    until ``min_depart_for_time`` to depart or will depart immediately if
    min_depart_for_time is None. Once the agent arrives at the destination,
    the trip defines how soon the agent can again depart from this location
    as controlled by the combination of ``depart_from_time``,
    ``min_stay_duration``, and ``max_stay_duration``.
    """
    __slots__ = ('destination', 'min_depart_for_time',
                 'max_depart_for_time', 'depart_from_time',
                 'min_stay_duration', 'max_stay_duration',
                 'next_trip_min_depart_overide')
    def __init__(self, destination, min_depart_for_time=None,
                 max_depart_for_time=None, depart_from_time=None,
                 min_stay_duration=None, max_stay_duration=None):
        """Constructor method

        Parameters
        ----------
        destination : (str, float)
            Specification of the location (edge id, meters distance along edge).
        min_depart_for_time : datetime, optional
            Earliest time to depart for the location. If None, the agent will
            leave immediately (defaults to None)
        max_depart_for_time : datetime, optional
            Latest time to depart for the location. If the agent doesn't leave
            within this time, this trip should be skipped (defaults to None)
        depart_from_time : datetime, optional
            Time to depart from the destination location if not overriden by
            min_stay_duration or max_stay_duration. If None, departure will
            defer to ``min_stay_duration`` (defaults to None)
        min_stay_duration : timedelta, optional
            Minimum time to stay at location. None is equivalent to zero
            duration (defaults to None)
        max_stay_duration : timedelta, optional
            Minimum time to stay at location. None is equivalent to an infinite
            duration (defaults to None)
        """
        self.min_depart_for_time = min_depart_for_time
        self.max_depart_for_time = max_depart_for_time

        if (self.max_depart_for_time is not None and
            self.min_depart_for_time is not None):
            assert self.max_depart_for_time >= self.min_depart_for_time

        self.depart_from_time = depart_from_time
        self.min_stay_duration = min_stay_duration
        self.max_stay_duration = max_stay_duration

        if (self.max_stay_duration is not None and
            self.min_stay_duration is not None):
            assert self.max_stay_duration >= self.min_stay_duration

        self.destination = destination
        self.next_trip_min_depart_overide = None

    def set_actual_arrival_time(self, t):
        """Call once the agent's time of arrival at the destination is known.

        Parameters
        ----------
        t : datetime
            Local time of the arrival.
        """
        # On the next trip, min_depart_for_time must not occur later than this
        # value.
        if self.max_stay_duration is not None:
            self.next_trip_min_depart_overide = self.max_stay_duration + t

        # Need to update the time to depart from this new
        # destination ('depart_from_time') conditioned on when
        # the agent actually arrived at this destination.
        if self.depart_from_time is None:
            if self.min_stay_duration is None:
                raise Exception('\'depart_from_time\' and '
                                '\'min_stay_duration\' can\'t both be None')
            else:
                self.depart_from_time = self.min_stay_duration + t
        elif self.min_stay_duration is not None:
            self.depart_from_time = max([self.depart_from_time,
                                         self.min_stay_duration + t])

        if self.max_stay_duration is not None:
            self.depart_from_time = min([self.depart_from_time,
                                         self.max_stay_duration + t])

    def __str__(self):
        string = ['Trip:\n']
        for attr in self.__slots__:
            string.append('%s: %s\n' % (attr, repr(getattr(self, attr))))

        return ''.join(string)

    def __repr__(self):
        return self.__str__()


class Agent:
    """When refering to an agent's schedule for a day, we consider the day to
    start when the agent wakes up and ends when the agent goes to sleep.
    
    All locations (e.g., home, primary, lane) are specified as a tuple of
    (net edge id, distance along edge) encoding a point on the road network.
    Agents can only move or reside at points along the road network.
    
    day_schedule is a list of Trip objects.
    """
    def __init__(self, home, primary, irregular_events,
                 location_label_map=None, uid=None, logger=None,
                 detect_stuck_agents=False):
        """
        Initialize Agent.

        Parameters
        ----------
        home : (edge id, distance along edge)
            Agent's home location.
        primary : (edge id, distance along edge)
            Agent's primary location (work, school, etc.).
        irregular_events : IrregularEvents
            Irregular events model.
        location_label_map : dict, optional
            Label mapping for certain locations for
            schedule visualization (defaults to None)
        uid : (str, int), optional
            Universal unique identifier of this agent, defauls to None
        logger :  logging.Logger, optional
            Logger (defauls to None)
        detect_stuck_agents : bool, optional
            Flag to detect if agents are stuck at a
            location longer than a threshold (defaults to `False`.)
        """
        if uid is None:
            self.uid = uuid.uuid4().int
        else:
            self.uid = uid

        self.home = home
        self.home_location = home.location
        self.primary = primary
        self.irregular_events = irregular_events

        # All agents are born at their home.
        self.current_location = home.location
        self.next_trip_min_depart_overide = None

        self.location_label_map = location_label_map

        self.current_location_label = location_label_map.get(home.location, self.edge_pos_str(home.location)) if location_label_map else None

        # Agent starts not mobile (i.e., stationary).
        self.mobile = False

        self.logger = logger

        # (lon_lat, local_time) of last update
        self.detect_stuck_agents = detect_stuck_agents
        self.stuck_agent_detector = None
        self.stuck_agent_detector_time = 100

        # Trigger calculation on first call.
        self.day_schedule = []
        self.active_trip = None

        # Next local time where we need check on our schedule. Dummy time used
        # to always trigger an update check on the first iteration.
        self.next_check_time = datetime(1, 1, 1, tzinfo=timezone.utc)

    @property
    def current_lon_lat(self):
        return self._current_lon_lat

    def update_current_lon_lat(self, lon_lat, local_time):
        """Update the current position of the agent in geographic
        coordinates.

        Parameters
        ----------
        lon_lat : tuple(float, float)
            Longitude (degrees) and latitude (degrees) of the agent's updated
            position at time ``local_time``.
        local_time : datetime
            Local time associated with the location update.
        """
        self._current_lon_lat = lon_lat

        if self.detect_stuck_agents and self.logger is not None and self.mobile:
            if (self.stuck_agent_detector is not None and
                self.stuck_agent_detector[0] == lon_lat):
                dt = (local_time - self.stuck_agent_detector[1]).total_seconds()
                if dt > self.stuck_agent_detector_time:
                    self.logger.warn("Agent %s hasn't moved from "
                                     "(%s, %s) in %0.1f seconds" %
                                     (str(self.uid), lon_lat[0], lon_lat[1],
                                      dt))
            else:
                self.stuck_agent_detector = (lon_lat, local_time)

    def calc_curr_day_schedule(self, local_time):
        """Called once self.day_schedule is empty and agent ready to leave
        home.
        
        The previous day's schedule should always leave the agent at home, as
        should the schedule created by this call.

        Parameters
        ----------
        local_time : datetime
            Local time of the request.
        """
        # This should always be true.
        assert self.current_location == self.home_location

        # We want the schedule centered on the day indicated by local_time.
        # But, schedule samplers return times in seconds past midnight. So, we
        # we first need to calculate the datetime object for that midnight.
        today = local_time.replace(hour=0, minute=0, second=0)

        ha, hd = self.home.sample()
        ha = today + timedelta(seconds=ha)
        hd = today + timedelta(seconds=hd)

        # TODO: calculate expected travel time from home to primary and leave
        # early to arrive on time. For now, we just assume travel time is zero,
        # so agent will always be late for their appointments.
        pa, pd = self.primary.sample()
        pa = today + timedelta(seconds=pa)
        pd = today + timedelta(seconds=pd)

        # We need to pack additional events into the agent's schedule. So,
        # let's discretize the schedule to ~10 minute blocks, and then we can
        # work on the discrete packing problem.

        anchor_trips = []
        # Send them to their primary destination.
        anchor_trips.append(Trip(destination=self.primary.location,
                                 min_depart_for_time=pa,
                                 max_depart_for_time=None,
                                 depart_from_time=pd,
                                 min_stay_duration=timedelta(hours=2),
                                 max_stay_duration=timedelta(hours=12)))

        # Also must leave agent at home to sleep at end of day.
        anchor_trips.append(Trip(destination=self.home.location,
                                 min_depart_for_time=ha,
                                 max_depart_for_time=None,
                                 depart_from_time=hd,
                                 min_stay_duration=timedelta(hours=6),
                                 max_stay_duration=None))

        # Incorporate irregular events.
        destination, duration = self.irregular_events.sample()
        #destination = []

        if len(destination) > 0:
            desired_trips = []
            for i in range(len(destination)):
                desired_trips.append(Trip(destination[i],
                                          min_depart_for_time=None,
                                          max_depart_for_time=ha,
                                          depart_from_time=None,
                                          min_stay_duration=duration[i],
                                          max_stay_duration=duration[i]))

            self.day_schedule = schedule_packer(anchor_trips, desired_trips)
        else:
            self.day_schedule = anchor_trips

        # print()
        # print(self.day_schedule)
        # print()

        #print('Added new daily schedule')
        #print(self.day_schedule)

    def next_destination(self, local_time):
        """Called each simulation step update returning next destination or
        None if the agent wants to remain at current location.
        
        agent.mobile will be set to True externally once the simulation engine
        starts moving the agent. This method will only be called when
        agent.mobile is False.

        Parameters
        ----------
        local_time : datetime
            Local time of the request.

        Returns
        -------
        tuple(edge id, distance along edge) or None
            The specification for the location that the agent should go to
            or None if the agent wants to stay at its current location.
        """
        if local_time < self.next_check_time:
            # This is a speed optimization.
            return None

        # This method is only called when an agent is not mobile
        assert self.mobile is False

        if self.active_trip is None:
            if len(self.day_schedule) == 0:
                self.calc_curr_day_schedule(local_time)

            self.active_trip = self.day_schedule.pop(0)

            if self.next_trip_min_depart_overide is not None:
                # To respect the previous trip's 'max_stay_duration', we might
                # need to adjust this trip's 'min_depart_for_time'.
                if (self.active_trip.min_depart_for_time is None or
                    self.active_trip.min_depart_for_time > self.next_trip_min_depart_overide):
                    self.active_trip.min_depart_for_time = self.next_trip_min_depart_overide

            # We need to recheck the new first element of the schedule.
            return self.next_destination(local_time)
        else:
            # At this point, the context of self.active_trip is active, meaning
            # that the agent is still at previous location but is allowed to
            # leave, or the agent has arrived at self.active_trip.destination
            # and may or may not have stayed long enough to release
            # self.active_trip's context.
            if self.active_trip.destination != self.current_location:
                # The agent is still at previous location but is allowed to
                # leave (i.e., the last trip has released its context) whenever
                # active_trip says that is ok.
                if (self.active_trip.max_depart_for_time is not None and
                    local_time > self.active_trip.max_depart_for_time):
                    # Agent missed the oppurtunity. Get rid of this trip and
                    # recursively call so that we sample the next trip.
                    # We don't update 'next_trip_min_depart_overide' becuase
                    # the next trip also needs to respect this same imperative.
                    self.active_trip = None
                    return self.next_destination(local_time)
                elif self.active_trip.min_depart_for_time is None:
                    # Leave immediately.
                    return self.active_trip.destination
                elif local_time >= self.active_trip.min_depart_for_time:
                    # Return the specification to leave for the desired
                    # destination.
                    return self.active_trip.destination
                else:
                    # Need to wait before departing for destination.
                    self.next_check_time = self.active_trip.min_depart_for_time
                    return None
            else:
                # The agent is at active_trip.destination, but the
                # question is whether they have stayed long enough.
                if local_time >= self.active_trip.depart_from_time:
                    # Ready to release context to the next trip.
                    self.next_trip_min_depart_overide = self.active_trip.next_trip_min_depart_overide
                    self.active_trip = None
                    return self.next_destination(local_time)

                self.next_check_time = self.active_trip.depart_from_time
                # Still wait at the destination.
                return

    def arrived_at_destination(self, loc, t):
        """Called when agent arrives at location.

        Parameters
        ----------
        loc : (edge id, distance along edge)
            Location that the agent arrived at.
        t : datetime
            Local time of the arrival.
        """
        self.current_location = loc

        if self.active_trip.destination == loc:
            self.active_trip.set_actual_arrival_time(t)

        if self.location_label_map is not None:
            self.current_location_label = self.location_label_map.get(loc, self.edge_pos_str(loc))

        self.mobile = False

    def lon_lat_str(self):

        return "{},{}".format(self.current_lon_lat[0], self.current_lon_lat[1])

    def edge_pos_str(self, location):

        return "{},{}".format(location[0], location[1])


class AgentSampler:
    """Class to sample Agents from distibutions defined in a city Map object."""
    def __init__(self, city_map, logger=None):
        """
        Constructor method

        Parameters
        ----------
        city_map : map.Map
            Map object encoding the road network and points of interest.
        logger : logging.Logger, optional
            Logger (defaults to None)
        """
        self.city_map = city_map
        self.logger = logger

    def sample_agents(self, num_agents, route_time=None,
                      home_work_sampler=None, location_labels=True):
        """Sample a collection of agents with routine home/work locations/arrival times as
        well as a series of non-routine locations.

        Parameters
        ----------
        num_agents : int
            Number of agents to sample
        route_time : function, optional
            Callback for calculating route time (not currently used) (defaults to None)
        home_work_sampler : distribution_sampler.SumoPairedLocationSampler, optional
            use probability distributions to choose home/work locations (defaults to None)
        location_labels : bool, optional
            Flag to add location labels to stop point logs (defaults to `True`)

        Returns
        -------
        dict(Agent)
            dictionary of agents for the simulation
        """
        agents = {}
        pbar = tqdm(total=num_agents, smoothing=0.01)
        for i in range(num_agents):
            # For now, let's select an agent's home and primary destination as
            # randomly-selected road segment.

            if home_work_sampler is None:
                home_edge, l1 = self.city_map.get_random_location()
                primary_edge, l2 = self.city_map.get_random_location()
            else:
                home, work = home_work_sampler.sample()
                home_edge, l1 = home
                primary_edge, l2 = work

            # ----------------------------------------------------------------
            # Define the schedule for when the agent is home.

            # What time do they arrive home at night.
            mean_arrival = truncnorm(-4, 4, 20*60*60, 3).rvs(1,)
            arrival_std = truncnorm(0, 2, 0, 2*60*60).rvs(1,)

            # What time do they depart in the morning.
            while True:
                mean_departure = truncnorm(-4, 4, 8*60*60, 3).rvs(1,)
                if mean_departure < mean_arrival:
                    break

            departure_std = truncnorm(0, 2, 0, 2*60*60).rvs(1,)

            days_of_week = np.ones(7, bool)
            home = RegularScheduledEvent(mean_arrival=mean_arrival,
                                         mean_departure=mean_departure,
                                         arrival_std=arrival_std,
                                         departure_std=departure_std,
                                         days_of_week=days_of_week,
                                         location=(str(home_edge), l1))

            # Define the schedule for when the agent is at primary.

            # What time do they arrive at primary.
            mean_arrival = truncnorm(-4, 4, 11*60*60, 3*60*60).rvs(1,)
            arrival_std = truncnorm(0, 2, 0, 2*60*60).rvs(1,)

            # What time do they depart in the morning.
            mean_departure = truncnorm(-4, 4, 14*60*60, 3*60*60).rvs(1,)
            departure_std = truncnorm(0, 2, 0, 2*60*60).rvs(1,)

            days_of_week = np.array([True, True, True, True, True, False, False])
            primary = RegularScheduledEvent(mean_arrival=mean_arrival,
                                            mean_departure=mean_departure,
                                            arrival_std=arrival_std,
                                            departure_std=departure_std,
                                            days_of_week=days_of_week,
                                            location=(str(primary_edge), l2))

            irregular_events = IrregularEvents(self.city_map)

            location_label_map = None
            if location_labels:
                location_label_map = {}
                location_label_map[home.location] = "home:%s" % "{},{}".format(home.location[0], home.location[1])
                location_label_map[primary.location] = "primary:%s" % "{},{}".format(primary.location[0], primary.location[1])

            agent = Agent(home=home, primary=primary,
                          irregular_events=irregular_events,
                          logger=self.logger,
                          location_label_map=location_label_map)

            agents[agent.uid] = agent

            pbar.update(1)

        return agents
