import uuid
import time
import pandas as pd
import logging
import os
from datetime import datetime
from collections import defaultdict
import pickle
from numpy.random import RandomState
import numpy as np
import scipy
from tqdm import tqdm
import random

# import traci as libsumo
import libsumo
import sumolib
from functools import lru_cache
from multiprocessing import Pool, Queue
from typing import List

# haystac_kw imports
from haystac_kw.ta1.agent_behavior_sim.agent import AgentSampler
from haystac_kw.ta1.agent_behavior_sim.map import Map
from haystac_kw.ta1.agent_behavior_sim.distribution_sampler import SumoPairedLocationSampler
from haystac_kw.utils.data_utils.loggers import (ULLTLog, StopPointLog,
                                            log_writer, log_consolidator)


class AgentBehaviorSim:
    """Generates mobility for SUMO starting from a synthetic population."""

    def _configure_loggers(self):
        """Setup the console and file logger."""
        self.logger = logging.getLogger("AgentSim")
        self.logger.setLevel(logging.DEBUG)

        if False:
            _console_handler = logging.StreamHandler()
            _console_handler.setLevel(logging.INFO)
            _console_handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
            )
            self.logger.addHandler(_console_handler)

        _file_handler = logging.FileHandler(
            "{}.debug.log".format(self.conf["logfile_prefix"]),
            mode='w'
        )
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s]:[%(name)s]:[%(module)s:%(funcName)s:%(lineno)d]:[%(levelname)s] "
                "%(message)s"))
        self.logger.addHandler(_file_handler)

    def __init__(self, conf: dict, profiling: bool=False):
        """
        Initialize the synthetic population.
        
        Parameters
        ----------
        conf : dict
            dictionary with the configurations
        profiling : bool, optional
            enable cProfile (defaults to `False`)
        """
        self.conf = conf

        self.num_agents = conf['num_agents']
        self.sim_start_epoch_seconds = conf['sim_start_epoch_seconds']
        self.sim_duration_seconds = conf['sim_duration_seconds']
        self.sumo_gui = conf['sumo_gui'].lower() == 'true'
        self.sumo_logging = conf['sumo_logging'].lower() == 'true'
        self.detect_stuck_agents = conf['detect_stuck_agents'].lower() == 'true'
        self.stuck_agent_detector_time = float(conf['stuck_agent_detector_time'])
        self.num_threads = str(conf['num_threads'])
        self.num_writers = None
        self.writer_queue = None
        if 'writer_workers' in conf:
            self.num_writers = int(conf['writer_workers'])
            self.writer_queue = Queue()
            self.writer_pool = Pool(
                self.num_writers, log_writer, (self.writer_queue,))

        if conf['stop_points_only'] == 'true':
            self.simulate = self.stop_point_sim
        else:
            self.simulate = self.sumo_sim

        # Settings for checkpointing the SUMO simulation
        self.checkpoint_frequency = int(
            conf['simulation_checkpoint_frequency'])
        self.checkpoint_dir = conf['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.home_work_sampler = None
        if conf['use_spatial_distributions']:
            self.home_work_sampler = SumoPairedLocationSampler(
                conf['home_dist'],
                conf['work_dist'],
                conf['pairwise_dist'],
                conf['edge_mapping']
            )

        self._configure_loggers()
        self._profiling = profiling
        self._random_generator = RandomState(seed=self.conf["seed"])

        self.roadnet_fname = list(
            sumolib.xml.parse(
                conf["sumocfg"],
                'net-file'))[0].value
        self.city_map = Map(self.roadnet_fname)

        self.agents = {}
        self.agent_sampler = AgentSampler(self.city_map, self.logger)

        self.logger.info("Starting sumo with file %s.", conf["sumocfg"])

        if self.sumo_logging:
            trace_file = "libsumo.log"
        else:
            trace_file = None

        if self.sumo_gui:
            libsumo.start(["sumo-gui", "-c", conf["sumocfg"],
                           "--start", "--time-to-teleport", "-1"],
                          traceFile=trace_file)
            self.using_sumo_gui = True
        else:
            libsumo.start(["sumo", "-c", conf["sumocfg"], '--threads',
                           str(self.num_threads), '--device.rerouting.threads',
                           str(self.num_threads),
                           "--time-to-teleport", "-1"], traceFile=trace_file)
            self.using_sumo_gui = False

        self.vehicle_to_agent_uid = {}
        if os.path.isdir(conf['init_checkpoint']):
            self.load_checkpoint(conf['init_checkpoint'])

    def create_agents(self, overwrite: bool=False):
        """Define all of the agents.

        Parameters
        ----------
        overwrite : bool, optional
            overwrite agents loaded from checkpoint (defaults to `False`)
        """
        if (len(self.agents) == 0) or overwrite:
            self.logger.info(f"Generating {self.num_agents} agents")
            agents = self.agent_sampler.sample_agents(self.num_agents,
                                                    self.get_route_time,
                                                    self.home_work_sampler, 
                                                    location_labels=True)
            self.agents = agents
        else:
            self.logger.info(f"Using Checkpointed Agents")

    @lru_cache(maxsize=500000)
    def get_route_time(self, edge1: str, edge2: str):
        """Query SUMO for the estimated travel time in seconds
        between two edges in the road network. Travel time is cached.

        Parameters
        ----------
        edge1 : string
            Name of source edge in the SUMO road network
        edge2 : string
            Name of destination edge in the SUMO road network

        Returns
        -------
        float
            Estimated travel time from edge1 to edge2 in seconds
        """
        return libsumo.simulation.findRoute(edge1, edge2).travelTime

    @lru_cache(maxsize=None)
    def get_route_id(self, route: List[str]):
        """Adds a route to the SUMO simulation and returns
        the route ID.

        Parameters
        ----------
        route : list(string)
            List of redges in the route.

        Returns
        -------
        string
            UUID of route in SUMO
        """
        uid_rt = str(uuid.uuid4())
        libsumo.route.add(uid_rt, route)
        return uid_rt

    @lru_cache(maxsize=10000)
    def convert_road_to_geo(self, edge: str, pos: float):
        """Converts a point along a road segment in the SUMO
        road network into a geographic coordinate.

        Parameters
        ----------
        edge : string
            Edge on the SUMO network.
        pos : float
            Location along length of edge.

        Returns
        -------
        Tuple(float)
            Longitude,Latitude of the location
        """
        return libsumo.simulation.convert2D(edge, pos, toGeo=True)

    def save_checkpoint(self, save_dir: str, local_time: datetime):
        """Save a checkpoint of the agent simulation so that it can be
        resumed at later date.

        Parameters
        ----------
        save_dir : string
            Output folder where checkpoint will be saved.
        local_time : datetime
            Current Time/Date within the simulation.
        """
        # Save SUMO state
        libsumo.simulation.saveState(
            os.path.join(save_dir, 'sumo.chkpt')
            )
        # Save all of our agents individually
        agent_dir = os.path.join(save_dir, 'agents')
        os.makedirs(agent_dir, exist_ok=True)
        for agent_uuid, agent in self.agents.items():
            pickle.dump(agent, open(os.path.join(
                agent_dir, str(agent_uuid)+'.pkl'
            ), 'wb'))
        # Save all of the random generator's states
        random_states = {
            '_random_generator': self._random_generator,
            'numpy_state': np.random.get_state(),
            'python_state': random.getstate()
        }
        pickle.dump(random_states, open(os.path.join(save_dir, 'random_states.pkl'), 'wb'))
        # Save current time
        pickle.dump(local_time.timestamp(), open(os.path.join(save_dir, 'local_time.pkl'),'wb'))
        pickle.dump(self.vehicle_to_agent_uid, open(os.path.join(save_dir, 'vehicle_to_agent_uid.pkl'),'wb'))

        for log in self.position_logs.values():
            log.save()

        for log in self.stop_point_logs.values():
            log.save()

    def load_checkpoint(self, checkpoint_dir: str):
        """Load the state of a previous simulation that it can be resumed.

        Parameters
        ----------
        checkpoint_dir : string
            Directory where the simulation checkpoints are saved.
        """
        # Load SUMO state
        libsumo.simulation.loadState(os.path.join(checkpoint_dir, 'sumo.chkpt'))
        # Load all of the agents individually
        for agent in os.listdir(os.path.join(checkpoint_dir, 'agents')):
            agent_uuid = agent.replace('.pkl', '')
            self.agents[int(agent_uuid)] = pickle.load(open(
                os.path.join(checkpoint_dir, 'agents', agent), 'rb'
            ))
        # Load all of the random states
        random_states = pickle.load(open(os.path.join(
            checkpoint_dir, 'random_states.pkl'
        ), 'rb'))
        self._random_generator = random_states['_random_generator']
        np.random.set_state(random_states['numpy_state'])
        random.setstate(random_states['python_state'])
        # Load local time
        self.sim_start_epoch_seconds = pickle.load(open(
            os.path.join(checkpoint_dir, 'local_time.pkl'), 'rb'
        ))
        # Load vehicle to agent id
        self.vehicle_to_agent_uid = pickle.load(open(
            os.path.join(checkpoint_dir, 'vehicle_to_agent_uid.pkl'), 'rb'
        ))

    def stop_point_sim(self):
        """Approximate simulation just computing stop points, only using SUMO
        for route times
        """
        self.logger.info("Running Stop Point Simulation (No Microsim)")
        # Calculate the local time relative to the city map for this
        # simulation time.

        stop_point_logs = {}
        for agent in self.agents.values():
            stop_point_logs[agent.uid] = StopPointLog(
                agent.uid, 'stop_point_logs', writer_queue=self.writer_queue)

        departed_ids = []
        arrived_ids = defaultdict(list)
        time_delta = libsumo.simulation.getDeltaT()
        timestep = 0
        active_vehicle_count = 0
        pbar = tqdm(total=self.sim_duration_seconds, smoothing=0.01)

        sim_end_local_time = datetime.fromtimestamp(
            self.sim_duration_seconds +
            self.sim_start_epoch_seconds,
            self.city_map.timezone)

        # Loop over simulation time steps.
        simulation_iter_step = 0
        while True:
            timestep += time_delta

            # Calculate the local time relative to the city map for this
            # simulation time.
            local_time = datetime.fromtimestamp(
                timestep + self.sim_start_epoch_seconds,
                self.city_map.timezone)

            # Once an agent has arrived, update the Agent instance to not
            # mobile.
            for uid in arrived_ids[simulation_iter_step]:
                agent = self.agents[uid]
                agent.mobile = False
                agent.arrived_at_destination(agent.sent_to_destination,
                                             local_time)
                edge,pos = agent.current_location
                agent.update_current_lon_lat(self.convert_road_to_geo(edge, pos), local_time)
                agent.sent_to_destination = None
                stop_point_logs[agent.uid].add_sample(local_time,
                                                  agent.current_lon_lat,
                                                  'arrived',
                                                  event_label=agent.current_location_label)
                active_vehicle_count-=1

            for uid in departed_ids:
                agent = self.agents[uid]
                edge, pos = agent.current_location
                agent.update_current_lon_lat(self.convert_road_to_geo(edge, pos), local_time)
                stop_point_logs[agent.uid].add_sample(local_time,
                                                agent.current_lon_lat,
                                                'departed',
                                                event_label=agent.current_location_label)
            departed_ids=[]

            for agent in self.agents.values():

                if not agent.mobile:
                    # Check if agent is ready to move on to its next
                    # destination.
                    dst = agent.next_destination(local_time)
                    if dst is not None:

                        src = agent.current_location

                        # print('Sending vehicle from', src, 'to', dst)
                        agent.sent_to_destination = dst

                        route_time = self.get_route_time(src[0], dst[0])
                        departed_ids.append(agent.uid)

                        # Compute an arrival time garaunteed to land on a time
                        # step
                        arrival_time = simulation_iter_step + \
                            (route_time // time_delta)
                        arrived_ids[arrival_time].append(agent.uid)
                        active_vehicle_count += 1
                        # Add vehicle to simulation

                        # Set the agent to status mobile.
                        agent.mobile = True
                        agent.current_location_label = None # TODO: Check edge case

            # Next simulation step.
            tic = time.time()

            pbar.update(1)
            pbar.set_description(
                'Sim time %0.3f with %i active SUMO vehicles. Time for SUMO simulation step: % 0.4f' %
                (timestep, active_vehicle_count, time.time() - tic))
            if local_time >= sim_end_local_time:
                break

            simulation_iter_step += 1

        pbar.close()

        for log in stop_point_logs.values():
            log.save()

        if self.num_writers is not None:
            for _ in range(self.num_writers):
                self.writer_queue.put(None)
            self.writer_pool.close()
            self.writer_pool.join()
        with Pool(int(self.num_threads)) as pool:
            pool.map(log_consolidator, [x.get_checkpoints()
                                        for x in stop_point_logs.values()])

    def sumo_sim(self):
        """Microsimulate agent movements using SUMO."""
        self.logger.info("Running SUMO Microsimulation")

        position_logs = {}
        stop_point_logs = {}
        for agent in self.agents.values():
            position_logs[agent.uid] = ULLTLog(
                agent.uid, 'trajectories', writer_queue=self.writer_queue)
            stop_point_logs[agent.uid] = StopPointLog(
                agent.uid, 'stop_point_logs', writer_queue=self.writer_queue)

        self.position_logs = position_logs
        self.stop_point_logs = stop_point_logs

        sim_end_local_time = datetime.fromtimestamp(
            self.sim_duration_seconds +
            self.sim_start_epoch_seconds,
            self.city_map.timezone)

        pbar = tqdm(total=self.sim_duration_seconds, smoothing=0.01)

        self.vehicle_to_agent_uid = {}

        # We are agnostic to the absolute sim time within sumo, the range of
        # which is controlled by the .sumocfg. We never call
        # libsumo.simulation.getTime(), we just keep track of timestep
        # increments ourselves.
        timestep = 0
        delta_t = libsumo.simulation.getDeltaT()

        destination_arrivals = 0
        curr_num_vehicles = 0

        local_time = datetime.fromtimestamp(timestep + self.sim_start_epoch_seconds,
                                             self.city_map.timezone)

        # Capture all agent's current position.
        for agent in self.agents.values():
            ll = libsumo.simulation.convert2D(agent.current_location[0],
                                              agent.current_location[1],
                                              laneIndex=0,
                                              toGeo=True)
            agent.update_current_lon_lat(ll, local_time)

        self.logger.info(f"Simulation starting at local time %s" % str(local_time))

        # Loop over simulation time steps.
        while True:
            tic0 = time.time()

            # Calculate the local time relative to the city map for this
            # simulation time.
            local_time = datetime.fromtimestamp(
                timestep + self.sim_start_epoch_seconds,
                self.city_map.timezone)

            if (timestep > 0 and
                ((timestep//delta_t) % self.checkpoint_frequency) == 0):
                save_dir = os.path.join(self.checkpoint_dir,
                                        local_time.strftime("%Y%m%d-%H%M%S"))
                os.makedirs(save_dir, exist_ok=True)
                self.save_checkpoint(save_dir, local_time)

            need_sumo_update = curr_num_vehicles > 0

            # Record the position of all active vehicles to their agents.
            for vid in libsumo.vehicle.getIDList():
                xy_pos = libsumo.vehicle.getPosition(vid)
                ll = libsumo.simulation.convertGeo(*xy_pos)
                agent = self.agents[self.vehicle_to_agent_uid[vid]]
                agent.update_current_lon_lat(ll, local_time)

                # TODO make this a config parameter verbose_route_logging.
                if False:
                    self.logger.info("Agent %s on road=%s, position=%s, lane=%s" %
                                     (str(agent.uid),
                                      libsumo.vehicle.getRoadID(vid),
                                      libsumo.vehicle.getLanePosition(vid),
                                      libsumo.vehicle.getLaneIndex(vid)))

            # Once an agent has arrived, update the Agent instance to not
            # mobile.
            arrived_ids = libsumo.simulation.getArrivedIDList()
            for vuid in arrived_ids:
                # print('arrived', uid)
                uid = self.vehicle_to_agent_uid[vuid]
                agent = self.agents[uid]
                agent.arrived_at_destination(agent.sent_to_destination,
                                             local_time)
                agent.sent_to_destination = None
                stop_point_logs[agent.uid].add_sample(local_time,
                                                      agent.current_lon_lat,
                                                      'arrived',
                                                      event_label=agent.current_location_label)
                del self.vehicle_to_agent_uid[vuid]
                destination_arrivals += 1
                curr_num_vehicles -= 1
                need_sumo_update = True

            departed_ids = libsumo.simulation.getDepartedIDList()
            for uid in departed_ids:
                # print('departed', uid)
                agent = self.agents[self.vehicle_to_agent_uid[uid]]
                stop_point_logs[agent.uid].add_sample(local_time,
                                                      agent.current_lon_lat,
                                                      'departed',
                                                      event_label=agent.current_location_label)

            for agent in self.agents.values():
                # Record all agents' current longitude and latitude, whether
                # mobile or not.
                if agent.current_lon_lat is not None:
                    position_logs[agent.uid].add_sample(local_time,
                                                        agent.current_lon_lat)

                if not agent.mobile:
                    # Check if agent is ready to move on to its next
                    # destination.
                    dst = agent.next_destination(local_time)
                    if dst is not None:
                        src = agent.current_location

                        # print('Sending vehicle from', src, 'to', dst)
                        agent.sent_to_destination = dst

                        if False:
                            self.logger.info('Sending agent %s from %s to %s' %
                                             (agent.uid, str(src), str(dst)))

                        # Create a new vehicle for this agent to travel in and
                        # keep track of the mapping between this SUMO vehicle
                        # ID and the agent's uid.
                        uid = str(uuid.uuid4().int)
                        self.vehicle_to_agent_uid[uid] = agent.uid

                        # Add vehicle to simulation
                        uid_rt = self.get_route_id((src[0], dst[0]))
                        libsumo.vehicle.add(uid, uid_rt, departPos=src[1],
                                            arrivalPos=dst[1])
                        curr_num_vehicles += 1

                        # Set the agent to status mobile.
                        agent.mobile = True

                        need_sumo_update = True

            # Next simulation step.
            tic = time.time()

            if need_sumo_update:
                libsumo.simulationStep()
                # curr_num_vehicles = libsumo.vehicle.getIDCount()

            timestep += delta_t

            dt1 = time.time() - tic
            dt2 = time.time() - tic0

            # if libsumo.vehicle.getIDCount() == 0:
            #     break

            pbar_str = ("Sim time %0.3f with %i active SUMO vehicles "
                        "(%i arrivals)    " %
                        (timestep, curr_num_vehicles, destination_arrivals))
            pbar_str = pbar_str + ('Total loop time: %0.4f (%0.1f%% in SUMO)' %
                                   (dt2, dt1 / dt2 * 100))
            pbar.set_description(pbar_str)
            pbar.update(1)

            if local_time >= sim_end_local_time:
                # Save simulation end state
                self.save_checkpoint(save_dir, local_time)
                break

        pbar.close()

        self.logger.info('Saving trajectories')
        for log in position_logs.values():
            log.save()

        self.logger.info('Saving stop points')
        for log in stop_point_logs.values():
            log.save()

        if self.num_writers is not None:
            for _ in range(self.num_writers):
                self.writer_queue.put(None)
            self.writer_pool.close()
            self.writer_pool.join()
        with Pool(int(self.num_threads)) as pool:
            pool.map(log_consolidator, [x.get_checkpoints()
                                        for x in position_logs.values()])
            pool.map(log_consolidator, [x.get_checkpoints()
                                        for x in stop_point_logs.values()])

    def close(self):
        libsumo.close()
