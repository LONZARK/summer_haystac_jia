import json
import glob
import logging

import pandas as pd
import ubelt as ub

from typing import List, Union, Optional, Callable
from pathlib import Path
from datetime import datetime

from haystac_kw.data.types.event import SimEvent
from haystac_kw.data.schemas.fas import Agent_FAS, Point_FAS


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AnomalyDetector")


class AnomalyDetector():
    """Base class for an agent anomaly detector."""
    def __init__(self, input_dir: Union[str, Path], output_dir: Union[str, Path]=""):
        """
        Class constructor

        Parameters
        ----------
        input_dir : str or Path
            Path to a directory of per-agent files containing 
            a distilled version of the ULLT data from HumoNet
            described `here <https://docs.google.com/document/d/1miBZiRazh7Ko6Bk06o2eGGVv3QgkWrDgfZCM_mqwxac/edit?usp=sharing>`__
        output_dir : str or Path, optional
            Directory to save all FAS results (defaults to "")
        """
        self.input_dir = input_dir

        # Output directory
        self.output_dir = Path(f"{output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Agent ids
        self.agent_ids: List[int] = []
        for agent_fn in glob.glob(f"{self.input_dir}/*.parquet"):
            agent_id = Path(agent_fn).stem # TODO: Convert to int here
            self.agent_ids.append(agent_id)

        # Detection results
        self.agent_data: List[Agent_FAS] = []
        self.point_data: List[Point_FAS] = []

    def load_data(self, fn: str, cls_: Optional[Callable]=None):
        """Load a parquet file into either a `pd.DataFrame` or a `type`

        Parameters
        ----------
        fn : str
            Filename to read
        cls_ : Callable, optional
            Python class that can be initialized with a `pd.Series`
            (defaults to None)

        Returns
        -------
        pd.DataFrame or a list of ``cls_``
            The contents of ``fn``
        """
        df = pd.read_parquet(fn)

        if type is not None:
            data = []
            #import pdb; pdb.set_trace()
            for i, row in df.iterrows():
                cls_obj = cls_.from_pd_series(row)
                data.append(cls_obj)
        else:
            data = df
            cls_ = pd.DataFrame
            
        log.debug(f"Loaded data from {fn} as {cls_}")
        return data

    def preprocess(self):
        """Optional method to preprocess or pre-compute"""
        pass

    def detect_point(self, points: List[SimEvent]) -> List[Point_FAS]:
        """Predict if the given point along a trajectory is anomalous

        Parameters
        ----------
        points : List[SimEvent]
            List of events along a trajectory

        Returns
        -------
        List[Point_FAS]
            All the point-level FAS results for an agent

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method
        """
        raise NotImplementedError("Subclasses should implement this.")

    def detect_agent(self, points: List[SimEvent]) -> Agent_FAS:
        """Determine if the given agent's trajectory is anomalous

        Parameters
        ----------
        points : List[SimEvent]
            List of events along a trajectory   

        Returns
        -------
        Agent_FAS
            An agent-level FAS result

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method
        """
        raise NotImplementedError("Subclasses should implement this.")

    def detect(self):
        """Determine if an agent is anomalous at an agent and point level."""
        self.preprocess()

        for agent_id in ub.ProgIter(self.agent_ids, desc="Detecting anomalies in agents"):
            self.point_data = []

            agent_db = self.load_data(f"{self.input_dir}/{agent_id}.parquet",
                                      SimEvent)

            self.point_data = self.detect_point(agent_db)
            self.save(self.point_data, agent_id, fn="fas_point.parquet")

            self.agent_data = self.detect_agent(agent_db)
            self.save([self.agent_data], agent_id, fn=f"fas_agent.parquet")

        self.postprocess()
        
    def postprocess(self):
        """Optional method to post-process the detection results"""
        pass

    def save(self,
             data: List[Agent_FAS or Point_FAS],
             agent_id: int,
             fn: str):
        """Save the agent's detections to files

        Parameters
        ----------
        agent_id : int
            ID of agent
        data : List[Agent_FAS, or Point_FAS]
            List of either agent or point level FAS
        fn : str
            Output filename, does not include the root path
        """
        agent_output = Path(f"{self.output_dir}/{agent_id}")
        agent_output.mkdir(parents=True, exist_ok=True)

        fn = Path(agent_output, fn)
        
        df = pd.DataFrame([json.loads(a.model_dump_json()) for a in data])
        df.to_parquet(fn)
        
        log.debug(f"Saved data to {fn}")
