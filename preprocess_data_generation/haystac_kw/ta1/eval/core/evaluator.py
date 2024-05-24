from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from typing import List, Dict, Union
from enum import Enum
from scipy.spatial import distance

from haystac_kw.ta1.eval.metrics.metric_base import Metric
from haystac_kw.ta1.eval.metrics import MetricsGroups
from haystac_kw.ta1.eval.core.metrics_data import MetricsHistogram, MetricsMatrix
from haystac_kw.utils.viz.histogram_utils import estimate_bins
import haystac_kw.ta1.eval.metrics as metrics


class Evaluator:
    """Class for computing metrics on a dataset of stop points"""
    def __init__(
            self,
            metrics: Union(
                List[Metric],
                Enum) = MetricsGroups.ALL_METRICS,
            initialize_hist=False,
            initialize_matrix=False,
            num_bins=30):
        """
        Constructor method.

        Parameters
        ----------
        metrics : List[Metric] or MetricsGroups, optional
            Metrics to be calculated (defaults to MetricsGroups.ALL_METRICS)
        initialize_hist : bool, optional
            Compute histogram bins based input data. (defaults to False)
        initialize_matrix : bool, optional
            Compute matrices based on input data (defaults to False)
        num_bins : int, optional
            Number of bins to use in histograms (defaults to 30)
        """

        if isinstance(metrics, Enum):
            metrics = metrics.value
        self.metrics = [x() for x in metrics]
        self.metrics_histograms = {x.name: MetricsHistogram()
                                   for x in self.metrics
                                   if x.type == 'histogram'}
        self.metrics_matrices = {x.name: MetricsMatrix()
                                 for x in self.metrics
                                 if x.type == 'matrix'}
        self.metrics_data = self.metrics_histograms
        self.metrics_data.update(self.metrics_matrices)

        self.initialize_hist = initialize_hist
        self.initialize_matrix = initialize_matrix
        self.num_bins = num_bins

    def run_metrics(self, event_table: pd.DataFrame):
        """Compute metrics on a Pandas DataFrame of stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Input data
        """
        for metric in self.metrics:
            result = metric.compute_metric(event_table)
            if metric.type == 'histogram':
                if self.initialize_hist:
                    bins = estimate_bins(result, self.num_bins)
                    self.metrics_histograms[metric.name].set_bins(bins)
                self.metrics_histograms[metric.name].update_scores(result)
            elif metric.type == 'matrix':
                result, aoi = result
                if self.initialize_hist and metric.type == 'matrix':
                    matrix_size = list(result.values())[0].shape[0]
                    self.metrics_matrices[metric.name].set_matrix_size(
                        matrix_size, aoi)
                self.metrics_matrices[metric.name].update_scores(result, aoi)
        self.initialize_hist = False
        self.initialize_matrix = False

    def remove_agent(self, agent: str) -> dict:
        """Remove agent from the computed metrics.

        Parameters
        ----------
        agent : str
            UUID of agent to be removed

        Returns
        -------
        dict
            Computed metrics with agent removed
        """
        agent_contributions = {}
        for metric_name, metric_data in self.metrics_data.items():
            agent_contributions[metric_name] = \
                metric_data.remove_agent(agent)
        return agent_contributions

    def add_agent(self, agent: str, scores: Dict[str, np.ndarray]) -> None:
        """Add agent to the computed metrics.

        Parameters
        ----------
        agent : str
            UUID of agent to be added.
        scores : Dict[str, np.ndarray]
            Dictionary of metrics scores computed for the agent.
        """
        for k, v in scores.items():
            if v is not None:
                self.metrics_data[k].add_agent(agent, v, pre_binned=True)

    def intialize_histograms(self, histograms: Dict[str, MetricsHistogram]):
        """Initialize all of the metrics histograms
        with the same bins as reference histograms.

        Parameters
        ----------
        histograms : Dict[str, MetricsHistogram]
            Dictionary of reference histograms.
        """
        for k, v in histograms.items():
            if k in self.metrics_histograms:
                self.metrics_histograms[k].set_bins(v.bins)
                self.num_bins = len(v.bins)
                self.initialize_hist = False

    def to_dict(self) -> Dict:
        """Serialize computed metrics to a distionary.

        Returns
        -------
        Dict
            Serialized metrics.
        """
        data = {}
        data['metrics'] = [type(x).__name__ for x in self.metrics]
        data['histograms'] = {k: v.to_dict() for k, v
                              in self.metrics_histograms.items()}
        data['matrices'] = {k: v.to_dict() for k, v
                            in self.metrics_matrices.items()}
        data['num_bins'] = self.num_bins
        data['initialize_hist'] = self.initialize_hist
        data['initialize_matrix'] = self.initialize_matrix
        return data

    def to_pickle(self, filename: str) -> None:
        """Save to disk as a dictionary.

        Parameters
        ----------
        filename : str
            File to save to.
        """
        data = self.to_dict()
        with open(filename, 'wb') as fout:
            pickle.dump(data, fout)

    def from_pickle(self, filename: str) -> Evaluator:
        """Reload Evaluator from disk.

        Parameters
        ----------
        filename : str
            Path to save to disk as a pickle.

        Returns
        -------
        Evaluator
            Returns a copy of Evaluator from disk.
        """
        with open(filename, 'rb') as fin:
            data = pickle.load(fin)
            return self.from_dict(data)

    def from_dict(self, data: Dict) -> Evaluator:
        """Load data from dictionary.

        Parameters
        ----------
        data : Dict
            Dictionary of computed metrics.

        Returns
        -------
        Evaluator
            Returns a copy of Evaluator from dict.
        """

        self.metrics_histograms = {k: MetricsHistogram().from_dict(v)
                                   for k, v in data['histograms'].items()}
        self.metrics_matrices = {k: MetricsMatrix().from_dict(v)
                                 for k, v in data['matrices'].items()}
        self.num_bins = data['num_bins']
        self.initialize_hist = data['initialize_hist']
        self.initialize_matrix = data['initialize_matrix']
        self.metrics = [getattr(metrics, x)() for x in data['metrics']]
        return self

    def calculate_js_divergence(
            self, evaluator: Evaluator) -> Dict[str, float]:
        """Compute JS divergence against a reference dataset.

        Parameters
        ----------
        evaluator : Evaluator
            Dataset to compute JS divergence against.

        Returns
        -------
        Dict[str, float]
            Dictionary of JS Divergence values for each metric.
        """

        divergences = {}
        for metric_name, metric_hist in self.metrics_histograms.items():
            if metric_name not in evaluator.metrics_histograms:
                continue
            metric_hist_ref = evaluator.metrics_histograms[metric_name]
            density1 = metric_hist.get_data(density=True)
            density2 = metric_hist_ref.get_data(density=True)
            divergences[metric_name] = \
                distance.jensenshannon(density1, density2) ** 2
        return divergences
