from __future__ import annotations

import numpy as np

from typing import Iterable, Dict
from scipy.sparse import csr_array, dok_array


class MetricsData:
    """Class for saving metrics computed metrics with the
    contribution of each agent disentangled.
    """
    def __init__(self):
        """
        Constructor method
        """

        self.agent_contributions = {}
        self.data = None

    def add_agent(self, agent: str, scores: Iterable, **kwargs) -> None:
        """Add scores from an agent to the histogram

        Parameters
        ----------
        agent : str
            UUID of agent to add
        scores : Iterable
            List of agent scores
        """

        self.agent_contributions[agent] = scores
        self.data += scores

    def remove_agent(self, agent: str) -> np.ndarray:
        """Remove agent scores from histogram

        Parameters
        ----------
        agent : str
            UUID of agent to remove

        Returns
        -------
        np.ndarray
            Agents contribution to the metric.
        """
        if agent in self.agent_contributions:
            agent_contribution = self.agent_contributions[agent]
            self.data -= agent_contribution
            del self.agent_contributions[agent]
            return agent_contribution

    def update_agent(self, agent: str, scores: Iterable) -> None:
        """Replace agent scores in the histogram

        Parameters
        ----------
        agent : str
            UUID of agent to update
        scores : Iterable
            New socres representing the agent's contribution
            to the metric.

        """
        if agent in self.agent_contributions:
            self.remove_agent(agent)
        self.add_agent(agent, scores)

    def update_scores(self, scores: Dict) -> None:
        """Update the histogram with a dictionary of scores

        Parameters
        ----------
        scores : Dict
            Update the contribution of multiple agents.
        """
        for agent, agent_scores in scores.items():
            self.update_agent(agent, agent_scores)

    def get_data(self, density=False):
        """Getter for data calculated from all agents

        Parameters
        ----------
        density : bool, optional
            Normalize data by sum of contributions (defaults to False)

        Returns
        -------
        _type_
            Computed contributions of each agent
        """
        if density:
            return self.data / np.sum(self.data.sum)
        return self.data

    def to_dict(self) -> Dict:
        raise NotImplementedError

    def from_dict(self, data: Dict) -> MetricsData:
        raise NotImplementedError


class MetricsHistogram(MetricsData):
    """Class for computing metrics Histogram with contributions of each
    agent disentangled.
    """
    def __init__(self):

        super(MetricsHistogram, self).__init__()
        # histogram bins
        self.bins = None

    def binnify_scores(self, scores: Iterable) -> np.ndarray:
        """Take in raw scores and binnify them according to
        the bins set in the MetricsHistogram Object

        Parameters
        ----------
        scores: Iterable
            List of agent scores
        """
        if self.bins is None:
            raise ValueError
        return np.histogram(scores, self.bins)[0]

    def add_agent(self, agent: str, scores: Iterable,
                  pre_binned=False) -> None:
        """Add scores from an agent to the histogram

        Parameters
        ----------
        agent : str
            UUID of agent to add.
        scores : Iterable
            Histogram bin counts
        pre_binned : bool, optional
            Scores are pre-binned to counts (defaults to False)
        """

        if pre_binned:
            binned_scores = scores
        else:
            binned_scores = self.binnify_scores(scores)
        super().add_agent(agent, binned_scores)

    def set_bins(self, bins: np.ndarray) -> None:
        """Set histogram bins

        Parameters
        ----------
        bins : np.ndarray
            Bins to use for histogram
        """
        self.bins = bins
        self.data = np.zeros(len(bins) - 1, dtype=np.int64)

    def to_dict(self) -> Dict:
        """Serialize Histogram to a dictionary

        Returns
        -------
        Dict
            Serialized histograms
        """
        data = {}
        data['bins'] = self.bins
        data['agent_contributions'] = self.agent_contributions
        data['histogram'] = self.data
        return data

    def from_dict(self, data: Dict) -> MetricsHistogram:
        """Restore Histogram from dictionary

        Parameters
        ----------
        data : Dict
            Serialized MetricsHistogram

        Returns
        -------
        MetricsHistogram
            MetricsHistogram with data loaded from dictionary.
        """
        self.agent_contributions = data['agent_contributions']

        if data['bins'] is not None:
            self.set_bins(data['bins'])
            for v in self.agent_contributions.values():
                self.data += v
        return self


class MetricsMatrix(MetricsData):
    """Class for computing metrics matrix with contributions of each
    agent disentangled.
    """
    def __init__(self):
        """
        Constructor method.
        """
        super(MetricsMatrix, self).__init__()
        self.matrix_size = None
        self.aoi = None

    def set_matrix_size(self, matrix_size: int, aoi: list) -> None:
        """Set matrix size matrix_size X matrix_size and initialize matrix

        Parameters
        ----------
        matrix_size : int
            Size of matrix
        aoi : list
            List of locations of interest
        """
        self.matrix_size = matrix_size
        self.aoi = aoi
        self.data = csr_array((matrix_size, matrix_size), dtype=np.int64)

    def to_dict(self) -> Dict:
        """Convert Matrix to dict for storage

        Returns
        -------
        Dict
            Agent contributions serialized as dictionary.
        """
        data = {}
        data['matrix_size'] = self.matrix_size
        data['aoi'] = self.aoi
        # Using sparse matrices, convert them to dicts before
        # storing them
        data['agent_contributions'] = {k: dict(v.todok()) for k, v
                                       in self.agent_contributions.items()}
        data['matrix'] = dict(self.data.todok())
        return data

    def update_scores(self, scores: Dict, aoi: list):
        """Update the matrix scores

        Parameters
        ----------
        scores : Dict
            dictionary of agent contributions
        aoi : list
            Locations of interest.
        """
        if self.aoi is None:
            self.set_matrix_size(len(aoi), aoi)
        if any(aoi != self.aoi):
            # AOI don't match, reconstruct matrix
            # with matching aoi
            aoi = [x for x in aoi if x in self.aoi]
            aoi_map = [self.aoi.index(x) for x in aoi]
            for k, v in scores.items():
                nv = dok_array(
                    (self.matrix_size, self.matrix_size), dtype=np.int64)
                v = v.todok()
                for idx, d in v.items():
                    i, j = idx
                    if i not in aoi_map:
                        continue
                    if j not in aoi_map:
                        continue
                    i = aoi_map[i]
                    j = aoi_map[j]
                    nv[i, j] = d
                scores[k] = csr_array(nv)
        super().update_scores(scores)

    def from_dict(self, data: Dict) -> MetricsMatrix:
        """Restore Matrix from dictionary

        Parameters
        ----------
        data : Dict
            Serialized MetricsMatrix

        Returns
        -------
        MetricsMatrix
            MetricsMatrix initialized from saved state.
        """
        self.agent_contributions = data['agent_contributions']
        # Agent contributions stored as sparse matrices
        # reiinitialize them in scipy
        for k, v in self.agent_contributions.items():
            data_v, i, j = [], [], []
            for ij, d in v.items():
                data.append(d)
                i.append[ij[0]]
                j.append(ij[1])
            v = dok_array(data_v, (i, j), dtype=np.int64)
            self.agent_contributions[k] = csr_array(v)
        self.aoi = data['aoi']
        # Compute matrix
        if data['matrix_size'] is not None:
            self.set_matrix_size(data['matrix_size'])
            for v in self.agent_contributions.values():
                self.data += v
        return self
