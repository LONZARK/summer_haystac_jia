import numpy as np

from pandas import DataFrame
from typing import Dict

from haystac_kw.ta1.eval.core.evaluator import Evaluator
from haystac_kw.utils.viz.plot_metrics import get_plot_and_js_divergence


class MetricsComparison:
    """Class for generating metrics plots for comparing
    a ULLT trajectory dataset with a reference dataset.
    """
    def __init__(self, dataset: Evaluator, reference_dataset: Evaluator):
        """
        Constructor method

        Parameters
        ----------
        dataset : Evaluator
            Dataset being evaluated
        reference_dataset : Evaluator
            Dataset to be used as reference
        """
        self.dataset = dataset
        self.reference_dataset = reference_dataset

    def average_divergence(self) -> float:
        """Calculate the average JS divergence over all
        metrics. All metrics are weighted equally.

        Returns
        -------
        float
            Average JS Divergence.
        """
        divergences = self.dataset.calculate_js_divergence(
            self.reference_dataset)
        avg_divergence = np.mean(list(divergences.values()))
        return avg_divergence

    def average_divergence_without_agent(self, agent: str) -> float:
        """Compute the average JS divergence with the specified
        agent removed. Agent is reintroduced to the dataset
        before this method returns.

        Parameters
        ----------
        agent : str
            UUID of the agent to be removed
            
        Returns
        -------
        float
            Average JS divergence without `agent`
        """
        agent_contributions = self.dataset.remove_agent(agent)
        average_divergence = self.average_divergence()
        self.dataset.add_agent(agent, agent_contributions)
        return average_divergence

    def remove_agent(self, agent: str) -> None:
        """Remove agent from all of the metrics.

        Parameters
        ----------
        agent : str
            UUID of agent to be removed
        """
        self.dataset.remove_agent(agent)

    def update_dataset(self, event_table: DataFrame) -> None:
        """Update dataset being evaluated with new stop points.

        Parameters
        ----------
        event_table : DataFrame
            Pandas Dataframe containing new stop points
        """
        # update metrics
        self.dataset.run_metrics(event_table)

    def plot_metrics(self, save_folder: str):
        """Plot comparison metrics plots

        Parameters
        ----------
        save_folder : str
            Location to save metrics plots
        """
        for metric in self.dataset.metrics:

            try:
                hist_1 = self.dataset.metrics_histograms[metric.name]
                hist_2 = self.reference_dataset.metrics_histograms[metric.name]
                bins = hist_1.bins
                kwargs = {}
                kwargs['minx'] = bins[0]
                kwargs['maxx'] = bins[-1]
                kwargs.update(metric.plot_params)
                kwargs['save_folder'] = save_folder
                get_plot_and_js_divergence(
                    hist_1.get_data(density=kwargs['is_density']),
                    hist_2.get_data(density=kwargs['is_density']),
                    bins,
                    **kwargs
                )
            except Exception as e:
                print(f'Failed to plot {metric.name}')
                print(e)

    def to_dict(self) -> Dict:
        """Serialize to a dictionary

        Returns
        -------
        Dict
            Comparison object serialized as dict
        """
        data = {}
        data['dataset'] = self.dataset.to_dict()
        data['reference_dataset'] = self.reference_dataset.to_dict()
        return data

    def from_dict(self, data: Dict):
        """Initialize from dictionary

        Parameters
        ----------
        data : Dict
            Data from saved Comparison object
        """
        self.dataset = data['dataset']
        self.reference_dataset = data['reference_dataset']
        return
