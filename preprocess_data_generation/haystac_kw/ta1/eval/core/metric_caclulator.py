import pickle

import pandas as pd

from typing import List, Union
from enum import Enum

from haystac_kw.ta1.eval.core.evaluator import Evaluator
from haystac_kw.ta1.eval.core.comparison import MetricsComparison
from haystac_kw.ta1.eval.metrics.metric_base import Metric
from haystac_kw.ta1.eval.metrics import MetricsGroups
import haystac_kw.ta1.eval.metrics as metrics


class MetricCalculator:

    def __init__(self, num_bins=30, metrics: Union[List[Metric], Enum] =
                 MetricsGroups.ALL_METRICS):
        """
        Wrapper Object for computing metrics.

        Parameters
        ----------
        num_bins : int, optional
            Number of bins to use in the metrics histograms (defaults to 30)
        metrics : List[Metric] or MetricsGroups, optional
            Metrics to be calculated (defaults to MetricsGroups.ALL_METRICS)
        """
        self.num_bins = num_bins
        self.clean = False
        self.reference_data = None
        self.subject_data = None
        self.ref_eval = None
        self.sub_eval = None
        self.comparsion = None
        self.metrics = metrics

    def __import_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Internal method for importing csv.

        Parameters
        ----------
        csv_file : str
            Path to csv file

        Returns
        -------
        pd.DataFrame
            Stop points loaded from csv file
        """
        df = pd.read_csv(csv_file)
        return df

    def __import_dataframe(self, dataframe: pd.DataFrame):
        """
        Internal method for importing pd.DataFrame

        Parameters
        ----------
        dataframe : pd.DataFrame
            Stop points
        """
        self.reference_data = dataframe

    def __import_data(self, event_table: pd.DataFrame = None,
                      csv_file: str = None) -> pd.DataFrame:
        """
        Internal method for loading data

        Parameters
        ----------
        event_table : pd.DataFrame, optional
            Stop Points DataFrame (defaults to None)
        csv_file : str, optional
            Stop Points csv file (defaults to None)

        Returns
        -------
        pd.DataFrame
            Stop points as Pandas DataFrame
        """
        if event_table is None and csv_file is None:
            print("Error: you must input at least one source")
            return

        # handle the importing of the data
        data = None
        if csv_file:
            data = self.__import_csv(csv_file)
        if event_table:
            data = self.__import_dataframe(event_table)
        return data

    def import_subject_data(self, event_table: pd.DataFrame = None,
                            csv_file: str = None) -> None:
        """Sets up the data from a pd.DataFrame (event_table) or
            a csv file, or a cached file from disk

        Parameters
        ----------
        event_table : pd.DataFrame, optional
            Reference stop points DataFrame (defaults to None)
        csv_file : str, optional
            Reference stop points csv file (defaults to None)
        """
        self.subject_data = self.__import_data(event_table, csv_file)
        self.clean = False

    def import_reference_data(self, event_table: pd.DataFrame = None,
                            csv_file: str = None) -> None:
        """Sets up the reference data from a pd.DataFrame (event_table) or
            a csv file, or a cached file from disk

        Parameters
        ----------
        event_table : pd.DataFrame, optional
            Reference stop points DataFrame (defaults to None)
        csv_file : str, optional
            Reference stop points csv file (defaults to None)
        """
        self.reference_data = self.__import_data(event_table, csv_file)
        self.clean = False

    def __is_data_ready(self) -> bool:
        """
        Determine if all of the necessary data are available.

        Returns
        -------
        bool
            Whether or not all data is available.
        """
        if self.reference_data is None or self.subject_data is None:
            print("Error: you must load subject and reference data.")
            return False
        else:
            return True

    def __setup_evaluators(self):
        """Internal for setting up the evaluators
        """
        # check to see if the reference sets needs to be run
        if self.ref_eval is None:
            self.ref_eval = Evaluator(metrics=self.metrics,
                                      initialize_hist=True,
                                      num_bins=self.num_bins)
            self.ref_eval.run_metrics(self.reference_data)

        # check for the clean status and subject set
        if self.sub_eval is None or self.clean is False:
            self.sub_eval = Evaluator(metrics=self.metrics,
                                      initialize_hist=False,
                                      num_bins=self.num_bins)
            # use the reference histograms to init the subject
            self.sub_eval.intialize_histograms(
                self.ref_eval.metrics_histograms)
            self.sub_eval.run_metrics(self.subject_data)

    def plot_comparison(self, output_folder: str):
        """Plot the loaded reference and subject data.

        Parameters
        ----------
        output_folder : str
            Folder to save plots
        """
        if not self.__is_data_ready():
            return

        # check to see if we need to run the metrics
        self.__setup_evaluators()

        # get the comparison
        self.comparsion = MetricsComparison(self.sub_eval, self.ref_eval)
        self.comparsion.plot_metrics(output_folder)
        print(f"Plots available in {output_folder}")

    def get_global_cost(self) -> float:
        """Get an average of all JS divergence scores for all metrics

        Returns
        -------
        float
            Average JS divergence across all metrics
        """
        if not self.__is_data_ready():
            return -99.9

        if not self.__is_data_ready():
            return -99.9
        if self.comparsion is None:
            print("Error the data needs to be plotted")
            return -99.9

        return self.comparsion.average_divergence()

    def remove_agent(self, agent_id: str):
        """Get rid of an agent from the subject data

        Parameters
        ----------
        agent_id : str
            UUID of agent to remove.
        """
        self.clean = False

        # get rid of the data item
        df = self.subject_data
        self.subject_data = df[df['agent_id'] != agent_id]

        # remove from the evaluator
        self.sub_eval.remove_agent(agent_id)

    def add_agent(self, event_table: pd.DataFrame = None,
                  stop_points: str = None):
        """Add an agent to the subject data
        we expect that the data is in the same format
        (with an agent id)

        Parameters
        ----------
        event_table : pd.DataFrame, optional
            Stop points for agent dataframe (defaults to None)
        stop_points : str, optional
            Stop points for agent csv file (defaults to None)
        """
        self.clean = False

        if event_table is None and stop_points is None:
            print("Error: you need to pass a pd.DataFrame of csv filename")
            return

        df = None
        if event_table:
            df = event_table
        if stop_points:
            df = pd.read_csv(stop_points)

        # append the data
        self.subject_data.append(df, ignore_index=True)

        # just call run_metrics with this agent's data and it won't
        # overwrite - it will only update the data for this agent
        self.sub_eval.run_metrics(df)

    def compute_highest_contributor(self, metric: str) -> float:
        """Get the average divergence score from all the metrics

        Parameters
        ----------
        metric : str
            Metric of interest.

        Returns
        -------
        float
            None

        Raises
        ------
        NotImplementedError
            Not Implemented
        """
        raise NotImplementedError

    def save(self, filename: str):
        """Save the state

        Parameters
        ----------
        filename : str
            Location to save to disc
        """
        data = {}
        data["clean"] = self.clean
        data["reference_data"] = self.reference_data
        data["subject_data"] = self.subject_data
        data["ref_eval"] = self.ref_eval.to_dict()
        data["sub_eval"] = self.sub_eval.to_dict()
        data["comparsion"] = self.comparsion.to_dict()
        data['metrics'] = [type(x).__name__ for x in self.metrics]

        with open(filename, 'wb') as fout:
            pickle.dump(data, fout)

    def resume(self, filename: str):
        """Restore the state from a saved file

        Parameters
        ----------
        filename : str
            Path to saved state.
        """
        data = {}
        with open(filename, 'rb') as fin:
            data = pickle.load(fin)

        self.reference_data = data["reference_data"]
        self.subject_data = data["subject_data"]
        self.ref_eval = Evaluator()
        self.ref_eval.from_dict(data["ref_eval"])
        self.sub_eval = Evaluator()
        self.sub_eval.from_dict(data["sub_eval"])
        self.comparsion = MetricsComparison(dataset=self.sub_eval,
                                            reference_dataset=self.ref_eval)
        self.comparsion.from_dict(data["comparsion"])
        self.metrics = [getattr(metrics, x)() for x in data['metrics']]
