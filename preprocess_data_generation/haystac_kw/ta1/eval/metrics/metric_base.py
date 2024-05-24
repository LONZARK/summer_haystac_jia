import pandas as pd


class Metric:
    """Base class for a metric object"""

    def __init__(self, metric_name: str, metric_type: str):
        """
        Constructor method

        Parameters
        ----------
        metric_name : str
            Name of the metric
        metric_type : str
            Type of the metric
        """
        self.metric_name = metric_name
        self.metric_type = metric_type

    def compute_metric(self, event_table: pd.DataFrame) -> dict:
        """Compute metric based on stop points.

        Parameters
        ----------
        event_table : pd.DataFrame
            Stop points

        Returns
        -------
        dict
            Computed metric, with agent's contributions disentangled.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the metric

        Returns
        -------
        str
            Name of the metric
        """
        return self.metric_name

    @property
    def type(self) -> str:
        """Metric type

        Returns
        -------
        str
            Metric type
        """
        return self.metric_type


class HistogramMetric(Metric):
    """Base class for a histogram metric object"""

    def __init__(self, metric_name: str):
        super(HistogramMetric, self).__init__(metric_name, 'histogram')
        self.plot_params = {
            "metric_name": metric_name,
            "show_plot": False,
            "save_filename": metric_name + '.png',
            "name_reference": "Reference",
            "name_subject": "Test Subject",
            "plot_log10_x": False,
            "plot_log10_y": False,
            "y_axis_label": "Density",
            "is_density": True
        }


class MatrixMetric(Metric):
    """Base class for a matrix metric object"""

    def __init__(self, metric_name: str):
        super(MatrixMetric, self).__init__(metric_name, 'matrix')
        self.plot_params = {
            "metric_name": metric_name,
            "show_plot": False,
            "save_filename": metric_name + '.png',
        }
