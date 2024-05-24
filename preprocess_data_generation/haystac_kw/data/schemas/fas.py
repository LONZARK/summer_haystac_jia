"""
Schema for the FindAlertSpecification (FAS) standard
(https://haystac-program.com/testing-and-evaluation/interfaces-and-standards/-/wikis/Standards/Find-Alert-Specification)
"""
import pydantic
import datetime
import pandas as pd

from typing import Union

from haystac_kw.data import BaseModel
from haystac_kw.data.types.time import validate_datetime


class Agent_FAS(BaseModel):
    """Prediction result for a trajectory in the test data

    Parameters
    ----------
    agent : pydantic.PositiveInt
        agent unique identifier
    is_anomalous : bool
        True if agent trajectory is anomalous, else False
    anomaly_score : pydantic.confloat
        score between 0 and 1 indicating the degree to
        which the agent trajectory is considered anomalous
    """
    model_config = pydantic.ConfigDict(strict=True)

    agent: pydantic.PositiveInt
    is_anomalous: bool
    anomaly_score: pydantic.confloat(ge=0, le=1)

    @classmethod
    def from_pd_series(cls, agent_row: pd.Series):
        """Initialize an agent FAS from a row in a pandas dataframe

        Parameters
        ----------
        agent_row : pd.Series
            Row of a pd.DataFrame

        Return
        ------
        Agent_FAS
            Agent-levcel FAS
        """
        return cls(
            agent=agent_row["agent"],
            is_anomalous=agent_row["is_anomalous"],
            anomaly_score=agent_row["anomaly_score"]
        )

class Point_FAS(BaseModel):
    """Prediction result for a trajectory point in the test data

    Parameters
    ----------
    agent : pydantic.PositiveInt
        agent unique identifier
    timestamp : datetime.datetime or str
        datetime with timezone
    is_anomalous : bool
        True if agent trajectory point
        is anomalous, else False
    anomaly_score : pydantic.confloat
        score between 0 and 1 indicating the degree to which
        the agent trajectory point is considered anomalous
    """
    model_config = pydantic.ConfigDict(strict=True)
    
    agent: pydantic.PositiveInt
    timestamp: Union[datetime.datetime, str]
    is_anomalous: bool
    anomaly_score: pydantic.confloat(ge=0, le=1)

    @classmethod
    @pydantic.field_validator('timestamp')
    def validate_datetime_type(cls, value):
        return validate_datetime(value)

    @classmethod
    def from_pd_series(cls, point_row: pd.Series):
        """Initialize a point FAS from a row in a pandas dataframe

        Parameters
        ----------
        point_row : pd.Series
            Row of a pandas dataframe

        Returns
        -------
        Point_FAS
            Point-level FAS
        """
        ts = point_row["timestamp"]

        return cls(
            agent=point_row["agent"],
            timestamp=ts,
            is_anomalous=point_row["is_anomalous"],
            anomaly_score=point_row["anomaly_score"]
        )

