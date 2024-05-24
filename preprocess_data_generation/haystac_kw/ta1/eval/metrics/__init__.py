from .metrics import (
    EncounterFrequencyMetric,
    TotalDistancePerDayMetric,
    RadiusOfGyrationMetric,
    NumberLocationsVisitedMetric,
    TemporalVariabilityMetric,
    LevelOfExplorationMetric,
    InterEncounterTimeMetric,
    OriginDestinationProbabilityMetric,
    LocationConnectivityMetric)
from enum import Enum

# All metrics available
all_metrics = [
    EncounterFrequencyMetric,
    TotalDistancePerDayMetric,
    RadiusOfGyrationMetric,
    NumberLocationsVisitedMetric,
    TemporalVariabilityMetric,
    LevelOfExplorationMetric,
    InterEncounterTimeMetric,
    OriginDestinationProbabilityMetric,
    LocationConnectivityMetric]

# Metrics that can be updated incrementally
# per agent
incremental_metrics = [
    TotalDistancePerDayMetric,
    RadiusOfGyrationMetric,
    NumberLocationsVisitedMetric,
    TemporalVariabilityMetric,
    LevelOfExplorationMetric,
    OriginDestinationProbabilityMetric,
    LocationConnectivityMetric
]


class MetricsGroups(Enum):
    ALL_METRICS = all_metrics
    INCREMENTAL_METRICS = incremental_metrics
