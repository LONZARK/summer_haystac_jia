import click
import pickle
import networkx
from networkx import DiGraph
import pandas as pd
from pathlib import Path
import numpy as np

@click.command()
@click.argument('road_network_pickle', type=Path)
@click.argument('poi_pickle', type=Path)
def main(road_network_pickle: Path, poi_pickle: Path):
    """
    Script to validate that a road network pickle and a poi
    pickle file are compatible

    Parameters
    ----------
    road_network_pickle : Path
        road_network.pkl file containing a road network DiGraph
    poi_pickle : Path
        poi.pkl file with poi referenced to the road network using
    """

    road_network = pickle.load(road_network_pickle.open('rb'))
    poi = pd.read_pickle(poi_pickle)

    poi_ref_road_ids = poi.id_road.unique()
    ids_in_road_network = \
        [int(x[2]['id']) for x in iter(road_network.edges(data=True)) if 'id' in x[2]]
    for x in poi_ref_road_ids:
        x = int(x)
        assert int(x) in ids_in_road_network
    print('Validated POI and ROAD NETWORK pickle files')


if __name__ == "__main__":

    main()
