from pathlib import Path
import click
import geopandas as gpd

from haystac_kw.data.schemas.has import HAS


@click.command()
@click.argument('has_dir', type=Path)
@click.argument('sumo_has_dir', type=Path)
@click.argument('road_network_file', type=Path)
@click.option('--id-column', type=str, default='id')
def main(
        has_dir: Path,
        sumo_has_dir: Path,
        road_network_file: Path,
        id_column: str):
    """
    Script to convert a directory of HAS files to a format
    readable into SUMO (referencing road network)

    Parameters
    ----------
    has_dir : Path
        Directory of HAS json files
    sumo_has_dir : Path
        Directory to save out converted HAS files
    road_network_file : Path
        Shapefile describing the road network (id column must contain SUMO road edge id)
    id_column : str
        Column containing SUMO edge id, (defaults to `id`)
    """

    road_network = gpd.read_file(road_network_file)
    road_network['id'] = road_network[id_column].values
    sumo_has_dir.mkdir(exist_ok=True)

    for has_file in has_dir.glob('*.json'):

        has = HAS.from_json(has_file.read_text())
        sumo_has = has.convert_to_sumo(road_network)
        sumo_has_dir.joinpath(has_file.name).write_text(sumo_has)


if __name__ == "__main__":

    main()
