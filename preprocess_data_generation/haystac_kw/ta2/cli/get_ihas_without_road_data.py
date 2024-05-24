import click
import pandas as pd
from pathlib import Path
from haystac_kw.data.schemas.has import InternalHAS


@click.command()
@click.argument('ihas_file')
@click.argument('stop_point_file')
@click.argument('output_has_file')
def get_file(ihas_file, stop_point_file, output_has_file):
    """
    Command line tool to remove road info from an exisitng iHAS file

    Parameters
    ----------
    ihas_file : str
        The internal HAS file

    stop_point_file: str
        The stop point parquet file that correlates to this iHAS

    output_has_file: str
        The resulting iHAS to save to
    """

    # open the existing has
    has = InternalHAS.from_json(Path(ihas_file).read_text())

    # open the stop points
    stop_point_table = pd.read_parquet(stop_point_file)

    has.drop_road_edge_instructions(stop_point_table)

    # save the new file
    output_has_file = Path(output_has_file)
    output_has_file.parent.mkdir(exist_ok=True)
    output_has_file.write_text(has.model_dump_json())

if __name__ == "__main__":
    get_file()
