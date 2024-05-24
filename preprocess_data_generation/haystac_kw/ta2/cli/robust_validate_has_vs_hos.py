import click
from pathlib import Path
from haystac_kw.ta2.validation.validate_has import robust_validate_ihas
from haystac_kw.data.schemas.hos import InternalHOS
from haystac_kw.data.schemas.has import InternalHAS
import os
import glob

@click.command()
@click.argument('ihas_file_or_folder')
@click.argument('ihos_file_or_folder')
@click.option('--road_network', default="", required=False)
@click.option('--poi', default="", required=False)
@click.option('--iterations', default=1000, required=False)
def validate(ihas_file_or_folder, ihos_file_or_folder, road_network,
             poi, iterations):
    """
    Command line tool to check HAS files vs. the HOS specification

    Parameters
    ----------
    ihas_file_or_folder : str
        The internal HAS file to check (or a folder of iHAS files)

    ihos_file_or_folder : str
        The internal HOS file to check against (or a folder of iHOS files)

    road_network : str
        The road network pkl file to use for estimating travel time (optional)

    poi : str
        The POI pkl file used to resolve Points Of Interest (optional)

    iterations : int
        How many iterations to test for (optional)
    """

    if poi == "":
        poi = None
    if road_network == "":
        road_network = None

    hoss = []
    hass = []
    if os.path.isdir(ihas_file_or_folder):
        hass = sorted(glob.glob(ihas_file_or_folder + "/*.json"))
        hoss = sorted(glob.glob(ihos_file_or_folder + "/*.json"))
    else:
        hass = [ihas_file_or_folder]
        hoss = [ihos_file_or_folder]

    has_count = 0
    hos_count = 0
    for i in range(len(hoss)):
        ihos_file = hoss[hos_count]
        hos_count += 1
        ihas_file = hass[has_count]

        # check the basenames match
        name = os.path.basename(ihas_file)
        hos_name = os.path.basename(ihos_file).split("_")[-1]
        if name.split('_')[-1] != hos_name:
            print(f"missing HAS for {hos_name}")
            continue
        has_count += 1

        hos = InternalHOS.from_json(Path(ihos_file).read_text())
        has = InternalHAS.from_json(Path(ihas_file).read_text().replace('DURATION','duration').replace('END_TIME','end_time'))
        print(f"reading HAS {name}")
        score = robust_validate_ihas(hos, has, road_network, poi, num_iter=iterations)
        print(f"  robustness (0-1) for {hos_name} = {score}")

if __name__ == "__main__":
    validate()
