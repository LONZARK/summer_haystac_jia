import pandas as pd
import geopandas as gpd
from haystac_kw.data.schemas.hos import HOS
from haystac_kw.data.schemas.has import HAS
from shapely import LineString
from pathlib import Path
import click
from tqdm import tqdm
from multiprocessing import Pool
import os

def load_hos(file: Path):
    return HOS.from_json(file.read_text())

def load_has(file: Path):
    return HAS.from_json(file.read_text().replace(
                    'DURATION',
                    'duration').replace(
                    'END_TIME',
                    'end_time'))


@click.command()
@click.argument('trajectory_dir', type=Path)
@click.argument('instruction_log', type=Path)
@click.argument('hos_dir', type=Path)
@click.argument('has_dir', type=Path)
@click.argument('output_dir', type=Path)
def main(trajectory_dir: Path,
         instruction_log: Path,
         hos_dir: Path,
         has_dir: Path,
         output_dir: Path):

    # Check inputs
    assert trajectory_dir.is_dir(), f'Agent trajectory dir `{trajectory_dir}` does not exist.'
    assert instruction_log.is_file(), f'Instruction log `{instruction_log}` does not exist.'
    assert hos_dir.is_dir(), f'HOS dir `{hos_dir}` does not exist.'
    assert has_dir.is_dir(), f'HAS dir `{has_dir}` does not exist.'

    # Create output folder
    output_dir.mkdir(exist_ok=True)

    # Load up instruction log
    instruction_log_all = pd.read_parquet(instruction_log)
    instruction_log_all = instruction_log_all.dropna()

    with Pool(os.cpu_count()) as pool:
        # Load HOS/HAS
        print('Loading HOS')
        hos_files = list(hos_dir.glob('*.json'))
        hos = list(tqdm(pool.imap_unordered(load_hos, hos_files), total=len(hos_files)))
        print('Loaded HOS')
        print('Loading HAS')
        has_files = list(has_dir.glob('*.json'))
        has = list(tqdm(pool.imap_unordered(load_has, has_files), total=len(has_files)))
        print('Loaded HAS')
    
    # Find mapping of HOS to HAS
    has2hos = {}
    objective2has = {}
    for x in hos:
        for y in has:
            if str(y.objective) == str(x.objective_uid):
                print(y.objective)
                has2hos[str(y.objective)] = x
                objective2has[y.objective] = y

    # Find mapping of instructions to HAS
    instr2has = {}
    for x in has:
        for movement in x.movements:
            for itinerary in movement.itineraries:
                for instruction in itinerary.itinerary:
                    instr2has[str(instruction.instruction_uid)] = x

    # Get agent parquets
    agent_trajectory_files = {
       x.stem:x for x in trajectory_dir.glob('*.parquet')
    }

    print('Creating Visualizations')
    for agent_id in tqdm(instruction_log_all.agent.unique()):
        # Iterate through each agent and visualize each
        # trajectory for each HOS the agent participates in
        
        if agent_id not in agent_trajectory_files:
            continue
        

        # Agent could potentially participate in several HAS
        # so we need to account for that and process them separately
        instruction_log_agent = instruction_log_all[instruction_log_all.agent == agent_id]
        agent_has = set([instr2has[str(x)].objective for x in instruction_log_agent.instruction.values])
        
        for current_has in agent_has:
            # Process each HAS

            # Filter instructions to just this HAS
            current_has = objective2has[current_has]
            instruction_log = instruction_log_agent[
                [instr2has[x]==current_has for x in instruction_log_agent.instruction.values]
            ]
            # Get HOS
            hos = has2hos[str(current_has.objective)]

            # Get the time frame covering the HAS instructions in the
            # trajectory
            time_start = instruction_log.time_stop.min()
            time_stop = instruction_log.time_stop.max()

            # Load trajectory and filter to section controlled
            # by HAS
            traj = pd.read_parquet(agent_trajectory_files[agent_id])
            traj = traj[traj.timestamp >= time_start]
            traj = traj[traj.timestamp <= time_stop]
            traj = LineString([(x, y)
                            for x, y in zip(traj.longitude, traj.latitude)])

            # Containers for our geometries
            geoms, geom_type = [], []

            # Grab HAS locations
            for movement in current_has.movements:
                for itinerary in movement.itineraries:
                    for instruction in itinerary.itinerary:
                        if hasattr(instruction, 'location'):
                            if instruction.instruction_uid in instruction_log.instruction.values:
                                geom_type.append('executed has location')
                                geoms.append(
                                    instruction.location.geoms[0].buffer(.002))
                            else:
                                geom_type.append('not executed has location')
                                geoms.append(
                                    instruction.location.geoms[0].buffer(.001))

            # Add trajectory linestring
            geoms += [traj]
            geom_type += ['agent_trajectory']

            # Add HOS POI
            for event in hos.events:
                geom_type.append('hos poi')
                geoms.append(event.location.geoms[0].buffer(.003))

            # Display using geopandas and save off to html file
            gdf = gpd.GeoDataFrame({'geom_type': geom_type,
                                    'geometry': geoms},
                                geometry='geometry',
                                crs='EPSG:4326')
            folium_map = gdf.explore('geom_type', cmap=['blue', 'red', 'green', 'gray'])
            folium_map.save(output_dir.joinpath(f'{agent_id}_{hos.objective_uid}.html'))


if __name__ == "__main__":

    main()
