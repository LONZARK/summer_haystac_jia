'''
Convert the parquet files that have agent id in the name to parquet
files with agent id in the table - also changing the id to int type
'''
from glob import glob
import pandas as pd
from os.path import basename
from time import time
from multiprocessing import Pool
import os
from tqdm import tqdm

input_dir = "./trajectories/"
output_dir = "./translated_files/"


def get_agent_namer():
    """Keeps track of agent naming based on integers

    Returns
    -------
    class
        an instance of the agent namer
    """
    id_tracker = {}
    counter = 1  # start this at 1 since no agent can be zero

    def get_agent_name(agent_uuid):
        nonlocal counter
        nonlocal id_tracker
        if agent_uuid not in id_tracker.keys():
            id_tracker[agent_uuid] = counter
            counter = counter + 1
        return id_tracker[agent_uuid]
    return get_agent_name


def extract_agent_uuid(agent_path):
    """Gets the uuid from the filename

    Parameters
    ----------
    agent_path : str
        path to the file

    Returns
    -------
    str
        uuid
    """
    return basename(agent_path).split('.')[0]


def get_pandas_from_parquet(filename):
    """Opens a parquet file adds the agent id column
    and returns a pandas dataframe

    Parameters
    ----------
    filename : str
        the filename

    Returns
    -------
    pandas.DataFrame
        pandas dataframe
    """
    global agent_namer
    df1 = pd.read_parquet(filename)
    agent_uid = str(extract_agent_uuid(filename))
    # print(f"agent_uid = {agent_uid}")
    agent = agent_namer(agent_uid)
    # print(f"agent_id = {agent}")
    # add the agent id column
    df1.insert(loc=0,
               column='agent',
               value=agent)
    # assert len(df1) != 604800
    if len(df1) > 604800:
        start = len(df1) - 604800
        df1 = df1[start:]
    return df1


def translate_parquet(filename):
    """Takes a parquet file that has the agent name in the filename
    and makes a new copy that has the agent id in the first column

    Parameters
    ----------
    filename : str
        the file

    Returns
    -------
    tuple(str, int)
        the original uuid and the new agent id (int)
    """
    df = get_pandas_from_parquet(filename)
    # save the new file
    new_file = output_dir + basename(filename)
    df.to_parquet(new_file)
    return (basename(filename), df['agent'].iloc[0])


if __name__ == '__main__':
    global agent_namer
    agent_namer = get_agent_namer()

    start_time = time()

    files = glob(f'{input_dir}/*.parquet')

    # make the folder
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    r = []
    '''
    # not using multiprocessing because the naming is not thread safe
    with Pool(20) as p:
        r = list(tqdm(p.imap(translate_parquet, files),
                      total=len(files)))
    '''
    for file in tqdm(files):
        r.append(translate_parquet(file))
    sr = set(r)
    # save the map of agent ids to uids
    with open('./agent_ids_uids.csv', 'w') as af:
        for item in sr:
            af.write(f"{item[1]},{item[0]}\n")

    print(f"final time: {time() - start_time}")
