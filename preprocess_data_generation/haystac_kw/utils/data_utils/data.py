import os, sys
import glob
import re
import copy
import collections
import warnings
import time
import gzip
import pyarrow
import datetime

from enum import Enum
from lxml import etree

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyspark.pandas as ps
from pyspark import SparkContext
from tqdm import tqdm
from shapely import Point
import geopandas as gpd
import xml.etree.ElementTree as ET

from haystac_kw.third_party_ext.sumo.get_osm import get_osm
# from haystac_kw.third_party_ext.sumo.get_osm import osmGet

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../.."))

def examine_parquet_data(path : str):
    """Print parquet data.

    Parameters
    ----------
    path : str
        parquet file path to examine
    """
    df = pd.read_parquet(path)

    print(df)
    print()

def load_parquet_as_dataframe(path: str, projection:str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """
    Load a parquet with columns `timestamp`,`latitude`,`longitude`. And
    return a GeoDataFrame with columns `timestamp`,`geometry`.

    Parameters
    ----------
    path : str
        Path to parquet file
    projection : str, optional
        Projection to use for geometry, by default 'EPSG:4326'

    Returns
    -------
    gpd.GeoDataFrame
        Parsed GeoDataFrame
    """
    trajectory = pd.read_parquet(path)
    trajectory = gpd.GeoDataFrame(trajectory, geometry=[Point(x,y) for x,y in zip(trajectory.longitude, trajectory.latitude)],
                                crs=projection)
    
    return trajectory[['timestamp','geometry']]

def get_unixtime(dt64):
    """
    Convert pandas datetime import UNIX timestamp int
    https://stackoverflow.com/questions/11865458/how-to-get-unix-timestamp-from-numpy-datetime64

    :param dt64: pandas datetime array
    :type dt64: np.ndarray
    :return: Array of unix timestamp ints
    :rtype: np.ndarray
    """
    return dt64.astype('datetime64[s]').astype('int')

class SimDataTypes(Enum):
    ULLT = 1
    SUMO = 2
    HUMONET = 3

class SimData:
    """Simulation data class; holds information on the simulation data and is primarily used for file conversion / output."""
    def __init__(self, datadir):
        """
        Initializes the simulation data class. 

        Parameters
        ----------
        datadir : path-like str
            directory containing the data the class represents.
        """
        self.datadir = datadir
        self.dtype = -1
        self.data = None

        datetime_str = str(datetime.date.today()) + ' 00:00:00'
        self.date_time = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

        self.parse_data()

        print("initializing data converter")
        print("============================")
        print("src data: %s\n" % (datadir))

    def parse_data(self):
        """The goal of this method is to populate self.data with a Pandas Dataframe object.
        All subclasses of SimData must implement some version of this.
        """
        pass

    def convert(self, target_dtype: SimDataTypes):
        """Converts data in self.data to a target data type. Unimplemented / unused so far in this implementation.
        TODO: Define more SimDataTypes and implement examples of this function.

        Parameters
        ----------
        target_dtype : SimDataTypes
            Target data type (defaults to SimDataTypes)
        """
        pass

    def export(self, savedir, format='parquet'):
        """Exports the data stored in the object to parquet format.

        Parameters
        ----------
        savedir : str
            directory to save the parquet files
        format : str, optional
            format of data to export as (defaults to 'parquet')

        Raises
        ------
        NotImplementedError
            Raises an error when ``format`` is unsupported
        """
        os.makedirs(savedir, exist_ok=True)
        savepath = savedir

        SparkContext.setSystemProperty('spark.executor.memory', '12g')

        sc = SparkContext("local", "App Name")

        if format == 'parquet':

            df = ps.from_pandas(self.data)
            df.to_parquet(savepath,
                          compression='gzip',
                          partition_cols='uid')

            # Commented: Alternative compression implementation. Use if above doesn't work
            # self.data.to_parquet(savepath,
            #                     compression='gzip',
            #                     partition_cols=['uid'])

        else:
            raise NotImplementedError

        SparkContext.stop(sc)

        print('export finished. saved to: %s\n' % (savepath))


class SUMOData(SimData):
    """Inherits the SimData class. The SUMOData class represents the SUMO FCD data type, and handles conversions accordingly."""
    def __init__(self, datadir):
        """
        Initializes the SUMO Sim Data class. 

        Parameters
        ----------
        datadir : path-like str
            directory path containing the data to be represented by this object
        """
        super().__init__(datadir)

        self.dtype = SimDataTypes.SUMO

    def parse_data(self):
        """Parses the data defined in the constructor and stores it in the data instance variable."""
        data = []

        xml_path = os.path.join(self.datadir, "output.fcd_data.xml")

        total_timesteps = 86400

        context = ET.iterparse(xml_path, events=("start", "end"))
        context = iter(context)
        ev, root = next(context)

        pbar = tqdm(total=total_timesteps)
        progress = 0

        veh_id_set = set()

        for ev, el in context:
            if ev == 'start' and el.tag == 'timestep':
                for veh in el:
                    attributes = veh.attrib
                    # print(str(attributes['id'][3:]))
                    data.append([str(attributes['id'][3:]),
                                float(attributes['y']),
                                float(attributes['x']),
                                int(float(el.attrib['time']))])

                    veh_id_set.add(str(attributes['id'][3:]))

                root.clear()
                pbar.update(1)
                progress += 1
                print(len(veh_id_set))

        pbar.close()

        self.data = pd.DataFrame(data, columns = ['uid','latitude','longitude', 'timestamp'])
        self.data = self.data.sort_values(['uid', 'timestamp'])

        formatted_ts = self.date_time + self.data['timestamp'].apply(lambda x: datetime.timedelta(seconds=x))
        formatted_ts = formatted_ts.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        self.data['timestamp'] = formatted_ts


# https://python-forum.io/thread-6204.html
def print_xml_structure(xml_path):
    """Print the FCD XML file that is outputted by SUMO.

    Parameters
    ----------
    xml_path : str
        path of the XML file
    """
    xml_root = etree.parse(xml_path)
    nice_tree = collections.OrderedDict()

    for tag in xml_root.iter():
        path = re.sub('\[[0-9]+\]', '', xml_root.getpath(tag))
        if path not in nice_tree:
            nice_tree[path] = []
        if len(tag.keys()) > 0:
            nice_tree[path].extend(attrib for attrib in tag.keys() if attrib not in nice_tree[path])

    for path, attribs in nice_tree.items():
        indent = int(path.count('/') - 1)
        print('{0}{1}: {2} [{3}]'.format('    ' * indent, indent, path.split('/')[-1], ', '.join(attribs) if len(attribs) > 0 else '-'))

def get_osm_data(args):
    """Launch the OSM grabber web app and save the data the user selected.

    Parameters
    ----------
    args : argparse.Namespace
        Commandline arguments

    Returns
    -------
    str
        Directory that the OSM data is saved to
    """
    args_cpy = copy.deepcopy(args)
    args_cpy.outputDir = os.path.abspath(os.path.join(args.outputDir, "osm_tmp"))

    osm_dir = args_cpy.outputDir

    print("Saving OSM data to: %s" % (args_cpy.outputDir))
    get_osm(args_cpy)

    return osm_dir

def get_bbox_from_osm(osm_file):
    """Returns the bounding box [west, south, east, north] contained in a OSM file

    Parameters
    ----------
    osm_file : str
        Path to an OSM file

    Returns
    -------
    list
        bounding box [west, south, east, north] contained in ``osm_file``
    """
    tree = ET.parse(osm_file)
    root = tree.getroot()

    lats, lons = [], []
    for node in root.findall('node'):
        lats.append(float(node.get('lat')))
        lons.append(float(node.get('lon')))
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # lat = y, lon = x
    bbox = [min_lon, min_lat, max_lon, max_lat]
    
    print(bbox)
    return bbox

def get_osm_from_bbox(input_args):
    """Get the OSM data based on the bounding box information.

    Parameters
    ----------
    input_args : argparse.Namespace
        Commandline arguments   

    Returns
    -------
    str
        output directory of the saved data
    """
    output_dir = os.path.abspath(os.path.join(input_args.outputDir, "osm_tmp"))
    os.makedirs(output_dir, exist_ok=True)

    bbox = ' '.join(input_args.bbox)
    opts = ['--bbox', input_args.bbox[0], input_args.bbox[1], input_args.bbox[2], input_args.bbox[3],
            '--output-dir', output_dir,
            '--gzip', '--shapes']
    print('osmGet.py args: ', opts)
    options = osmGet.get_options(args=opts)

    osmGet.get(options)

    return options.output_dir

def decompress_gz_files(osm_dir):
    """Decompress the GZ files saved by get_osm.

    Parameters
    ----------
    osm_dir : str
        Directory containing the OSM / GZ compressed files.

    Returns
    -------
    tuple(str, str, str)
        - osmFile : OSM file path
        - netFile : Net file path
        - polyFile : Polygon file path
    """
    assert osm_dir is not None, "osm data missing from %s; \nplease download it first with get_osm_data"

    print("decompressing files.. ")
    time.sleep(1)

    osm_dir = osm_dir

    gz_files = [os.path.join(osm_dir, "osm_bbox.osm.xml.gz"),
                os.path.join(osm_dir, "osm.net.xml.gz"),
                os.path.join(osm_dir, "osm.poly.xml.gz")]

    outputs = []

    for gz in gz_files:
        if not os.path.exists(gz):
            warnings.warn(f'File {gz} was not found... skipping')
            outputs.append(None)
            continue

        target = gz.replace(".xml.gz", "")
        with open(gz, 'rb') as gz_f, open(target, 'w', encoding='utf8') as tof:
            decom_str = gzip.decompress(gz_f.read()).decode('utf-8')
            tof.write(decom_str)

        outputs.append(target)

    [osmFile, netFile, polyFile] = outputs
    return osmFile, netFile, polyFile
