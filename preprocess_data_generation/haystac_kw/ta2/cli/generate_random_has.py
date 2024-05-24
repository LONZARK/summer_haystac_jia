""" This file generates random HAS given a folder of *original* government-given HOS files. 
In Kitware terms, it goes from external HOS --> internal HAS. The script also generates 
subfolders containing external Singapore HOS and internal Singapore HOS in addition to 
internal Singapore HAS. The location can be swapped by changing what the poi_pickle_path 
represents, i.e. swapping the poi file with buildings representing another location.

Usage example: 

 python haystac_ta1/haystac_kw/ta2/cli/generate_random_has.py \
    --output test \
    --gov-hos-dir [path/to/baseline_hos] \
    --poi-pickle-path [path/to/poi.pkl]
"""

import os 
import glob 
import sys 
import argparse 
from haystac_kw.utils.specifications.generate_singapore_hos import generate_all_hos_in_dir
from haystac_kw.utils.specifications.generate_random_has import *
from haystac_kw.utils.data_utils.conversions import hos2internal_hos
from haystac_kw.ta2.validation.validate_has import validate_has
from haystac_kw.ta2.cli.validate_has_vs_hos import validate

def main(args):
    
    gov_hos_savedir = os.path.join(args.output, "external_hos_singapore")

    # First, generate the Singapore HOS files from given HOS
    generate_all_hos_in_dir(args.gov_hos_dir, gov_hos_savedir, args.poi_pickle_path)
    
    # Next, convert government HOS for Singapore into internal HOS format 
    internal_hos_savedir = os.path.join(args.output, "internal_hos_singapore")
    hos2internal_hos(gov_hos_savedir, args.set_unique_sp_csv, internal_hos_savedir, 75)

    # hos2internalhos saves a stoppoints parquet file, we reference its path below
    stop_points_pq_path = os.path.join(internal_hos_savedir, "StopPoints.parquet")
    hos_list = glob.glob(os.path.join(gov_hos_savedir, "*.json"))
    internal_has_output_dir = os.path.join(args.output, "generated_internal_has", "ihas")

    # random HAS creation has gov HOS inputs, see hos_file arg
    for hos_file in hos_list: 
        create_n_random_has(
            args.poi_pickle_path,
            output_dir=internal_has_output_dir,
            n=args.num_generated,
            hos_file=hos_file,
            stop_point_file=stop_points_pq_path,
            internal_format=True)
    
    generated_has_directories = glob.glob(os.path.join(internal_has_output_dir+"*"))
    generated_internal_has_list = glob.glob(os.path.join(internal_has_output_dir+ "*", "*.json"))

    for has_output in generated_internal_has_list:

        hos_id_name = os.path.dirname(has_output).split("/")[-1].replace("ihas_", "") + ".json"
        corresponding_hos_file = os.path.join(internal_hos_savedir, hos_id_name)

        # TODO: make validation a util method and call it below instead of the block
        # validate(has_output, corresponding_hos_file, "True")
        has = {}
        hos = {}
        with open(corresponding_hos_file, 'r') as hf:
            hos = json.load(hf)
        with open (has_output, 'r') as hf:
            has = json.load(hf)

        # do a quick check that they gave the right files
        if "schema_type" not in has or "schema_type" not in hos:
            print("ERROR: schema_type not in file - are you sure these are HOS/HAS files?")
            return
        if hos['schema_type'] != "HOS":
            print(f"ERROR: the schema_type for {hos_file} is not HOS.")
            return
        if has['schema_type'] != "HAS":
            print(f"ERROR: the schema_type for {has_output} is not HAS.")
            return

        ret = validate_has(has, hos, verbose=False)
        if ret:
            print("The HAS matches the requirements of the HOS!")
        
        print("HAS file: ", has_output)
        print("HOS file: ", corresponding_hos_file)
        print()
    
    print("Done. See generated internal HAS files at the directories: ")
    
    for has_dir in generated_has_directories:
        print(has_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Random HAS generator',
        description='Generates random [INTERNAL] HAS files into a given directory. May or may not include a specified [EXTERNAL] HOS file.')

    parser.add_argument(
        '-o',
        '--output',
        default="baseline_random_internal_has_output",
        help="Directory to output HAS files to. Does not need to exist.")
    
    parser.add_argument(
        '--gov-hos-dir',
        required=True,
        help="Directory of government HOS files to generate internal HAS for.")
    
    parser.add_argument(
        '-p',
        '--poi-pickle-path',
        required=True,
        help="Path of POI pickle file file, required if HOS file is given. This file (typically poi.pkl) holds the geometry information of buildings to sample from.")
    
    parser.add_argument(
        '--set-unique-sp-csv',
        default="/mnt/krsdata2/other/projects/haystac/trial1_ta2_data/baseline_agent_set_unique_stop_point_table.csv",
        help="Location of set unique stop points CSV file. Typically named '*_agent_set_unique_stop_point_table.csv'.")
    
    parser.add_argument('-n', '--num-generated',
                        type=int,
                        default=3,
                        help="Number of random HAS to generate. Default: 10")
    
    args = parser.parse_args()

    main(args)