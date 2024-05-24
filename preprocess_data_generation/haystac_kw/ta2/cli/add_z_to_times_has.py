import click
import json
import glob
import os
from haystac_kw.utils.s3.s3_utils import cp_from_s3, cp_to_s3


@click.command()
@click.argument('input_has_folder')
@click.argument('output_has_folder')
def add_z(input_has_folder, output_has_folder):
    """
    Command line tool to add the missing Z to times for iHAS files

    Parameters
    ----------
    input_has_folder : str
        The folder of HAS files to edit

    output_has_folder: str
        The output folder
    """

    # if s3 bucket - setup local folders
    input_has_folder = cp_from_s3(input_has_folder)
    s3_output_dir = None
    if output_has_folder.startswith('s3://'):
        s3_output_dir = output_has_folder
        output_has_folder = "/tmp/output_dir"

    # add output folder if not there
    os.makedirs(output_has_folder, exist_ok=True)

    ins = glob.glob(input_has_folder + "/*.json")
    for hasf in ins:
        # read in the HAS
        has = {}
        with open(hasf, 'r') as hf:
            has = json.load(hf)
        basename = os.path.basename(hasf)

        # now parse through the structure looking for times
        for move_dict in has["movements"]:
            for itins_dict in move_dict["itineraries"]:
                for itin_dict in itins_dict["itinerary"]:
                    for cmd in itin_dict:
                        #print(f"cmd = {cmd}")
                        for item in itin_dict[cmd]:
                            if item == "end_time":
                                if "Z" not in itin_dict[cmd][item]:
                                    itin_dict[cmd][item] += "Z"
                            elif item == "time_window":
                                if "Z" not in itin_dict[cmd][item]["begin"]:
                                    itin_dict[cmd][item]["begin"] += "Z"
                                if "Z" not in itin_dict[cmd][item]["end"]:
                                    itin_dict[cmd][item]["end"] += "Z"

        # save the new file
        with open(output_has_folder + "/" + basename, 'w') as hf:
            json.dump(has, hf, indent=4)

        # copy up to s3 if the output was to a bucket
        if s3_output_dir:
            cp_to_s3(output_has_folder, s3_output_dir)

if __name__ == "__main__":
    add_z()
