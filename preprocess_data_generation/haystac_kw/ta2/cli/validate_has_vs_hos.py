import click
import json
from haystac_kw.ta2.validation.validate_has import validate_has


@click.command()
@click.argument('has_file')
@click.argument('hos_file')
@click.option('--verbose', default=False)
@click.option('--ignorez', default=False)
def validate(has_file, hos_file, verbose, ignorez):
    """
    Command line tool to check HAS files vs. the HOS specification

    Parameters
    ----------
    has_file : str
        The HAS file to check

    hos_file: str
        The HOS file to check against
    """

    has = {}
    hos = {}
    with open(hos_file, 'r') as hf:
        hos = json.load(hf)
    with open(has_file, 'r') as hf:
        has = json.load(hf)

    # do a quick check that they gave the right files
    if "schema_type" not in has or "schema_type" not in hos:
        print("ERROR: schema_type not in file - are you sure these are HOS/HAS files?")
        return
    if hos['schema_type'] != "HOS":
        print(f"ERROR: the schema_type for {hos_file} is not HOS.")
        return
    if has['schema_type'] != "HAS":
        print(f"ERROR: the schema_type for {has_file} is not HAS.")
        return

    ret = validate_has(has, hos, verbose=verbose, ignorez=ignorez)
    if ret:
        print("The HAS matches the requirements of the HOS!")


if __name__ == "__main__":
    validate()
