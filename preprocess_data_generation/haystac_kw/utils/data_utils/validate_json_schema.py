import argparse
import json
import jsonschema

from typing import Union, Tuple
from jsonschema import validate


def get_json_data(filename: str) -> dict:
    """This function loads the given json data based on file path

    Parameters
    ----------
    filename : str
        json filename to load

    Returns
    -------
    dict
        json data as a Python dictionary object
    """

    with open(filename, 'r') as file:
        schema = json.load(file)
    return schema


def validate_json(json_file: Union[str, dict], schema_file: str) -> Tuple[bool, str] :
    """Validates a json file given another input json schema file.

    Parameters
    ----------
    json_file : str or dict
        path of json to validate
    schema_file : str
        path of json schema to validate with

    Returns
    -------
    (bool, str)
        - boolean indicating whether the json is valid,
        - string message with details
    """

    # Describe what kind of json you expect.
    execute_api_schema = get_json_data(schema_file)
    if type(json_file) is dict:
        json_data = json_file
    else:
        json_data = get_json_data(json_file)

    try:
        validate(instance=json_data, schema=execute_api_schema)
    except jsonschema.exceptions.ValidationError as err:
        print(err)
        err = "Given JSON data is InValid"
        return False, err

    message = "Given JSON data is Valid"
    return True, message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='JSON Validator',
                        description='Validates a given JSON \
                            against a given schema')
        
    parser.add_argument('-j', '--json', 
                        required=True, 
                        help="Path of JSON file")
    parser.add_argument('-s', '--schema', 
                        required=True, 
                        help="Path of schema file")

    args = parser.parse_args()

    print(validate_json(args.json, args.schema))

