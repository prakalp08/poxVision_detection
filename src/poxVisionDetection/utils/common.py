import os 
import sys
from box.exceptions import BoxValueError 
import yaml
from poxVisionDetection import CustomException,logging
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any 
import base64

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    '''
        Reads yaml file and returns 
        
        Args: 
            path_to_yaml (str) : Path like input, OS independant

        Raise:
            CustomException -> for any kind of exception 

        Return 
            configBox: ConfigBox type 
    '''
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info("YAML FILE {path_to_yaml} LOADED SUCCESSFULLY ")
            return ConfigBox(content)
    except Exception as e:
        CustomException(e,sys)

@ensure_annotations
def create_directory(path_to_directory : list, verbose=True):
    '''
        Creates a list of directory 

        args:
            path_to_directory         : List of path of directory 
    '''
    for path in path_to_directory:
        os.makedirs(path, exist_ok = True)
        if verbose:
            logging.info(f"CREATED DIRECTORY AT : {path}")

@ensure_annotations
def save_json(path : Path, data : dict):
    '''
        save json data

        args:
            path (Path) : Path to the json file
            data (dict) : Data to be saved in json file
    '''

    with open(path , 'w') as f:
        json.dump(data, f , indent = 4)

@ensure_annotations
def load_json(path : Path) -> ConfigBox:
    '''
        load json file data

        Args:
            path (Path) : Path to json file

        Returns:
            ConfigBox : Data as class attributes insted of dict 
    '''

    with open(path) as file:
        content = json.load(file)

    logging.info(f"JSON FILE HAS BEEN LOADED SUCCESSFULLY FROM THE PATH {path}")
    return ConfigBox(file)

@ensure_annotations
def get_size(path : Path) -> str:
    '''
        Get size in KB

        Args:
            Path (Path) : path of the file

        Returns:
            str: Size in KB
    '''

    size_in_kb = round(os.path.getsize(path)/1024)
    return f'~ {size_in_kb} KB'

def decodeImage(imagestring, filename):
    imagedata = base64.b64decode(imagestring)
    with open(filename, 'wb') as f:
        f.write(imagedata)
        f.close()
