import os
import urllib.request as request
import zipfile 
from poxVisionDetection import logging,CustomException
from poxVisionDetection.utils.common import get_size
from poxVisionDetection.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config

    def download_file(self):
        '''
            will get the dataset for the remote git hub link provided
            
            Create local_file_folder where the file will be stored in .zip
        '''
        if not os.path.exists(self.config.local_data_file):
            filename, header = request.urlretrieve(
                url       = self.config.source_url,              # THE LINK WHERE THE FILE IS AVAILABLE IN THE GIT HUB
                filename  = self.config.local_data_file          # THE LOCAL PATH WHERE THE FILE WILL BE SAVED 
            )
            logging.info(f'{filename} DOWNLOADED WILL THE FOLLOWING INFO : {header}')
        else:
            logging.info(f'THE FILE ALREDY EXISTS OF SIZE : {get_size(Path(self.config.local_data_file))}')

    def extract_zip_file(self):
        ''' 
            zip_file_path  : str
            Extract the zip file into the data directory 
            function returns None
        '''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path , exist_ok = True)
        with zipfile.ZipFile(self.config.local_data_file , 'r') as zip_file:
            zip_file.extractall(unzip_path)
