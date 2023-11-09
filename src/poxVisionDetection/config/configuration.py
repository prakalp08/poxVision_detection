from poxVisionDetection.constants import *
from poxVisionDetection.utils.common import read_yaml,create_directory
from poxVisionDetection.entity.config_entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directory([self.config.artifacts_root])  # THIS WILL CREATE THE PARENT DIRECTORY artifacts 
                                                        # WHERE ALL THE DATA RELATED FOLDERS WILL BE PRESENT

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        '''
            Will get all the data_ingestion related configuration form the config file 
        '''
        config = self.config.data_ingestion

        create_directory([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir            = config.root_dir,
            source_url          = config.source_url,
            local_data_file     = config.local_data_file,
            unzip_dir           = config.unzip_dir
        )

        return data_ingestion_config