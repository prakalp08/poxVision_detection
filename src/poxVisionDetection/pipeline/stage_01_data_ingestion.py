from poxVisionDetection.config.configuration import ConfigurationManager
from poxVisionDetection.components.data_ingestion import DataIngestion
from poxVisionDetection import logging,CustomException
import sys

STAGE_NAME = "DATA INGESTION STAGE"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config                         = ConfigurationManager()
        data_ingestion_config          = config.get_data_ingestion_config()
        data_ingestion                 = DataIngestion(config = data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logging.info(f'\n >>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        DIobj = DataIngestionTrainingPipeline()
        DIobj.main()
        logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
    except Exception as e:
        logging.exception(e)
        CustomException(e,sys)