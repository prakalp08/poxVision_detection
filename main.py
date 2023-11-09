import sys
from poxVisionDetection import logging,CustomException
from poxVisionDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "DATA INGESTION STAGE"

try:
    logging.info(f'\n >>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<< \n')
    DIobj = DataIngestionTrainingPipeline()
    DIobj.main()
    logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
except Exception as e:
    logging.exception(e)
    CustomException(e,sys)