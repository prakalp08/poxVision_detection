import sys
from poxVisionDetection import logging,CustomException
from poxVisionDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from poxVisionDetection.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "DATA INGESTION STAGE"

try:
    logging.info(f'\n >>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<< \n')
    DIobj = DataIngestionTrainingPipeline()
    DIobj.main()
    logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
except Exception as e:
    logging.exception(e)
    CustomException(e,sys)


STAGE_NAME = 'PREPARE BASE MODEL'

try:
    logging.info(f'\n\n >>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
    PBMobj = PrepareBaseModelTrainingPipeline()
    PBMobj.main()
    logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
except Exception as e:
    logging.exception(CustomException(e,sys))
