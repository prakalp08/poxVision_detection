import sys
from poxVisionDetection import logging,CustomException
from poxVisionDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from poxVisionDetection.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from poxVisionDetection.pipeline.stage_03_model_training import ModelTrainingPipeline
from poxVisionDetection.pipeline.stage_04_model_evaluation import EvaluationPipeline

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


STAGE_NAME = 'TRAINING'

try:
    logging.info(f'\n >>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
    MTobj = ModelTrainingPipeline()
    MTobj.main()
    logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
except Exception as e:
        logging.exception(CustomException(e,sys))

STAGE_NAME = 'EVALUATION PIPELINE'

try:
    logging.info(f'\n >>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
    EVLobj = EvaluationPipeline()
    EVLobj.main()
    logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
except Exception as e:
    logging.exception(CustomException(e,sys))