from poxVisionDetection.config.configuration import ConfigurationManager
from poxVisionDetection.components.prepare_base_model import PrepareBaseModel
from poxVisionDetection import logging,CustomException
import sys 

STAGE_NAME = 'PREPARE BASE MODEL'

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config                       = ConfigurationManager()
        prepare_base_model_config    = config.get_prepare_base_model()
        prepare_base_model           = PrepareBaseModel(config = prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.updated_base_model()

if __name__ == '__main__':
    try:
        logging.info(f'\n >>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        PBMobj = PrepareBaseModelTrainingPipeline()
        PBMobj.main()
        logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
    except Exception as e:
        logging.exception(CustomException(e,sys))