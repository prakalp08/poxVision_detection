from poxVisionDetection.config.configuration import ConfigurationManager
from poxVisionDetection.components.prepare_callbacks import PrepareCallback
from poxVisionDetection.components.training import Training
from poxVisionDetection import logging,CustomException
import sys

STAGE_NAME = 'TRAINING'

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config                        = ConfigurationManager()
        prepare_callbacks_config      = config.get_prepare_callbacks_config()
        prepare_callbacks             = PrepareCallback(config = prepare_callbacks_config)
        callback_list                 = prepare_callbacks.get_tb_ckpt_callbacks()
        training_config               = config.get_training_config()
        training                      = Training(config = training_config)

        training.get_base_model()
        training.training_valid_generator()

        training.train_model_status(
            callback_list=callback_list
        )


if __name__ == '__main__':
    try:
        logging.info(f'\n >>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        MTobj = ModelTrainingPipeline()
        MTobj.main()
        logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
    except Exception as e:
        logging.exception(CustomException(e,sys))