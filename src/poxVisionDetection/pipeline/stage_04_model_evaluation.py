from poxVisionDetection.config.configuration import ConfigurationManager
from poxVisionDetection.components.evaluation import Evaluation
from poxVisionDetection import logging,CustomException
import sys

STAGE_NAME = 'EVALUATION PIPELINE'

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config         = ConfigurationManager()
        val_config     = config.get_validation_config()
        evaluation     = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()

if __name__ == "__main__":
    try:
        logging.info(f'\n >>>>>>>>>>>>>>>>>>>>>>>> {STAGE_NAME} <<<<<<<<<<<<<<<<<<<<<<<<<<< \n')
        EVLobj = EvaluationPipeline()
        EVLobj.main()
        logging.info(f'\n >>>>>>>>>>>>> {STAGE_NAME} >>>>>>> [COMPLETED] <<<<<<<<<<<<<<<<<<<\n\nX===============================================================================X')
    except Exception as e:
        logging.exception(CustomException(e,sys))