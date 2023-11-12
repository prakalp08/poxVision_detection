import tensorflow as tf
from pathlib import Path
from poxVisionDetection.entity.config_entity import EvaluationConfig
from poxVisionDetection.utils.common import save_json
from tensorflow.keras.applications.resnet50 import preprocess_input

class Evaluation:
    def __init__(self,config : EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        valid_datagenerator         = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function     = preprocess_input,
            shear_range                = 0.2,
            zoom_range                 = 0.2,
            validation_split           = 0.4,
        )

        self.valid_generator        = valid_datagenerator.flow_from_directory(
            directory                  = self.config.training_data,
            target_size                = self.config.params_image_size[:-1],
            batch_size                 = self.config.params_batch_size,
            class_mode                 = 'categorical',  
            subset                     = 'validation',
        )

    @staticmethod
    def load_model(path : Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model                   = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score                   = model.evaluate(self.valid_generator)

    def save_score(self):
        score = {'loss' : self.score[0], 'accuracy' : self.score[1]}
        save_json(path = Path('score.json'), data = score)