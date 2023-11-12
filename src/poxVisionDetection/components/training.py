from poxVisionDetection.entity.config_entity import TrainingConfig
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

class Training:
    def __init__(self, config : TrainingConfig):
        self.config = config

    def get_base_model(self):
        # LOADING THE UPDATED BASE MODEL
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def training_valid_generator(self):
        valid_datagenerator         = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function     = preprocess_input,
            shear_range                = 0.2,
            zoom_range                 = 0.2,
            validation_split           = 0.4,
        )

        # THIS GENERATOR HAS BEEN CREATED FOR THE TRAINING
        self.train_generator        = valid_datagenerator.flow_from_directory(
            directory                   = self.config.training_data,
            target_size                 = self.config.params_image_size[:-1],
            batch_size                  = self.config.params_batch_size,
            class_mode                  = 'categorical',
            subset                      = 'training',
        )

        # THIS GENERATOR HAS BEEN CREATED FOR THE VALIDATION
        self.valid_generator        = valid_datagenerator.flow_from_directory(
            directory                  = self.config.training_data,
            target_size                = self.config.params_image_size[:-1],
            batch_size                 = self.config.params_batch_size,
            class_mode                 = 'categorical',  
            subset                     = 'validation',
        )

    def train(self, callback_list : list):
        trained_model = self.model.fit(

            self.train_generator,

            epochs                       = self.config.params_epochs,
            steps_per_epoch              = 3,

            validation_data              = self.valid_generator,
            validation_steps             = 2,

            callbacks                    = callback_list
        )

        self.save_model(
            path                     = self.config.training_model_path,
            model                    = self.model
        )
        
        return trained_model

    def train_model_status(self, callback_list: list):
        trained_model = self.train(callback_list)
        plt.plot(trained_model.history['accuracy'])
        plt.plot(trained_model.history['val_accuracy'])
        plt.axis(ymin=0.0,ymax=1)
        plt.grid()
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train','validation'])
        plt.show()

    @staticmethod
    def save_model(path : Path, model : tf.keras.Model):
        model.save(path)