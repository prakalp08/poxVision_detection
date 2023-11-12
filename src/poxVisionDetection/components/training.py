from poxVisionDetection.entity.config_entity import TrainingConfig
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
            rescale                    = 1.0/255,
            validation_split           = 0.20,
        )

        # THIS VALID GENERATOR IS CREATED FOR VALIDATION COMPLETELY NEW SET OF IMAGES
        self.valid_generator        = valid_datagenerator.flow_from_directory(
            directory                  = self.config.training_data,
            target_size                = self.config.params_image_size[:-1],
            batch_size                 = self.config.params_batch_size,
            interpolation              = 'bilinear',       
            subset                     = 'validation',
            shuffle                    = False,
        )

        if self.config.params_is_augmentation:
            train_generator         = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale                = 1.0/255,
                validation_split       = 0.2,
                rotation_range         = 40,
                horizontal_flip        = True,
                width_shift_range      = 0.2,
                height_shift_range     = 0.2,
                shear_range            = 0.2,
                zoom_range             = 0.2,
            )
        else:
            train_generator            = valid_datagenerator

        # THIS GENERATOR IS MADE FOR TRAINING 
        self.train_generator        = train_generator.flow_from_directory(
            target_size                 = self.config.params_image_size[:-1],
            batch_size                  = self.config.params_batch_size,
            directory                   = self.config.training_data,
            interpolation               = 'bilinear',
            subset                      = 'training',
            shuffle                     = True,
        )

    def train(self, callback_list : list):
        self.steps_per_epoch         = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps        = self.valid_generator.samples // self.valid_generator.batch_size

        trained_model = self.model.fit(
            self.train_generator,
            epochs                       = self.config.params_epochs,
            steps_per_epoch              = self.steps_per_epoch,
            validation_steps             = self.validation_steps,
            validation_data              = self.valid_generator,
            callbacks                    = callback_list
        )
        return trained_model

    def train_model_status(self, callback_list: list):
        trained_model = self.train(callback_list)
        plt.plot(trained_model.history['accuracy'])
        plt.plot(trained_model.history['val_accuracy'])
        plt.axis(ymin=0.4,ymax=1)
        plt.grid()
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train','validation'])
        plt.show()

        self.save_model(
            path                     = self.config.training_model_path,
            model                    = self.model
        )

    @staticmethod
    def save_model(path : Path, model : tf.keras.Model):
        model.save(path)