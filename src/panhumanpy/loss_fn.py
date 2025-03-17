
import tensorflow as tf
from tensorflow import keras
import numpy as np

list_of_loss_fns = ['level_wt_focal_loss']
        


@keras.utils.register_keras_serializable(package="my_losses", name="level_wt_focal_loss")
class LvlWtFocalLoss(tf.keras.losses.Loss):

        def __init__(self, level_wt, gamma=2., **kwargs):

            super().__init__(**kwargs)

            self.gamma = gamma
            self.level_wt = level_wt


        def call(self, y_true, y_pred):

            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            cross_entropy = -y_true * tf.math.log(y_pred)
            loss = tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

            return self.level_wt*tf.reduce_sum(loss, axis=-1)


        def get_config(self):

            base_config = super().get_config()

            return {**base_config, "level_wt": self.level_wt, "gamma": self.gamma}


