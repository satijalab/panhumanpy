# write loss functions here

import tensorflow as tf
from tensorflow import keras
import numpy as np

list_of_loss_fns = ['focal_loss', 
                    'level_wt_focal_loss', 
                    'level_depth_wt_focal_loss',
                    'class_level_wt_focal_loss',
                    'log_class_level_wt_focal_loss']

@keras.utils.register_keras_serializable(package="my_losses", name="focal_loss")
class FocalLoss(tf.keras.losses.Loss):

        def __init__(self, gamma=2., **kwargs):

            super().__init__(**kwargs)

            self.gamma = gamma


        def call(self, y_true, y_pred):

            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            cross_entropy = -y_true * tf.math.log(y_pred)
            loss = tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

            return tf.reduce_sum(loss, axis=-1)


        def get_config(self):

            base_config = super().get_config()

            return {**base_config, "gamma": self.gamma}
        


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
        


@keras.utils.register_keras_serializable(package="my_losses", name="class_level_wt_focal_loss")
class ClassLvlWtFocalLoss(tf.keras.losses.Loss):

        def __init__(self, level_wt, class_wts, gamma=2., **kwargs):

            super().__init__(**kwargs)

            self.gamma = gamma
            self.level_wt = level_wt
            self.class_wts = class_wts


        def call(self, y_true, y_pred):

            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            cross_entropy = -y_true * tf.math.log(y_pred)
            loss = self.class_wts*tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

            return self.level_wt*tf.reduce_sum(loss, axis=-1)


        def get_config(self):

            base_config = super().get_config()

            return {**base_config, "level_wt": self.level_wt, "class_wts":self.class_wts, "gamma": self.gamma}
        

@keras.utils.register_keras_serializable(package="my_losses", name="log_class_level_wt_focal_loss")
class LogClassLvlWtFocalLoss(tf.keras.losses.Loss):

        def __init__(self, level_wt, class_wts, gamma=2., **kwargs):

            super().__init__(**kwargs)

            self.gamma = gamma
            self.level_wt = level_wt
            class_wts_min = np.min(class_wts)
            class_wts_resc = (np.e/class_wts_min)*class_wts
            self.log_class_wts = np.log(class_wts_resc)


        def call(self, y_true, y_pred):

            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            cross_entropy = -y_true * tf.math.log(y_pred)
            loss = self.log_class_wts*tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

            return self.level_wt*tf.reduce_sum(loss, axis=-1)


        def get_config(self):

            base_config = super().get_config()

            return {**base_config, "level_wt": self.level_wt, "class_wts":self.log_class_wts, "gamma": self.gamma}
        


@keras.utils.register_keras_serializable(package="my_losses", name="level_depth_wt_focal_loss")
class LvlDepthWtFocalLoss(tf.keras.losses.Loss):

        def __init__(self, level_wt, depth, depth_pow=0.5, gamma=2., **kwargs):

            super().__init__(**kwargs)

            self.gamma = gamma
            self.level_wt = level_wt
            self.depth = depth
            self.depth_pow = depth_pow


        def call(self, y_true, y_pred):

            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            cross_entropy = -y_true * tf.math.log(y_pred)
            loss = tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

            return self.level_wt*(self.depth**self.depth_pow)*tf.reduce_sum(loss, axis=-1)


        def get_config(self):

            base_config = super().get_config()

            return {**base_config, "level_wt": self.level_wt, "depth":self.depth, "depth_pow":self.depth_pow, "gamma": self.gamma}
        

@keras.utils.register_keras_serializable(package="my_losses", name="weighted_entropy_loss")
class WeightedProductEntropyLoss(tf.keras.losses.Loss):

        def __init__(self, relative_coeff, overall_coeff, **kwargs):

            super().__init__(**kwargs)

            self.relative_coeff = relative_coeff
            self.overall_coeff = overall_coeff


        def call(self, y_true, y_pred):

            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)
            
            loss = self.relative_coeff*self.overall_coeff*y_pred

            return tf.reduce_sum(loss, axis=-1)


        def get_config(self):

            base_config = super().get_config()

            return {**base_config, "relative_coeff": self.relative_coeff, "overall_coeff": self.overall_coeff}
        
@keras.utils.register_keras_serializable(package="my_losses", name="trivial_loss")
class TrivialLoss(tf.keras.losses.Loss):

        def __init__(self, **kwargs):

            super().__init__(**kwargs)


        def call(self, y_true, y_pred):

            
            return tf.zeros_like(y_true, dtype=tf.float32)


        def get_config(self):

            base_config = super().get_config()

            return base_config
        


# ##############################################################################
# The following is not used, and not updated.    
# ##############################################################################
# 
#     
#@keras.utils.register_keras_serializable(package="my_losses", name="fetch_loss")
class FetchLoss():

    def __init__(self, loss_name, **kwargs):
        self.loss_name = loss_name

        self.kwargs = kwargs

        if self.loss_name not in list_of_loss_fns:
            raise ValueError("Loss fn "+ self.loss_name + " has not been defined yet.")
        
    def eval(self):
                
        if self.loss_name == 'focal_loss':
            if 'gamma' in self.kwargs:
                gamma = self.kwargs['gamma']
            else:
                gamma = 2.
            
            def loss_func(y_true, y_pred):
                y_true = tf.convert_to_tensor(y_true, tf.float32)
                y_pred = tf.convert_to_tensor(y_pred, tf.float32)
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
                cross_entropy = -y_true * tf.math.log(y_pred)
                loss = tf.math.pow(1 - y_pred, gamma) * cross_entropy

                return tf.reduce_sum(loss, axis=-1)
        
            return loss_func
        
        elif self.loss_name == 'level_wt_focal_loss':
            if 'gamma' in self.kwargs:
                gamma = self.kwargs['gamma']
            else:
                gamma = 2.

            if 'level_wt' in self.kwargs:
                level_wt = self.kwargs['level_wt']
            else:
                level_wt = 1.0

            def loss_func(y_true, y_pred):
                y_true = tf.convert_to_tensor(y_true, tf.float32)
                y_pred = tf.convert_to_tensor(y_pred, tf.float32)
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
                cross_entropy = -y_true * tf.math.log(y_pred)
                loss = tf.math.pow(1 - y_pred, gamma) * cross_entropy

                return level_wt*tf.reduce_sum(loss, axis=-1)
        
            return loss_func
             
        
    #def get_config(self):

        #return {'loss_name':self.loss_name, **self.kwargs}
        
    
        
        # else:
        # add other losses here


