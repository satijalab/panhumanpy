'''
Custom calibration model architecture. Needed for deserialization of saved
model.
'''


import tensorflow as tf
import keras



@keras.saving.register_keras_serializable(package="panhumanpy._tools.v1.calibration")
class TemperatureScalingTwoParam(tf.keras.layers.Layer):
    """Two-parameter temperature scaling layer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.theta_0 = self.add_weight(
            name='theta_0',
            shape=(),
            initializer='zeros',
            trainable=True,
            dtype=tf.float32
        )
        self.theta_1 = self.add_weight(
            name='theta_1',
            shape=(),
            initializer='ones',
            trainable=True,
            dtype=tf.float32
        )
        super().build(input_shape)
        
    def call(self, inputs):
        logits, uncertainty = inputs
        logits = tf.cast(logits, tf.float32)
        uncertainty = tf.cast(uncertainty, tf.float32)
        
        temperature = self.theta_0 + self.theta_1 * uncertainty
        temperature = tf.clip_by_value(
            tf.nn.softplus(temperature), 
            clip_value_min=0.001, 
            clip_value_max=1000.0
        )
        
        scaled_logits = logits / temperature
        return tf.nn.softmax(scaled_logits)
    
    def get_config(self):
        return super().get_config()



@keras.saving.register_keras_serializable(package="panhumanpy._tools.v1.calibration")
def ece_metric(n_bins=20):
    """
    Create a custom ECE metric function for Keras
    """
    def ece(y_true, y_pred):
        """
        Compute Expected Calibration Error (ECE) as a TensorFlow metric
        """
        # Get confidences (max probability) and predictions
        confidences = tf.reduce_max(y_pred, axis=1)
        predictions = tf.argmax(y_pred, axis=1)
        
        # Convert y_true to int64 if needed
        y_true_int = tf.cast(y_true, tf.int64)
        
        # Check if predictions are correct
        correct = tf.cast(tf.equal(predictions, y_true_int), tf.float32)
        
        ece_sum = 0.0
        
        # Loop through bins
        for i in range(n_bins):
            bin_lower = tf.cast(i, tf.float32) / tf.cast(n_bins, tf.float32)
            bin_upper = tf.cast(i + 1, tf.float32) / tf.cast(n_bins, tf.float32)
            
            # Find samples in this bin
            # For the first bin, include the lower bound
            if i == 0:
                in_bin = tf.logical_and(
                    tf.greater_equal(confidences, bin_lower),  # >= for first bin
                    tf.less_equal(confidences, bin_upper)
                )
            else:
                in_bin = tf.logical_and(
                    tf.greater(confidences, bin_lower),
                    tf.less_equal(confidences, bin_upper)
                )
            
            # Proportion of samples in bin
            prop_in_bin = tf.reduce_mean(tf.cast(in_bin, tf.float32))
            
            # Only compute if there are samples in the bin
            def compute_bin_ece():
                accuracy_in_bin = tf.reduce_mean(tf.boolean_mask(correct, in_bin))
                avg_confidence_in_bin = tf.reduce_mean(tf.boolean_mask(confidences, in_bin))
                return tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            def no_samples_in_bin():
                return 0.0
            
            bin_ece = tf.cond(
                tf.greater(prop_in_bin, 0),
                compute_bin_ece,
                no_samples_in_bin
            )
            
            ece_sum += bin_ece
        
        return ece_sum
    
    # Set the function name for better display in Keras
    ece.__name__ = 'ece_metric'
    return ece

@keras.saving.register_keras_serializable(package="panhumanpy._tools.v1.calibration")
def brier_score_metric():
    """
    Create a custom Brier Score metric function for Keras
    """
    def brier_score(y_true, y_pred):
        """
        Compute Brier Score as a TensorFlow metric
        """
        # Convert y_true to one-hot encoding
        num_classes = tf.shape(y_pred)[1]
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true_int, num_classes, dtype=tf.float32)
        
        # Compute Brier Score: mean of squared differences
        squared_diffs = tf.square(y_pred - y_true_onehot)
        brier = tf.reduce_mean(tf.reduce_sum(squared_diffs, axis=1))
        
        return brier
    
    # Set the function name for better display in Keras
    brier_score.__name__ = 'brier_score_metric'
    return brier_score


custom_calibration_objects = {
    'temperature_scaling_entropy_informed':TemperatureScalingTwoParam,
    'ece_metric': ece_metric,
    'brier_score_metric':brier_score_metric,
    # Add other calibration methods here as needed
}