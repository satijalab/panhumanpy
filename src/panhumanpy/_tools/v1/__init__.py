"""
keras model  metadata for internal use by the package for model version
v1.
"""

model_meta = {
    'inference_model_name': 'M0.2',
    'inference_model_loss_function': 'level_wt_focal_loss',
    'max_depth': 8,
    'inference_model_embedding_layer': 'dense_3',
    'feature_panel_size': 5055,
    'calibration': 'temperature_scaling_entropy_informed',
    'calibrators':[f"temp_scaler_L_{i+1}.keras" for i in range(8)]
}