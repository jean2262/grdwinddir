"""
inspired from predict_wind_direction.py (project_rmarquart) and generate_L2A_winddir_regression_product.py
January 2024
"""
import glob
import logging
import os
from tqdm import tqdm
from l2awinddirection.M64RN4 import M64RN4_regression
from prediction_generator import predict_wind_direction
from patch_generator import tiling_prod, tiling_by_point


def load_model_m64rn4(model_path, input_shape, data_augmentation, learning_rate):
    """
    Load multiple instances of M64RN4_regression model from saved weights.

    Parameters:
    - model_path (list): List of paths to the saved model weights.
    - input_shape (tuple): Shape of the input data expected by the model.
    - data_augmentation (bool): Whether data augmentation is applied during training.
    - learning_rate (float): Learning rate used during training.

    Returns:
    - model_m64rn4 (list): List of loaded M64RN4_regression models.
    """

    model_m64rn4 = []
    for path in tqdm(model_path, desc="Loading models"):
        try:
            m64rn4_reg = M64RN4_regression(input_shape, data_augmentation)
            m64rn4_reg.create_and_compile(learning_rate)
            m64rn4_reg.model.load_weights(path)
            model_m64rn4.append(m64rn4_reg)
        except Exception as e:
            logging.info(f"Error loading model weights from {path}: {e}")
    return model_m64rn4


def wind_dir_prediction(file_path, model_path, input_shape, tiling_mode, tile_size, posting_loc, resolution, noverlap=0,
                        centering=False, side='left', data_augmentation=True, learning_rate=1e-3, save=False,
                        save_dir='.'):
    """
    Predict wind direction using saved models and radar/SAR tiles.

    Parameters: - file_path (str): Path to the radar or SAR dataset. - model_path (str): Path to the directory
    containing saved model weights. - input_shape (tuple): Shape of the input data expected by the model. -
    tiling_mode (str): Mode for tiling the dataset. Options: 'tiling_all', 'tiling_points'. - tile_size (int or
    dict): Size of the tiles for tiling the dataset. If dict, should contain 'line' and 'sample' keys. - posting_loc
    (tuple, optional): Location for tiling the dataset, required if tiling_mode is 'tiling_points'. Defaults to None.
    - resolution (str, optional): Resolution of the dataset. Defaults to None. - noverlap (int or dict, optional):
    Number of overlapping pixels between adjacent tiles. If dict, should contain 'line' and 'sample' keys. Defaults
    to 0. - centering (bool, optional): Whether to center the tiles. Defaults to False. - side (str, optional): Side
    to use when centering the tiles ('left' or 'right'). Defaults to 'left'. - data_augmentation (bool, optional):
    Whether data augmentation is applied during training. Defaults to True. - learning_rate (float, optional):
    Learning rate used during training. Defaults to 1e-3. - save (bool, optional): Whether to save the predicted
    tiles. Defaults to True. - save_dir (str, optional): Directory where the predicted tiles should be saved.
    Defaults to '.'.

    Returns:
    - tiles_with_prediction (xarray.Dataset): Radar or SAR tiles dataset with predicted wind direction added.
    """
    path_best_models = glob.glob(os.path.join(model_path, "*.hdf5"))
    all_model = load_model_m64rn4(model_path=path_best_models, input_shape=input_shape,
                                  data_augmentation=data_augmentation, learning_rate=learning_rate)
    if tiling_mode == 'tiling_all':
        dataset, tiles = tiling_prod(path=file_path, tile_size=tile_size, resolution=resolution, noverlap=noverlap,
                                     centering=centering, side=side, save=save, save_dir=save_dir)
    elif tiling_mode == 'tiling_points':
        if posting_loc is None:
            raise ValueError("Posting location must be provided when tiling_mode is 'tiling_points'.")
        dataset, tiles = tiling_by_point(path=file_path, posting_loc=posting_loc, tile_size=tile_size,
                                         resolution=resolution, save=save, save_dir=save_dir)
    else:
        raise ValueError("Invalid tiling mode. Please choose either 'tiling_all' or 'tiling_points'.")

    tiles_with_prediction = predict_wind_direction(tiles=tiles, model=all_model, input_shape=input_shape, save=save,
                                                   save_dir=save_dir)

    return tiles, tiles_with_prediction
