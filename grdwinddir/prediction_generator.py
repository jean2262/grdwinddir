"""
inspired from predict_wind_direction.py (project_rmarquart) and generate_L2A_winddir_regression_product.py
January 2024
"""
import os
import numpy as np
import xarray as xr
import scipy as sp
from tqdm import tqdm
from patch_generator import save_tile
from joblib import Parallel, delayed


def predict_wind_direction(tiles, model, input_shape, n_jobs, save=False, save_dir='.'):
    """
    Predicts wind direction based on radar or SAR tiles.

    Parameters:
    - tiles (str or xarray.Dataset): Path to the radar or SAR dataset or a xarray dataset containing tiles data.
    - model: The trained wind direction prediction model.
    - input_shape (tuple): Shape of the input data expected by the model.
    - save (bool): If True, save the resulting dataset.
    - save_dir (str): Directory where the dataset will be saved if save is True.

    Returns:
    - tiles_with_prediction (xarray.Dataset): The original tiles dataset with predicted wind direction and
      standard deviation added.
    """
    if isinstance(tiles, str) and os.path.isfile(tiles):
        tiles = xr.open_dataset(tiles)

    def process_tile(tile):
        proc_tile = []
        for pol in tile.pol.values:
            to_proc_tile = tile.sel(pol=pol)
            x = to_proc_tile.sigma0_detrend.values
            x_normalized = np.array([((x - np.average(x)) / np.std(x)).reshape(input_shape)])  # Normalize data
            heading_angle = np.mean(np.deg2rad(tile["ground_heading"].values))
            predictions = np.ones((1, len(model))) * np.NaN

            if x_normalized.size > 0:
                predictions_usable = launch_prediction(x_normalized, model)
                predictions = predictions_usable

            mean_wind_dir_prediction = sp.stats.circmean(predictions, np.pi, axis=1)
            std_prediction = sp.stats.circstd(predictions, np.pi, axis=1)
            predicted_wind_dir = mean_wind_dir_prediction + heading_angle

            mean_wind_dir = np.rad2deg(predicted_wind_dir) % 180
            std_predicted = np.rad2deg(std_prediction) % 180
            proc_tile.append(to_proc_tile.assign(mean_wind_direction=mean_wind_dir[0], wind_std=std_predicted[0]))

        return xr.concat(proc_tile, dim='pol')

    new_tiles = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_tile)(tiles.sel(tile=i)) for i in tqdm(range(len(tiles.tile)), desc='Prediction processing')
    )

    tiles_with_prediction = xr.concat(new_tiles, dim='tile')

    tiles_with_prediction['mean_wind_direction'].attrs.update({
        'comment': '180° ambiguous wind direction, clockwise, relative to geographic North.',
        'units': 'degree',
        'vmin': 0,
        'vmax': 180
    })
    tiles_with_prediction['wind_std'].attrs.update({
        'comment': 'Standard deviation associated with wind direction.',
        'units': 'degree',
        'vmin': 0,
        'vmax': 180
    })

    if save:
        save_tile(tiles_with_prediction, save_dir)

    return tiles_with_prediction


def launch_prediction(x_normalized, model_regs):
    """
    Predicts wind direction of the given vector.
    Parameters:
        x_normalized (numpy.array): normalized array used for prediction.
        model_regs (list of M64RN4): list of the M64RN4 models used for prediction.
    Returns:
        predictions (numpy.array): array of predictions.
    """
    predictions = np.zeros((x_normalized.shape[0], len(model_regs)))

    for i, m64rn4 in enumerate(model_regs):
        predictions[:, i] = np.squeeze(m64rn4.model.predict(x_normalized))

    return predictions
