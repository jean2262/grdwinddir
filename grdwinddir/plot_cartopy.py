import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.wkt
import numpy as np


def plot_cartopy_data(ds=None, tiles=None, polarization='VV', file_name='map', vmin=None, vmax=None):
    """
    Plot radar/SAR data on a Cartopy map.

    Parameters:
        ds (xarray.Dataset, optional): Radar/SAR data to plot.
        tiles (xarray.Dataset, optional): Tiles data to plot.
        polarization (str, optional): Polarization to plot. Defaults to 'VV'.
        file_name (str, optional): Name of the plot. Defaults to 'map'.

    Raises:
        ValueError: If neither 'ds' nor 'tiles' is provided.
    """
    if ds is None and tiles is None:
        raise ValueError("Either 'ds' or 'tiles' must be provided.")

    fig, ax = plt.subplots(figsize=(12, 11), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)

    legend_added = False
    if ds is not None:
        ds_p = ds.sel(pol=polarization)
        ax.pcolormesh(ds_p.longitude.data, ds_p.latitude.data, ds_p.sigma0.data, transform=ccrs.PlateCarree(),
                      cmap='gray', zorder=1, vmin=vmin, vmax=vmax)

    if tiles is not None:
        for i in range(len(tiles.tile)):
            tile = tiles.sel(tile=i, pol=polarization)
            img = tile.sigma0_detrend
            lon = tile.longitude
            lat = tile.latitude
            ax.pcolormesh(lon.data, lat.data, img.data, transform=ccrs.PlateCarree(), cmap='gray', zorder=1,
                          vmin=vmin, vmax=vmax)

            poly = shapely.wkt.loads(str(tile.tile_footprint.values))
            ax.plot(*poly.exterior.xy, '-b', linewidth=0.5, label='Patches footprint', zorder=1)

            if 'mean_wind_direction' in tile.variables:
                ax.scatter(tile.lon_centroid, tile.lat_centroid, label='Patches centroids', color='red', s=20, zorder=2)
                arrow_scal = 0.06
                angle_rad = np.deg2rad(90 - tile.mean_wind_direction)
                dx = arrow_scal * np.cos(angle_rad)
                dy = arrow_scal * np.sin(angle_rad)
                ax.arrow(tile.lon_centroid, tile.lat_centroid, dx, dy, fc='red', ec='red',
                         label='Predicted Wind direction',
                         zorder=2)
                ax.arrow(tile.lon_centroid, tile.lat_centroid, -dx, -dy, fc='red', ec='red', zorder=2)

            if not legend_added:
                ax.legend(loc='upper right')
                legend_added = True
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_title(file_name)
    plt.show()
