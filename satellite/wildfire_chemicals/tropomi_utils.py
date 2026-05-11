"""
tropomi_utils.py
----------------
Helper functions for downloading, reading, cropping and plotting
Sentinel-5P TROPOMI data from the Copernicus Data Space Ecosystem (CDSE).

Functions are extracted from the ddeq package
(https://github.com/tglauch/ddeq) and adapted to work without installing it.

Usage in a notebook:
    from tropomi_utils import (
        make_sources, get_bounding_box, list_files, Download,
        open_netCDF, reduce_dims_and_vars, crop_and_save,
        plot_orbit, plot_extent,
    )
"""

import math
import os
import sys
import time
import zipfile

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Variable lists taken from ddeq.download_S5P
# ---------------------------------------------------------------------------
product_variables = {
    "L2__CH4___": {
        "keep_variables": [
            "methane_mixing_ratio",
            "methane_mixing_ratio_precision",
            "methane_mixing_ratio_bias_corrected",
        ]
    },
    "L2__CLOUD_": {
        "keep_variables": [
            "cloud_fraction",
            "cloud_fraction_precision",
            "cloud_top_pressure",
            "cloud_top_pressure_precision",
            "cloud_base_pressure",
            "cloud_base_pressure_precision",
            "cloud_top_height",
            "cloud_top_height_precision",
            "cloud_base_height",
            "cloud_base_height_precision",
            "cloud_optical_thickness",
            "cloud_optical_thickness_precision",
        ]
    },
    "L2__CO____": {
        "keep_variables": [
            "carbonmonoxide_total_column",
            "carbonmonoxide_total_column_precision",
            "carbonmonoxide_total_column_corrected",
        ]
    },
    "L2__HCHO__": {
        "keep_variables": [
            "formaldehyde_tropospheric_vertical_column",
            "formaldehyde_tropospheric_vertical_column_precision",
        ]
    },
    "L2__NO2___": {
        "keep_variables": [
            "nitrogendioxide_tropospheric_column",
            "nitrogendioxide_tropospheric_column_precision",
            "averaging_kernel",
            "air_mass_factor_total",
            "air_mass_factor_troposphere",
            "tm5_constant_a",
            "tm5_constant_b",
            "cloud_fraction_crb",
            "solar_zenith_angle",
            "surface_altitude",
        ]
    },
    "L2__O3____": {
        "keep_variables": [
            "ozone_total_vertical_column",
            "ozone_total_vertical_column_precision",
        ]
    },
    "L2__SO2___": {
        "keep_variables": [
            "sulfurdioxide_total_vertical_column",
            "sulfurdioxide_total_vertical_column_precision",
            "averaging_kernel",
            "air_mass_factor",
            "tm5_constant_a",
            "tm5_constant_b",
            "cloud_fraction_crb",
            "surface_altitude",
        ]
    },
}

global_vars = [
    "time_utc",
    "delta_time",
    "qa_value",
    "surface_pressure",
    "longitude_bounds",
    "latitude_bounds",
]


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------

def make_sources(name, lon, lat, diameter=50000):
    """
    Create a minimal xarray Dataset that mimics the ddeq 'sources' format.

    Parameters
    ----------
    name : str or list of str
        Source label(s).
    lon : float or list of float
        Longitude(s) of the source(s).
    lat : float or list of float
        Latitude(s) of the source(s).
    diameter : float or list of float, optional
        Source diameter(s) in metres (default 50 km).

    Returns
    -------
    xr.Dataset
    """
    names = [name] if isinstance(name, str) else name
    lons = [lon] if np.isscalar(lon) else list(lon)
    lats = [lat] if np.isscalar(lat) else list(lat)
    diams = [diameter] if np.isscalar(diameter) else list(diameter)

    ds = xr.Dataset(coords={"source": names})
    ds["lon"] = xr.DataArray(lons, dims="source",
                             attrs={"name": "longitude of point source"})
    ds["lat"] = xr.DataArray(lats, dims="source",
                             attrs={"name": "latitude of point source"})
    ds["diameter"] = xr.DataArray(diams, dims="source",
                                  attrs={"name": "source diameter", "units": "m"})
    ds["label"] = xr.DataArray(names, dims="source")
    return ds


def _get_location(sources, name=None):
    """Return (lon, lat, diameter) from a sources Dataset, optionally by name."""
    if name is not None:
        sources = sources.sel(source=name)
    return sources["lon"], sources["lat"], sources["diameter"]


# ---------------------------------------------------------------------------
# Bounding-box / polygon helpers
# ---------------------------------------------------------------------------

def get_bounding_box(lon0, lat0, distance):
    """
    Calculate a bounding box around (lon0, lat0) with the given radius.

    Parameters
    ----------
    lon0, lat0 : float   – source coordinates (degrees)
    distance   : float   – radius in metres

    Returns
    -------
    (min_lon, max_lon, min_lat, max_lat) : float
    """
    R = 6378137  # Earth radius in metres
    lat = lat0 * math.pi / 180
    lon = lon0 * math.pi / 180
    d = distance / R

    min_lat = (lat - d) * 180 / math.pi
    max_lat = (lat + d) * 180 / math.pi
    min_lon = (lon - d / math.cos(lat)) * 180 / math.pi
    max_lon = (lon + d / math.cos(lat)) * 180 / math.pi

    return min_lon, max_lon, min_lat, max_lat


def area_of_interest(min_lon, max_lon, min_lat, max_lat):
    """Format a WKT POLYGON string for the CDSE API."""
    return (
        "POLYGON(("
        f"{min_lon} {min_lat},"
        f"{min_lon} {max_lat},"
        f"{max_lon} {max_lat},"
        f"{max_lon} {min_lat},"
        f"{min_lon} {min_lat}))"
    )


# ---------------------------------------------------------------------------
# CDSE catalogue search
# ---------------------------------------------------------------------------

def list_files(
    west_lon,
    east_lon,
    south_lat,
    north_lat,
    start_date,
    end_date,
    product_abbreviation,
    latency,
    level,
    orbit="-",
    only_latest=False,
):
    """
    Query the CDSE OData catalogue for TROPOMI files.

    Parameters
    ----------
    west_lon, east_lon, south_lat, north_lat : float or str
        Bounding box of the area of interest.
    start_date, end_date : str  – 'YYYY-MM-DD'
    product_abbreviation  : str – e.g. 'L2__NO2___'
    latency               : str – 'NRTI', 'OFFL' or 'RPRO'
    level                 : str – 'L2' or 'L1b'
    orbit                 : str – orbit number or '-' for all
    only_latest           : bool – keep only the latest AUX file per day

    Returns
    -------
    ids, filenames : list of str
    """
    ids = []
    filenames = []

    orbit = None if orbit == "-" else orbit

    if (west_lon is None and east_lon is None
            and south_lat is None and north_lat is None):
        footprint = None
    else:
        footprint = area_of_interest(west_lon, east_lon, south_lat, north_lat)

    for date in pd.date_range(start_date, end_date):
        t0 = date.strftime("%Y-%m-%dT00:00:00Z")
        t1 = date.strftime("%Y-%m-%dT23:59:59Z")

        url_init = (
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
            "?$filter=Collection/Name eq 'SENTINEL-5P'"
        )
        filters = [url_init]

        if product_abbreviation:
            filters.append(
                f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
                f"'productType' and att/OData.CSC.StringAttribute/Value eq "
                f"'{product_abbreviation}')"
            )
        if level:
            filters.append(
                f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
                f"'processingLevel' and att/OData.CSC.StringAttribute/Value eq "
                f"'{level}')"
            )
        if latency:
            filters.append(
                f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
                f"'processingMode' and att/OData.CSC.StringAttribute/Value eq "
                f"'{latency}')"
            )
        if orbit:
            filters.append(
                f"Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq "
                f"'orbitNumber' and att/OData.CSC.IntegerAttribute/Value eq "
                f"'{orbit}')"
            )
        if footprint:
            filters.append(
                f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
            )
        filters.append(f"ContentDate/Start ge {t0}")
        filters.append(f"ContentDate/Start le {t1}")

        query = " and ".join(filters)

        try:
            products = requests.get(query).json()
        except Exception as exc:
            raise ConnectionError(f"Error connecting to the CDSE server: {exc}")

        if "value" in products:
            ids_day = [v["Id"] for v in products["value"]]
            fns_day = [v["Name"] for v in products["value"]]

            if only_latest and product_abbreviation == "AUX_CTMANA":
                idx = np.argsort(fns_day)[-1]
                ids_day = [ids_day[idx]]
                fns_day = [fns_day[idx]]

            ids.extend(ids_day)
            filenames.extend(fns_day)
        else:
            print(f'No results for {date.strftime("%Y-%m-%d")}.')

    return ids, filenames


# ---------------------------------------------------------------------------
# CDSE download
# ---------------------------------------------------------------------------

class Download:
    """Download TROPOMI files from CDSE using OAuth2 (Keycloak) tokens."""

    def __init__(self, path, username, password):
        self.save_path = path
        self.username = username
        self.password = password
        self.keycloak_token = None
        os.makedirs(path, exist_ok=True)

    def get_keycloak(self):
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        try:
            r = requests.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                "/protocol/openid-connect/token",
                data=data,
            )
            r.raise_for_status()
        except Exception:
            raise Exception(
                f"Keycloak token creation failed. Server response: {r.json()}"
            )
        return r.json()["access_token"]

    def download_files(self, ids, filenames):
        if self.keycloak_token is None:
            self.keycloak_token = self.get_keycloak()

        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {self.keycloak_token}"})

        for file_id, file_name in zip(ids, filenames):
            nc_candidate = os.path.join(self.save_path, file_name)
            if os.path.exists(nc_candidate):
                print(f"Already downloaded: {file_name}")
                continue

            try:
                print(f"Downloading {file_name} …")
                url = (
                    f"https://catalogue.dataspace.copernicus.eu/odata/v1/"
                    f"Products({file_id})/$value"
                )
                response = session.get(url, allow_redirects=False, stream=True)

                while response.status_code in (301, 302, 303, 307, 401):
                    url = response.headers["Location"]
                    response = session.get(url, allow_redirects=False, stream=True)
                    if response.status_code == 401:
                        self.keycloak_token = self.get_keycloak()
                        session.headers.update(
                            {"Authorization": f"Bearer {self.keycloak_token}"}
                        )

                if response.status_code not in range(200, 299):
                    raise Exception(
                        f"Server error {response.status_code} for {file_name}"
                    )

                folder_name = (
                    response.headers.get("Content-Disposition", "")
                    .split("filename=")[-1]
                    .strip('"')
                )
                folderpath = os.path.join(self.save_path, folder_name)

                file_size = int(response.headers.get("Content-Length", 0))
                progress = tqdm(total=file_size, unit="iB", unit_scale=True,
                                desc=f"  {folder_name}", miniters=1)
                with open(folderpath, "wb") as fh:
                    for chunk in response.iter_content(1024):
                        fh.write(chunk)
                        progress.update(len(chunk))
                progress.close()

                # Unzip and flatten to save_path
                with zipfile.ZipFile(folderpath, "r") as zf:
                    for member in zf.namelist():
                        if member.endswith(".nc"):
                            zf.extract(member, self.save_path)
                            src = os.path.join(self.save_path, member)
                            dst = os.path.join(self.save_path, os.path.basename(member))
                            os.rename(src, dst)

                os.remove(folderpath)
                try:
                    os.rmdir(folderpath[:-4])
                except OSError:
                    pass

            except Exception as exc:
                print(f"  Error downloading {file_name}: {exc}")


# ---------------------------------------------------------------------------
# Reading TROPOMI NetCDF files
# ---------------------------------------------------------------------------

def open_netCDF(filename, path="./"):
    """
    Open a TROPOMI L2 NetCDF file, merging the PRODUCT and SUPPORT_DATA groups.

    Returns an xarray Dataset with 'lon'/'lat' coordinates.
    """
    if path is None:
        full_filename = filename
    else:
        full_filename = os.path.join(path, filename)

    groups = [
        "PRODUCT",
        "PRODUCT/SUPPORT_DATA/GEOLOCATIONS",
        "PRODUCT/SUPPORT_DATA/INPUT_DATA",
    ]

    try:
        ds = xr.merge([xr.open_dataset(full_filename, group=g) for g in groups])
        ds = ds.rename({"longitude": "lon", "latitude": "lat"})
        ds.attrs["original file name"] = os.path.basename(filename)
        ds.attrs["data source"] = "https://catalogue.dataspace.copernicus.eu/"
        return ds
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found in '{path}'")


def reduce_dims_and_vars(data_S5p):
    """
    Drop unnecessary dimensions and variables from a TROPOMI dataset,
    squeeze the time dimension, and format time_utc.
    """
    product = data_S5p.attrs["original file name"][9:19]

    if product in ["L2__O3____", "L2__SO2___", "L2__HCHO__", "L2__CLOUD_"]:
        data_S5p = data_S5p.set_coords(("lat", "lon"))

    keep_dimensions = [
        "scanline", "ground_pixel", "time", "lat", "lon",
        "corner", "orbit", "layer", "vertices",
    ]
    keep_variables = product_variables[product]["keep_variables"] + global_vars

    for dim in list(data_S5p.indexes):
        if dim not in keep_dimensions:
            data_S5p = data_S5p.drop_dims(dim)

    for var in list(data_S5p.data_vars):
        if var not in keep_variables:
            data_S5p = data_S5p.drop_vars(var)

    if "time" in data_S5p.dims:
        data_S5p = data_S5p.squeeze("time")

    # normalise time_utc to datetime64
    if data_S5p.time_utc.dtype == object:
        data_S5p["time_utc"] = (
            data_S5p.time_utc.str.strip("Z").astype("datetime64[ns]")
        )

    # extract orbit number from filename
    orbit_str = data_S5p.attrs["original file name"][52:57]
    if orbit_str.isdigit():
        data_S5p = data_S5p.assign_coords(orbit=int(orbit_str))

    return data_S5p


# ---------------------------------------------------------------------------
# Cropping and saving
# ---------------------------------------------------------------------------

def _crop_data(data_S5p, source_name, lon0, lat0, distance):
    """
    Crop a TROPOMI dataset to a bounding box around (lon0, lat0).

    Returns (cropped_dataset, was_data_found).
    """
    data_S5p = data_S5p.set_coords(("latitude_bounds", "longitude_bounds"))

    min_lon, max_lon, min_lat, max_lat = get_bounding_box(lon0, lat0, distance)

    mask_lon = (data_S5p.lon >= min_lon) & (data_S5p.lon <= max_lon)
    mask_lat = (data_S5p.lat >= min_lat) & (data_S5p.lat <= max_lat)

    has_data = (
        np.count_nonzero(
            ~np.isnan(np.where(mask_lon & mask_lat, data_S5p.qa_value, np.nan))
        ) > 0
    )

    cropped = data_S5p.where(mask_lon & mask_lat, drop=has_data)
    cropped.attrs = dict(data_S5p.attrs)
    cropped.attrs.update(
        {
            "description": "Sentinel-5P data cropped to a source",
            "source": source_name,
            "distance around source [m]": distance,
        }
    )
    return cropped, has_data


def _save_file(data_S5p, path_open="./", path_save="./", delete=False, overwrite=None):
    """Save a cropped TROPOMI dataset to NetCDF."""
    filename_output = (
        f'{data_S5p.attrs["source"]}_{data_S5p.attrs["original file name"]}'
    )
    os.makedirs(path_save, exist_ok=True)
    out_path = os.path.join(path_save, filename_output)

    if os.path.exists(out_path):
        if overwrite is None:
            overwrite = input(
                f"[WARNING] {filename_output} already exists – overwrite? [y/n]: "
            )
        if overwrite.lower() in ("no", "n"):
            print("File not saved.")
            return
        os.remove(out_path)
        time.sleep(2)

    data_S5p.to_netcdf(path=out_path)
    print(f"Saved: {out_path}")

    if delete:
        os.remove(os.path.join(path_open, data_S5p.attrs["original file name"]))


def crop_and_save(
    all_filenames,
    sources,
    distance,
    delete=False,
    overwrite=None,
    path_open="./",
    path_save="./",
):
    """
    For each file in *all_filenames*, open it, reduce variables,
    crop to each source in *sources*, and save.

    Parameters
    ----------
    all_filenames : list of str
    sources       : xr.Dataset  (from make_sources)
    distance      : float       – crop radius in metres
    delete        : bool        – delete raw file after cropping
    overwrite     : None / 'y'  – passed to _save_file
    path_open     : str         – directory of raw files
    path_save     : str         – directory for cropped output
    """
    for filename in all_filenames:
        ds = open_netCDF(filename, path=path_open)
        ds = reduce_dims_and_vars(ds)

        for src in sources.source.values:
            lon0 = float(sources.sel(source=src)["lon"].values)
            lat0 = float(sources.sel(source=src)["lat"].values)
            cropped, save = _crop_data(ds, src, lon0, lat0, distance)
            if save:
                _save_file(cropped, path_open=path_open,
                           path_save=path_save, delete=delete,
                           overwrite=overwrite)
        sys.stdout.flush()
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_orbit(filename, variable="nitrogendioxide_tropospheric_column", path="./"):
    """Quick overview plot of an entire TROPOMI orbit."""
    ds = open_netCDF(filename, path)
    ds = reduce_dims_and_vars(ds)

    vmin = float(np.nanquantile(ds[variable], 0.01))
    vmax = float(np.nanquantile(ds[variable], 0.99))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines()
    ds[variable].plot.pcolormesh(
        ax=ax, x="lon", y="lat", add_colorbar=True,
        cmap="viridis", vmin=vmin, vmax=vmax,
    )
    date_str = pd.to_datetime(ds.time_utc.mean().values).strftime("%Y-%m-%d")
    plt.title(f"date: {date_str}  |  orbit: {ds.attrs.get('orbit', '?')}")
    plt.tight_layout()
    return ax.figure


def plot_extent(
    data_S5p,
    var,
    sources,
    vmin=None,
    vmax=None,
    zoom=True,
    qa=True,
    ha="left",
    va="center",
):
    """
    Map plot of *var* from a cropped TROPOMI dataset.

    Parameters
    ----------
    data_S5p : xr.Dataset   – cropped TROPOMI dataset (output of crop_and_save)
    var      : str          – variable name to plot (e.g. 'NO2')
    sources  : xr.Dataset   – from make_sources()
    vmin, vmax : float      – colorbar limits (auto if None)
    zoom     : bool         – zoom to the bounding box
    qa       : bool         – mask pixels with qa_value ≤ 0.75
    ha, va   : str          – text alignment for source label
    """
    source_name = data_S5p.attrs.get("source", "")
    lon0 = float(sources.sel(source=source_name)["lon"].values)
    lat0 = float(sources.sel(source=source_name)["lat"].values)

    try:
        distance = data_S5p.attrs["distance around source [m]"]
    except KeyError:
        distance = data_S5p.attrs.get("distance around source [km]", 300) * 1e3

    min_lon, max_lon, min_lat, max_lat = get_bounding_box(lon0, lat0, distance)

    lon = data_S5p.lon
    lat = data_S5p.lat
    value = data_S5p[var].values.copy().astype(float)
    label = data_S5p[var].attrs.get("long_name", var)
    units = data_S5p[var].attrs.get("units", "")

    if qa:
        if "qa_value" in data_S5p.data_vars:
            value = np.where(data_S5p["qa_value"] > 0.75, value, np.nan)
        elif "clouds" in data_S5p.data_vars:
            value = np.where(data_S5p["clouds"] < 0.25, value, np.nan)

    if vmin is None:
        vmin = float(np.nanquantile(value, 0.01))
    if vmax is None:
        vmax = float(np.nanquantile(value, 0.99))

    x_offset = 0.15 if ha == "left" else -0.15

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines(resolution="10m", color="k", linewidth=1.0)
    ax.axis("equal")
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_0_boundary_lines_land", "10m"
        ),
        edgecolor="k", facecolor="none", linewidth=1.0,
    )
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    m = ax.pcolormesh(
        lon, lat, value,
        vmin=vmin, vmax=vmax,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    ax.scatter(lon0, lat0, marker="o", s=20, c="black",
               edgecolor="white", transform=ccrs.PlateCarree())
    ax.text(
        lon0 + x_offset, lat0, source_name,
        clip_on=True, horizontalalignment=ha, verticalalignment=va,
        path_effects=[PathEffects.withStroke(linewidth=2.5, foreground="w")],
    )
    fig.colorbar(m, ax=ax).set_label(f"{label} [{units}]\n", wrap=True)

    date_str = pd.to_datetime(data_S5p.time_utc.mean().values).strftime("%Y-%m-%d")
    orbit_val = data_S5p.attrs.get("orbit", getattr(data_S5p, "orbit", "?"))
    ax.set_title(date_str)

    if zoom:
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

    return fig
