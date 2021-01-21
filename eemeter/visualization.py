#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2019 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
import numpy as np
import pandas as pd

from .features import (
    merge_features,
    compute_usage_per_day_feature,
    compute_temperature_features,
)


__all__ = ("plot_energy_signature", "plot_time_series")


def plot_time_series(meter_data, temperature_data, **kwargs):
    """Plot meter and temperature data in dual-axes time series.

    Parameters
    ----------
    meter_data : :any:`pandas.DataFrame`
        A :any:`pandas.DatetimeIndex`-indexed DataFrame of meter data with the column ``value``.
    temperature_data : :any:`pandas.Series`
        A :any:`pandas.DatetimeIndex`-indexed Series of temperature data.
    **kwargs
        Arbitrary keyword arguments to pass to
        :any:`plt.subplots <matplotlib.pyplot.subplots>`

    Returns
    -------
    axes : :any:`tuple` of :any:`matplotlib.axes.Axes`
        Tuple of ``(ax_meter_data, ax_temperature_data)``.
    """
    # TODO(philngo): include image in docs.
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.")

    default_kwargs = {"figsize": (16, 4)}
    default_kwargs.update(kwargs)
    fig, ax1 = plt.subplots(**default_kwargs)

    ax1.plot(
        meter_data.index,
        meter_data.value,
        color="C0",
        label="Energy Use",
        drawstyle="steps-post",
    )
    ax1.set_ylabel("Energy Use")

    ax2 = ax1.twinx()
    ax2.plot(
        temperature_data.index,
        temperature_data,
        color="C1",
        label="Temperature",
        alpha=0.8,
    )
    ax2.set_ylabel("Temperature")

    fig.legend()

    return ax1, ax2


def plot_energy_signature(
    meter_data,
    temperature_data,
    temp_col=None,
    ax=None,
    title=None,
    figsize=None,
    **kwargs
):
    """Plot meter and temperature data in energy signature.

    Parameters
    ----------
    meter_data : :any:`pandas.DataFrame`
        A :any:`pandas.DatetimeIndex`-indexed DataFrame of meter data with the column ``value``.
    temperature_data : :any:`pandas.Series`
        A :any:`pandas.DatetimeIndex`-indexed Series of temperature data.
    temp_col : :any:`str`, default ``'temperature_mean'``
        The name of the temperature column.
    ax : :any:`matplotlib.axes.Axes`
        The axis on which to plot.
    title : :any:`str`, optional
        Chart title.
    figsize : :any:`tuple`, optional
        (width, height) of chart.
    **kwargs
        Arbitrary keyword arguments to pass to
        :any:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    ax : :any:`matplotlib.axes.Axes`
        Matplotlib axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.")

    # format data
    temperature_mean = compute_temperature_features(meter_data.index, temperature_data)
    usage_per_day = compute_usage_per_day_feature(meter_data, series_name="meter_value")
    df = merge_features([usage_per_day, temperature_mean.temperature_mean])

    if figsize is None:
        figsize = (10, 4)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if temp_col is None:
        temp_col = "temperature_mean"

    ax.scatter(df[temp_col], df.meter_value, **kwargs)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Energy Use per Day")

    if title is not None:
        ax.set_title(title)

    return ax
