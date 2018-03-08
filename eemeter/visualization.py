import numpy as np

from eemeter import merge_temperature_data

__all__ = (
    'plot_candidate',
    'plot_energy_signature',
    'plot_model_fit',
    'plot_time_series',
)


def plot_time_series(meter_data, temperature_data, **kwargs):
    ''' Plot meter and temperature data in dual-axes time series.

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
    '''
    # TODO(philngo): include image in docs.
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError('matplotlib is required for plotting.')

    default_kwargs = {'figsize': (16, 4) }
    default_kwargs.update(kwargs)
    fig, ax1 = plt.subplots(**default_kwargs)

    ax1.plot(meter_data.index, meter_data.value, color='C0', label='Energy Use')
    ax1.set_ylabel('Energy Use')

    ax2 = ax1.twinx()
    ax2.plot(temperature_data.index, temperature_data, color='C1', label='Temperature', alpha=0.8)
    ax2.set_ylabel('Temperature')

    fig.legend()

    return ax1, ax2


def plot_energy_signature(
    meter_data, temperature_data, temp_col=None, ax=None, title=None,
    figsize=None, **kwargs
):
    ''' Plot meter and temperature data in energy signature.

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
    '''
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError('matplotlib is required for plotting.')

    # format data
    df = merge_temperature_data(meter_data, temperature_data)

    if figsize is None:
        figsize = (10, 4)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if temp_col is None:
        temp_col = 'temperature_mean'

    ax.scatter(df[temp_col], df.meter_value, **kwargs)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Energy Use')

    if title is not None:
        ax.set_title(title)

    return ax


def plot_candidate(
    candidate, best=False, ax=None, title=None, figsize=None, temp_range=None,
    alpha=None, **kwargs
):
    ''' Plot a candidate model.

    Parameters
    ----------
    candidate : :any:`eemeter.CandidateModel`
        A candidate model with a predict function.
    best : :any:`bool`, optional
        Whether this is the best candidate or not.
    ax : :any:`matplotlib.axes.Axes`, optional
        Existing axes to plot on.
    title : :any:`str`, optional
        Chart title.
    figsize : :any:`tuple`, optional
        (width, height) of chart.
    temp_range : :any:`tuple`
        (min, max) temperatures to plot model.
    alpha : :any:`float` between 0 and 1
        Transparency, 0 fully transparent, 1 fully opaque.
    **kwargs
        Keyword arguments for :any:`matplotlib.axes.Axes.plot`

    Returns
    -------
    ax : :any:`matplotlib.axes.Axes`
        Matplotlib axes.
    '''
    if temp_range is None:
        temp_min, temp_max = 30, 90
    else:
        temp_min, temp_max = temp_range

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError('matplotlib is required for plotting.')

    if figsize is None:
        figsize = (10, 4)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if candidate.status == 'QUALIFIED':
        color = 'C2'
    elif candidate.status == 'DISQUALIFIED':
        color = 'C3'
    else:
        return

    if alpha is None:
        alpha = 0.3

    if best:
        color = 'C1'
        alpha = 1

    temps = np.arange(temp_min, temp_max)
    prediction = candidate.predict(temps)

    ax.plot(temps, prediction, color=color, alpha=alpha, **kwargs)

    if title is not None:
        ax.set_title(title)

    return ax


def plot_model_fit(
    model_fit, ax=None, title=None, figsize=None, with_candidates=False,
    candidate_alpha=None, temp_range=None
):
    ''' Plot a model fit.

    Parameters
    ----------
    model_fit : :any:`eemeter.ModelFit`
        The model fit to plot.
    ax : :any:`matplotlib.axes.Axes`, optional
        Existing axes to plot on.
    title : :any:`str`, optional
        Chart title.
    figsize : :any:`tuple`, optional
        (width, height) of chart.
    with_candidates : :any:`bool`
        If True, also plot candidate models.
    candidate_alpha : :any:`float` between 0 and 1
        Transparency at which to plot candidate models. 0 fully transparent,
        1 fully opaque.

    Returns
    -------
    ax : :any:`matplotlib.axes.Axes`
        Matplotlib axes.
    '''
    # TODO(philngo): the model fit object should probably own this.
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError('matplotlib is required for plotting.')

    if figsize is None:
        figsize = (10, 4)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if temp_range is None:
        temp_range = (20, 90)

    if with_candidates:
        for candidate in model_fit.candidates:
            plot_candidate(candidate, ax=ax, temp_range=temp_range, alpha=candidate_alpha)
    plot_candidate(model_fit.model, ax=ax, best=True, temp_range=temp_range)

    if title is not None:
        ax.set_title(title)

    return ax
