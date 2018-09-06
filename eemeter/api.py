import numpy as np

__all__ = ("CandidateModel", "DataSufficiency", "EEMeterWarning", "ModelResults")


def _noneify(value):
    if value is None:
        return None
    return None if np.isnan(value) else value


class CandidateModel(object):
    """ Contains information about a candidate model.

    Attributes
    ----------
    model_type : :any:`str`
        The type of model, e..g., :code:`'hdd_only'`.
    formula : :any:`str`
        The R-style formula for the design matrix of this model, e.g., :code:`'meter_value ~ hdd_65'`.
    status : :any:`str`
        A string indicating the status of this model. Possible statuses:

        - ``'NOT ATTEMPTED'``: Candidate model not fitted due to an issue
          encountered in data before attempt.
        - ``'ERROR'``: A fatal error occurred during model fit process.
        - ``'DISQUALIFIED'``: The candidate model fit was disqualified
          from the model selection process because of a decision made after
          candidate model fit completed, e.g., a bad fit, or a parameter out
          of acceptable range.
        - ``'QUALIFIED'``: The candidate model fit is acceptable and can be
          considered during model selection.
    predict_func : :any:`callable`
        A function of the following form:
        ``predict_func(candidate_model, inputs) -> outputs``
    plot_func : :any:`callable`
        A function of the following form:
        ``plot_func(candidate_model, inputs) -> outputs``
    model_params : :any:`dict`, default :any:`None`
        A flat dictionary of model parameters which must be serializable
        using the :any:`json.dumps` method.
    model : :any:`object`
        The raw model (if any) used in fitting. Not serialized.
    result : :any:`object`
        The raw modeling result (if any) returned by the `model`. Not serialized.
    r_squared_adj : :any:`float`
        The adjusted r-squared of the candidate model.
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        A list of any warnings reported during creation of the candidate model.
    """

    def __init__(
        self,
        model_type,
        formula,
        status,
        predict_func=None,
        plot_func=None,
        model_params=None,
        model=None,
        result=None,
        r_squared_adj=None,
        warnings=None,
    ):
        self.model_type = model_type
        self.formula = formula
        self.status = status  # NOT ATTEMPTED | ERROR | QUALIFIED | DISQUALIFIED
        self.model = model
        self.result = result
        self.r_squared_adj = r_squared_adj
        self.predict_func = predict_func
        self.plot_func = plot_func

        if model_params is None:
            model_params = {}
        self.model_params = model_params

        if warnings is None:
            warnings = []
        self.warnings = warnings

    def __repr__(self):
        return "CandidateModel(model_type='{}', formula='{}', status='{}', r_squared_adj={})".format(
            self.model_type,
            self.formula,
            self.status,
            round(self.r_squared_adj, 3) if self.r_squared_adj is not None else None,
        )

    def json(self):
        """ Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        return {
            "model_type": self.model_type,
            "formula": self.formula,
            "status": self.status,
            "model_params": self.model_params,
            "r_squared_adj": _noneify(self.r_squared_adj),
            "warnings": [w.json() for w in self.warnings],
        }

    def predict(self, *args, **kwargs):
        """ Predict for this model. Arguments may vary by model type.
        """
        if self.predict_func is None:
            raise ValueError(
                "This candidate model cannot be used for prediction because"
                " the predict_func attr is not set."
            )
        else:
            return self.predict_func(
                self.model_type, self.model_params, *args, **kwargs
            )

    def plot(self, *args, **kwargs):
        """ Predict for this model. Arguments may vary by model type.
        """
        if self.plot_func is None:
            raise ValueError(
                "This candidate model cannot be used for plotting because"
                " the plot_func attr is not set."
            )
        else:
            return self.plot_func(self, *args, **kwargs)


class DataSufficiency(object):
    """ Contains the result of a data sufficiency check.

    Attributes
    ----------
    status : :any:`str`
        A string indicating the status of this result. Possible statuses:

        - ``'NO DATA'``: No baseline data was available.
        - ``'FAIL'``: Data did not meet criteria.
        - ``'PASS'``: Data met criteria.
    criteria_name : :any:`str`
        The name of the criteria method used to check for baseline data sufficiency.
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        A list of any warnings reported during the check for baseline data sufficiency.
    settings : :any:`dict`
        A dictionary of settings (keyword arguments) used.
    """

    def __init__(self, status, criteria_name, warnings=None, settings=None):
        self.status = status  # NO DATA | FAIL | PASS
        self.criteria_name = criteria_name

        if warnings is None:
            warnings = []
        self.warnings = warnings

        if settings is None:
            settings = {}
        self.settings = settings

    def __repr__(self):
        return (
            "DataSufficiency("
            "status='{status}', criteria_name='{criteria_name}')".format(
                status=self.status, criteria_name=self.criteria_name
            )
        )

    def json(self):
        """ Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        return {
            "status": self.status,
            "criteria_name": self.criteria_name,
            "warnings": [w.json() for w in self.warnings],
            "settings": self.settings,
        }


class EEMeterWarning(object):
    """ An object representing a warning and data associated with it.

    Attributes
    ----------
    qualified_name : :any:`str`
        Qualified name, e.g., `'eemeter.method_abc.missing_data'`.
    description : :any:`str`
        Prose describing the nature of the warning.
    data : :any:`dict`
        Data that reproducibly shows why the warning was issued.
    """

    def __init__(self, qualified_name, description, data):
        self.qualified_name = qualified_name
        self.description = description
        self.data = data

    def __repr__(self):
        return "EEMeterWarning(qualified_name={})".format(self.qualified_name)

    def json(self):
        """ Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        return {
            "qualified_name": self.qualified_name,
            "description": self.description,
            "data": self.data,
        }


class ModelResults(object):
    """ Contains information about the chosen model.

    Attributes
    ----------
    status : :any:`str`
        A string indicating the status of this result. Possible statuses:

        - ``'NO DATA'``: No baseline data was available.
        - ``'NO MODEL'``: No candidate models qualified.
        - ``'SUCCESS'``: A qualified candidate model was chosen.

    method_name : :any:`str`
        The name of the method used to fit the baseline model.
    model : :any:`eemeter.CandidateModel` or :any:`None`
        The selected candidate model, if any.
    r_squared_adj : :any:`float`
        The adjusted r-squared of the selected model.
    candidates : :any:`list` of :any:`eemeter.CandidateModel`
        A list of any model candidates encountered during the model
        selection and fitting process.
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        A list of any warnings reported during the model selection and fitting
        process.
    metadata : :any:`dict`
        An arbitrary dictionary of metadata to be associated with this result.
        This can be used, for example, to tag the results with attributes like
        an ID::

            {
                'id': 'METER_12345678',
            }

    settings : :any:`dict`
        A dictionary of settings used by the method.
    metrics : :any:`ModelMetrics`
        A ModelMetrics object, if one is calculated and associated with this
        model. (This initializes to None.) The ModelMetrics object contains
        model fit information and descriptive statistics about the underlying data.
    """

    def __init__(
        self,
        status,
        method_name,
        model=None,
        r_squared_adj=None,
        candidates=None,
        warnings=None,
        metadata=None,
        settings=None,
    ):
        self.status = status  # NO DATA | NO MODEL | SUCCESS
        self.method_name = method_name
        self.model = model
        self.r_squared_adj = r_squared_adj

        if candidates is None:
            candidates = []
        self.candidates = candidates

        if warnings is None:
            warnings = []
        self.warnings = warnings

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        if settings is None:
            settings = {}
        self.settings = settings

        self.metrics = None

    def __repr__(self):
        return "ModelResults(status='{}', method_name='{}', r_squared_adj={})".format(
            self.status, self.method_name, self.r_squared_adj
        )

    def json(self, with_candidates=False):
        """ Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """

        def _json_or_none(obj):
            return None if obj is None else obj.json()

        data = {
            "status": self.status,
            "method_name": self.method_name,
            "model": _json_or_none(self.model),
            "r_squared_adj": _noneify(self.r_squared_adj),
            "warnings": [w.json() for w in self.warnings],
            "metadata": self.metadata,
            "settings": self.settings,
            "metrics": _json_or_none(self.metrics),
            "candidates": None,
        }
        if with_candidates:
            data["candidates"] = [candidate.json() for candidate in self.candidates]
        return data

    def plot(
        self,
        ax=None,
        title=None,
        figsize=None,
        with_candidates=False,
        candidate_alpha=None,
        temp_range=None,
    ):
        """ Plot a model fit.

        Parameters
        ----------
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
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting.")

        if figsize is None:
            figsize = (10, 4)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if temp_range is None:
            temp_range = (20, 90)

        if with_candidates:
            for candidate in self.candidates:
                candidate.plot(ax=ax, temp_range=temp_range, alpha=candidate_alpha)
        self.model.plot(ax=ax, best=True, temp_range=temp_range)

        if title is not None:
            ax.set_title(title)

        return ax
