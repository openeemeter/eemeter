import numpy as np
from pydantic import BaseModel, field_serializer
from enum import Enum

from scipy.stats import t as t_dist
from scipy.stats import norm as norm_dist

from copy import deepcopy as copy

import numba
from numba.extending import overload


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


min_pos_system_value = (np.finfo(float).tiny * (1e20)) ** (1 / 2)
max_pos_system_value = (np.finfo(float).max * (1e-20)) ** (1 / 2)
ln_min_pos_system_value = np.log(min_pos_system_value)
ln_max_pos_system_value = np.log(max_pos_system_value)


class ModelType(str, Enum):
    # Full model \_/
    HDD_TIDD_CDD_SMOOTH = "hdd_tidd_cdd_smooth"
    HDD_TIDD_CDD = "hdd_tidd_cdd"

    # Heating, temp independent \__
    HDD_TIDD_SMOOTH = "hdd_tidd_smooth"
    HDD_TIDD = "hdd_tidd"

    # Temp independent, cooling __/
    TIDD_CDD_SMOOTH = "tidd_cdd_smooth"
    TIDD_CDD = "tidd_cdd"

    # Temp independent, ___
    TIDD = "tidd"


class ModelCoefficients(BaseModel):
    """
    A class used to represent the coefficients of a model.

    Attributes
    ----------
    model_type : ModelType
        The type of the model.
    intercept : float
        The intercept of the model.
    hdd_bp : float | None
        The heating degree days breakpoint of the model, if applicable.
    hdd_beta : float | None
        The heating degree days beta of the model, if applicable.
    hdd_k : float | None
        The heating degree days k of the model, if applicable.
    cdd_bp : float | None
        The cooling degree days breakpoint of the model, if applicable.
    cdd_beta : float | None
        The cooling degree days beta of the model, if applicable.
    cdd_k : float | None
        The cooling degree days k of the model, if applicable.

    Methods
    -------
    from_np_arrays(coefficients, coefficient_ids)
        Constructs a ModelCoefficients object from numpy arrays of coefficients and their corresponding ids.
    to_np_array()
        Converts the ModelCoefficients object to a numpy array.
    """
    """
    A class used to represent the coefficients of a model.

    Attributes
    ----------
    model_type : ModelType
        The type of the model.
    intercept : float
        The intercept of the model.
    hdd_bp : float | None
        The heating degree days breakpoint of the model, if applicable.
    hdd_beta : float | None
        The heating degree days beta of the model, if applicable.
    hdd_k : float | None
        The heating degree days k of the model, if applicable.
    cdd_bp : float | None
        The cooling degree days breakpoint of the model, if applicable.
    cdd_beta : float | None
        The cooling degree days beta of the model, if applicable.
    cdd_k : float | None
        The cooling degree days k of the model, if applicable.

    Methods
    -------
    from_np_arrays(coefficients, coefficient_ids)
        Constructs a ModelCoefficients object from numpy arrays of coefficients and their corresponding ids.
    to_np_array()
        Converts the ModelCoefficients object to a numpy array.
    """
    
    model_type: ModelType
    intercept: float
    hdd_bp: float | None = None
    hdd_beta: float | None = None
    hdd_k: float | None = None
    cdd_bp: float | None = None
    cdd_beta: float | None = None
    cdd_k: float | None = None

    @property
    def is_smooth(self):
        return self.model_type in [
            ModelType.HDD_TIDD_CDD_SMOOTH,
            ModelType.HDD_TIDD_SMOOTH,
            ModelType.TIDD_CDD_SMOOTH,
        ]

    @property
    def model_key(self):
        """Used inside OptimizedResult when reducing model"""
        match self.model_type:
            case ModelType.HDD_TIDD_CDD_SMOOTH:
                return "hdd_tidd_cdd_smooth"
            case ModelType.HDD_TIDD_CDD:
                return "hdd_tidd_cdd"
            case ModelType.HDD_TIDD_SMOOTH | ModelType.TIDD_CDD_SMOOTH:
                return "c_hdd_tidd_smooth"
            case ModelType.HDD_TIDD | ModelType.TIDD_CDD:
                return "c_hdd_tidd"
            case ModelType.TIDD:
                return "tidd"

    @classmethod
    def from_np_arrays(cls, coefficients, coefficient_ids):
        """
        This class method creates a ModelCoefficients instance from numpy arrays of coefficients and their corresponding ids.

        Args:
            cls (class): The class to which this class method belongs.
            coefficients (np.array): A numpy array of coefficients.
            coefficient_ids (list): A list of coefficient ids.

        Returns:
            ModelCoefficients: An instance of ModelCoefficients class.

        Raises:
            ValueError: If the coefficient_ids do not match any of the expected patterns.

        The method matches the coefficient_ids to predefined patterns and based on the match, 
        it initializes a ModelCoefficients instance with the corresponding model_type and coefficients. 
        If the coefficient_ids do not match any of the predefined patterns, it raises a ValueError.
        """

        match coefficient_ids:
            case [
                "hdd_bp",
                "hdd_beta",
                "hdd_k",
                "cdd_bp",
                "cdd_beta",
                "cdd_k",
                "intercept",
            ]:
                hdd_bp=coefficients[0]
                hdd_beta=coefficients[1]
                hdd_k=coefficients[2]
                cdd_bp=coefficients[3]
                cdd_beta=coefficients[4]
                cdd_k=coefficients[5]
                if cdd_bp < hdd_bp:
                    hdd_bp, cdd_bp = cdd_bp, hdd_bp
                    hdd_beta, cdd_beta = cdd_beta, hdd_beta
                    hdd_k, cdd_k = cdd_k, hdd_k
                return ModelCoefficients(
                    model_type=ModelType.HDD_TIDD_CDD_SMOOTH,
                    hdd_bp=hdd_bp,
                    hdd_beta=hdd_beta,
                    hdd_k=hdd_k,
                    cdd_bp=cdd_bp,
                    cdd_beta=cdd_bp,
                    cdd_k=cdd_bp,
                    intercept=coefficients[6],
                )
            case [
                "hdd_bp",
                "hdd_beta",
                "cdd_bp",
                "cdd_beta",
                "intercept",
            ]:
                hdd_bp=coefficients[0]
                hdd_beta=coefficients[1]
                cdd_bp=coefficients[2]
                cdd_beta=coefficients[3]
                if cdd_bp < hdd_bp:
                    hdd_bp, cdd_bp = cdd_bp, hdd_bp
                    hdd_beta, cdd_beta = cdd_beta, hdd_beta
                return ModelCoefficients(
                    model_type=ModelType.HDD_TIDD_CDD,
                    hdd_bp=hdd_bp,
                    hdd_beta=hdd_beta,
                    cdd_bp=cdd_bp,
                    cdd_beta=cdd_beta,
                    intercept=coefficients[4],
                )
            case [
                "c_hdd_bp",
                "c_hdd_beta",
                "c_hdd_k",
                "intercept",
            ]:
                if coefficients[1] < 0:  # model is heating dependent
                    hdd_bp = coefficients[0]
                    hdd_beta = coefficients[1]
                    hdd_k = coefficients[2]
                    cdd_bp = cdd_beta = cdd_k = None
                    model_type = ModelType.HDD_TIDD_SMOOTH
                else:  # model is cooling dependent
                    cdd_bp = coefficients[0]
                    cdd_beta = coefficients[1]
                    cdd_k = coefficients[2]
                    hdd_bp = hdd_beta = hdd_k = None
                    model_type = ModelType.TIDD_CDD_SMOOTH
                return ModelCoefficients(
                    model_type=model_type,
                    hdd_bp=hdd_bp,
                    hdd_beta=hdd_beta,
                    hdd_k=hdd_k,
                    cdd_bp=cdd_bp,
                    cdd_beta=cdd_beta,
                    cdd_k=cdd_k,
                    intercept=coefficients[3],
                )
            case [
                "c_hdd_bp",
                "c_hdd_beta",
                "intercept",
            ]:
                if coefficients[1] < 0:  # model is heating dependent
                    hdd_bp = coefficients[0]
                    hdd_beta = coefficients[1]
                    cdd_bp = cdd_beta = None
                    model_type = ModelType.HDD_TIDD
                else:  # model is cooling dependent
                    cdd_bp = coefficients[0]
                    cdd_beta = coefficients[1]
                    hdd_bp = hdd_beta = None
                    model_type = ModelType.TIDD_CDD
                return ModelCoefficients(
                    model_type=model_type,
                    hdd_bp=hdd_bp,
                    hdd_beta=hdd_beta,
                    cdd_bp=cdd_bp,
                    cdd_beta=cdd_beta,
                    intercept=coefficients[2],
                )
            case [
                "intercept",
            ]:
                return ModelCoefficients(
                    model_type=ModelType.TIDD,
                    intercept=coefficients[0],
                )
            case _:
                raise ValueError

    def to_np_array(self):
        """
        This method converts the model parameters into a numpy array based on the model type.

        The model type determines which parameters are included in the array. The parameters are:
        - hdd_bp: The base point for heating degree days (HDD)
        - hdd_beta: The beta coefficient for HDD
        - hdd_k: The smoothing parameter for HDD
        - cdd_bp: The base point for cooling degree days (CDD)
        - cdd_beta: The beta coefficient for CDD
        - cdd_k: The smoothing parameter for CDD
        - intercept: The model's intercept

        Returns:
            np.array: A numpy array containing the relevant parameters for the model type.
        """

        match self.model_type:
            case ModelType.HDD_TIDD_CDD_SMOOTH:
                return np.array(
                    [
                        self.hdd_bp,
                        self.hdd_beta,
                        self.hdd_k,
                        self.cdd_bp,
                        self.cdd_beta,
                        self.cdd_k,
                        self.intercept,
                    ]
                )
            case ModelType.HDD_TIDD_CDD:
                return np.array(
                    [
                        self.hdd_bp,
                        self.hdd_beta,
                        self.cdd_bp,
                        self.cdd_beta,
                        self.intercept,
                    ]
                )
            case ModelType.HDD_TIDD_SMOOTH:
                return np.array(
                    [self.hdd_bp, self.hdd_beta, self.hdd_k, self.intercept]
                )
            case ModelType.TIDD_CDD_SMOOTH:
                return np.array(
                    [self.cdd_bp, self.cdd_beta, self.cdd_k, self.intercept]
                )
            case ModelType.HDD_TIDD:
                return np.array([self.hdd_bp, self.hdd_beta, self.intercept])
            case ModelType.TIDD_CDD:
                return np.array([self.cdd_bp, self.cdd_beta, self.intercept])
            case ModelType.TIDD:
                return np.array([self.intercept])

    # @field_serializer('model_type')
    # def serialize_model_type(self, model_type, _info):
    #     # StrEnum would help avoid this
    #     match model_type:
    #         case ModelType.HDD_TIDD_CDD_SMOOTH:
    #             return "hdd_tidd_cdd_smooth"
    #         case ModelType.HDD_TIDD_CDD:
    #             return "hdd_tidd_cdd"
    #         case ModelType.HDD_TIDD_SMOOTH:
    #             return "hdd_tidd_smooth"
    #         case ModelType.HDD_TIDD:
    #             return "hdd_tidd"
    #         case ModelType.TIDD_CDD_SMOOTH:
    #             return "tidd_cdd_smooth"
    #         case ModelType.TIDD_CDD:
    #             return "tidd_cdd"
    #         case ModelType.TIDD:
    #             return "tidd"


@overload(np.clip)
def np_clip(a, a_min, a_max):
    """
    This function applies a clip operation on the input array 'a' using the provided minimum and maximum values.
    The clip operation ensures that all elements in 'a' are within the range [a_min, a_max].
    If an element in 'a' is less than 'a_min', it is replaced with 'a_min'.
    If an element in 'a' is greater than 'a_max', it is replaced with 'a_max'.
    NaN values in 'a' are preserved as NaN.

    Parameters:
    a (numpy array): The input array to be clipped.
    a_min (float): The minimum value for the clip operation.
    a_max (float): The maximum value for the clip operation.

    Returns:
    numpy array: The clipped array.
    """

    @numba.vectorize
    def _clip(a, a_min, a_max):
        """
        This is a vectorized implementation of the clip function.
        It applies the clip operation on each element of the input array 'a'.

        Parameters:
        a (float): The input value to be clipped.
        a_min (float): The minimum value for the clip operation.
        a_max (float): The maximum value for the clip operation.

        Returns:
        float: The clipped value.
        """

        if np.isnan(a):
            return np.nan
        elif a < a_min:
            return a_min
        elif a > a_max:
            return a_max
        else:
            return a

    def clip_impl(a, a_min, a_max):
        """
        This is a numba implementation of the clip function.
        It applies the clip operation on the input array 'a' using the provided minimum and maximum values.

        Parameters:
        a (numpy array): The input array to be clipped.
        a_min (float): The minimum value for the clip operation.
        a_max (float): The maximum value for the clip operation.

        Returns:
        numpy array: The clipped array.
        """

        return _clip(a, a_min, a_max)

    return clip_impl


def OoM(x, method="round"):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    return OoM_numba(x, method=method)


@numba.jit(nopython=True, cache=numba_cache)
def OoM_numba(x, method="round"):
    """
    This function calculates the order of magnitude (OoM) of each element in the input array 'x' using the specified method.
    
    Parameters:
    x (numpy array): The input array for which the OoM is to be calculated.
    method (str): The method to be used for calculating the OoM. It can be one of the following:
                  "round" - round to the nearest integer (default)
                  "floor" - round down to the nearest integer
                  "ceil" - round up to the nearest integer
                  "exact" - return the exact OoM without rounding

    Returns:
    x_OoM (numpy array): The array of the same shape as 'x' containing the OoM of each element in 'x'.
    """

    x_OoM = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi == 0.0:
            x_OoM[i] = 1.0

        elif method.lower() == "floor":
            x_OoM[i] = np.floor(np.log10(np.abs(xi)))

        elif method.lower() == "ceil":
            x_OoM[i] = np.ceil(np.log10(np.abs(xi)))

        elif method.lower() == "round":
            x_OoM[i] = np.round(np.log10(np.abs(xi)))

        else:  # "exact"
            x_OoM[i] = np.log10(np.abs(xi))

    return x_OoM


def RoundToSigFigs(x, p):
    """
    This function rounds the input array 'x' to 'p' significant figures.

    Parameters:
    x (numpy.ndarray): The input array to be rounded.
    p (int): The number of significant figures to round to.

    Returns:
    numpy.ndarray: The rounded array.
    """

    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - OoM(x_positive))
    return np.round(x * mags) / mags


def t_stat(alpha, n, tail=2):
    """
    Calculate the t-statistic for a given alpha level, sample size, and tail type.

    Parameters:
    alpha (float): The significance level.
    n (int): The sample size.
    tail (int or str): The type of tail test. Can be 1 or "one" for one-tailed test, 
                       and 2 or "two" for two-tailed test. Default is 2.

    Returns:
    float: The calculated t-statistic.
    """

    degrees_of_freedom = n - 1
    if tail == "one" or tail == 1:
        perc = 1 - alpha
    elif tail == "two" or tail == 2:
        perc = 1 - alpha / 2

    return t_dist.ppf(perc, degrees_of_freedom, 0, 1)


def unc_factor(n, interval="PI", alpha=0.05):
    """
    Calculates the uncertainty factor for a given sample size, confidence interval type, and significance level.

    Parameters:
    n (int): The sample size.
    interval (str, optional): The type of confidence interval. Defaults to "PI" (Prediction Interval).
    alpha (float, optional): The significance level. Defaults to 0.05.

    Returns:
    float: The uncertainty factor.
    """

    if interval == "CI":
        return t_stat(alpha, n) / np.sqrt(n)

    if interval == "PI":
        return t_stat(alpha, n) * (1 + 1 / np.sqrt(n))


MAD_k = 1 / norm_dist.ppf(
    0.75
)  # Conversion factor from MAD to std for normal distribution


def median_absolute_deviation(x):
    """
    This function calculates the Median Absolute Deviation (MAD) of a given array.
    
    Parameters:
    x (numpy array): The input array for which the MAD is to be calculated.

    Returns:
    float: The calculated MAD of the input array.
    """

    mu = np.median(x)
    sigma = np.median(np.abs(x - mu)) * MAD_k

    return sigma


@numba.jit(nopython=True, cache=numba_cache)
def weighted_std(x, w, mean=None, w_sum_err=1e-6):
    """
    Calculate the weighted standard deviation of a given array.

    Parameters:
    x (numpy.ndarray): The input array.
    w (numpy.ndarray): The weights for the input array.
    mean (float, optional): The mean value. If None, the mean is calculated from the input array. Defaults to None.
    w_sum_err (float, optional): The error tolerance for the sum of weights. Defaults to 1e-6.

    Returns:
    float: The calculated weighted standard deviation.
    """

    n = float(len(x))

    w_sum = np.sum(w)
    if w_sum < 1 - w_sum_err or w_sum > 1 + w_sum_err:
        w /= w_sum

    if mean is None:
        mean = np.sum(w * x)

    var = np.sum(w * np.power((x - mean), 2)) / (1 - 1 / n)

    return np.sqrt(var)


def fast_std(x, weights=None, mean=None):
    """
    Function to calculate the approximate standard deviation quickly of a given array. 
    This function can handle both weighted and unweighted calculations.

    Parameters:
    x (numpy.ndarray): The input array for which the standard deviation is to be calculated.
    weights (numpy.ndarray, optional): An array of weights for the input array. Defaults to None.
    mean (float, optional): The mean of the input array. If not provided, it will be calculated. Defaults to None.

    Returns:
    float: The calculated standard deviation.
    """

    if isinstance(weights, int) or isinstance(weights, float):
        weights = np.array([weights])

    if weights is None or len(weights) == 1 or np.allclose(weights - weights[0], 0):
        if mean is None:
            return np.std(x)

        else:
            n = float(len(x))
            var = np.sum(np.power((x - mean), 2)) / n
            return np.sqrt(var)

    else:
        if mean is None:
            mean = np.average(x, weights=weights)

        return weighted_std(x, weights, mean)
