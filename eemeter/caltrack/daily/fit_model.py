import numpy as np
import pandas as pd
from scipy.stats import linregress

import itertools
from copy import deepcopy as copy
from timeit import default_timer as timer

from eemeter.caltrack.daily.utilities.config import (
    DailySettings,
    caltrack_2_1_settings,
    caltrack_legacy_settings,
    update_daily_settings,
)
from eemeter.caltrack.daily.utilities.selection_criteria import selection_criteria
from eemeter.caltrack.daily.utilities.ellipsoid_test import ellipsoid_split_filter

from eemeter.caltrack.daily.fit_base_models import (
    fit_initial_models_from_full_model,
    fit_final_model,
)


# TODO: 'check_caltrack_compliant' and check constraints where possible


class FitModel:
    def __init__(
        self,
        meter_data,
        model="2.1",
        settings=None,
        check_caltrack_compliant=True,
        verbose=False,
    ):
        """
        A class to fit a model to the input meter data.

        Parameters:
        -----------
        meter_data : pandas.DataFrame
            A dataframe containing meter data.
        model : str, optional
            The model to use. Default is '2.1'.
        settings : dict, optional
            DailySettings to be changed. Default is None.
        check_caltrack_compliant : bool, optional
            Whether to check if the data is CalTRACK compliant. Default is True.
        verbose : bool, optional
            Whether to print verbose output. Default is False.

        Attributes:
        -----------
        settings : dict
            A dictionary of settings.
        seasonal_options : list
            A list of seasonal options.
        day_options : list
            A list of day options.
        combo_dictionary : dict
            A dictionary of combinations.
        df_meter : pandas.DataFrame
            A dataframe of meter data.
        error : dict
            A dictionary of error metrics.
        combinations : list
            A list of combinations.
        components : list
            A list of components.
        fit_components : list
            A list of fit components.
        wRMSE_base : float
            The mean bias error for no splits.
        best_combination : list
            The best combination of splits.
        model : sklearn.pipeline.Pipeline
            The final fitted model.
        id : str
            The index of the meter data.
        """

        # Initialize settings
        # Note: Model designates the base settings, it can be '2.1' or '2.0'
        #       Settings is to be a dictionary of settings to be changed

        if settings is None:
            settings = {}

        if model.replace(" ", "").replace("_", ".").lower() in ["caltrack2.1", "2.1"]:
            self.settings = caltrack_2_1_settings(**settings)
        elif model.replace(" ", "").replace("_", ".").lower() in ["caltrack2.0", "2.0"]:
            self.settings = caltrack_legacy_settings(**settings)
        else:
            raise Exception(
                "Invalid 'settings' choice: must be 'CalTRACK 2.1', '2.1', 'CalTRACK 2.0', or '2.0'"
            )

        # Initialize seasons and weekday/weekend
        self.seasonal_options = [
            ["su_sh_wi"],
            ["su", "sh_wi"],
            ["su_sh", "wi"],
            ["su_wi", "sh"],
            ["su", "sh", "wi"],
        ]
        self.day_options = [["wd", "we"]]

        n_week = list(range(len(self.settings.is_weekday)))
        self.combo_dictionary = {
            "su": "summer",
            "sh": "shoulder",
            "wi": "winter",
            "fw": [n + 1 for n in n_week],
            "wd": [n + 1 for n in n_week if self.settings.is_weekday[n + 1]],
            "we": [n + 1 for n in n_week if not self.settings.is_weekday[n + 1]],
        }

        # Initialize dataframe
        self.df_meter = self._initialize_data(meter_data)

        self.verbose = verbose

        self.error = {
            "wRMSE_train": np.nan,
            "wRMSE_test": np.nan,
            "RMSE_train": np.nan,
            "RMSE_test": np.nan,
            "MAE_train": np.nan,
            "MAE_test": np.nan,
        }

        # Begin fitting
        self.combinations = self._combinations()
        self.components = self._components()
        self.fit_components = self._fit_components()

        self.wRMSE_base = self._get_error_metrics("fw-su_sh_wi")[
            0
        ]  # calculate mean bias error for no splits
        self.best_combination = self._best_combination(print_out=False)
        self.model = self._final_fit(self.best_combination)

        self.id = meter_data.index.unique()[0]
        wRMSE, RMSE, MAE = self._get_error_metrics(self.best_combination)
        self.error["wRMSE_train"] = wRMSE
        self.error["RMSE_train"] = RMSE
        self.error["MAE_train"] = MAE

    def _initialize_data(self, meter_data):
        """
        Initializes the meter data by performing the following operations:
        - Renames the 'model' column to 'model_old' if it exists
        - Converts the index to a DatetimeIndex if it is not already
        - Adds a 'season' column based on the month of the index using the settings.season dictionary
        - Adds a 'day_of_week' column based on the day of the week of the index
        - Removes any rows with NaN values in the 'temperature' or 'observed' columns
        - Sorts the data by the index
        - Reorders the columns to have 'season' and 'day_of_week' first, followed by the remaining columns

        Parameters:
        - meter_data: A pandas DataFrame containing the meter data

        Returns:
        - A pandas DataFrame containing the initialized meter data
        """

        if "model" in meter_data.columns:
            meter_data = meter_data.rename(columns={"model": "model_old"})

        cols = list(meter_data.columns)

        if "datetime" in cols:
            meter_data.set_index("datetime", inplace=True)
            cols.remove("datetime")

        if not isinstance(meter_data.index, pd.DatetimeIndex):
            try:
                meter_data.index = pd.to_datetime(meter_data.index)
            except:
                raise TypeError("Could not convert 'meter_data.index' to datetime")

        for col in ["season", "day_of_week"]:
            if col in cols:
                meter_data.drop([col], axis = 1, inplace = True)
                cols.remove(col)

        meter_data["season"] = meter_data.index.month.map(self.settings.season)
        meter_data["day_of_week"] = meter_data.index.dayofweek + 1
        meter_data = meter_data[np.isfinite(meter_data["temperature"])]
        if "observed" in cols:
            meter_data = meter_data[np.isfinite(meter_data["temperature"])]
        meter_data = meter_data.sort_index()
        meter_data = meter_data[["season", "day_of_week", *cols]]

        return meter_data

    def _combinations(self):
        """
        This method generates all possible combinations of seasonal and day options for the given data.
        It then trims the combinations based on certain conditions such as minimum number of days per season,
        and whether to allow separate splits for summer, shoulder and winter seasons.
        """

        settings = self.settings

        def _get_combinations():
            def add_prefix(list_str, prefix):
                return [f"{prefix}-{s}" for s in list_str]

            def expand_combinations(combos_in):
                """
                Given a list of combinations, expands each combination by adding a new item to it.
                The new item is chosen from the intersection of the items in two specific combinations.
                The new item is then added to a third combination, which is created by combining the remaining items from the two specific combinations.
                The resulting expanded combinations are returned as a list.

                Parameters:
                combos_in (list): A list of combinations, where each combination is a list of items.

                Returns:
                list: A list of expanded combinations, where each expanded combination is a list of items.
                """

                combo_expanded = []
                for combo in combos_in:
                    combo_expanded.append(list(combo))
                    prefixes = [item[0] for item in combo]

                    if "wd" in prefixes and "we" in prefixes:
                        i_wd = prefixes.index("wd")
                        i_we = prefixes.index("we")
                    else:
                        continue

                    if "fw" in prefixes:
                        i_fw = prefixes.index("fw")
                    else:
                        i_fw = None

                    for item in combo[i_wd][1]:
                        if item in combo[i_we][1]:
                            combo_0_trim = [x for x in combo[i_wd][1] if x != item]
                            combo_1_trim = [x for x in combo[i_we][1] if x != item]

                            if i_fw is None:
                                fw_item = ["fw", [item]]
                            else:
                                fw_item = ["fw", [*combo[i_fw][1], item]]

                            if len(combo_0_trim) == 0 and len(combo_1_trim) == 0:
                                combo_new = [fw_item]
                            elif len(combo_0_trim) > 0 and len(combo_1_trim) == 0:
                                combo_new = [fw_item, [combo[i_wd][0], combo_0_trim]]
                            elif len(combo_0_trim) == 0 and len(combo_1_trim) > 0:
                                combo_new = [fw_item, [combo[i_we][0], combo_1_trim]]
                            else:
                                combo_new = [
                                    fw_item,
                                    [combo[i_wd][0], combo_0_trim],
                                    [combo[i_we][0], combo_1_trim],
                                ]

                            combo_expanded.append(combo_new)

                return combo_expanded

            def stringify(combos):
                """
                Converts a list of tuples into a list of strings, where each string is a combination of the tuple values
                separated by '__'. The tuples are expected to have a prefix and a value, and the prefix is used to add context
                to the value.

                Parameters:
                    combos (list): A list of tuples, where each tuple contains a prefix and a value.

                Returns:
                    list: A list of strings, where each string is a combination of the tuple values separated by '__'.
                """

                combos_str = []
                for combo in combos:
                    combo = [add_prefix(item[1], item[0]) for item in combo]
                    combo = [item for sublist in combo for item in sublist]
                    combo = "__".join(combo)

                    combos_str.append(combo)

                combos_str = sorted(list(set(combos_str)), key=lambda x: (len(x), x))

                return combos_str

            for days in self.day_options:
                season_day_combo = []
                for day in days:
                    season_day_combo.append(
                        list(itertools.product([day], self.seasonal_options))
                    )

                combos_expanded = list(itertools.product(*season_day_combo))
                for _ in range(max([len(item) for item in self.seasonal_options])):
                    combos_expanded = expand_combinations(combos_expanded)

                combos_str = stringify(combos_expanded)

            return combos_str

        def _trim_combinations(combo_list, split_min_days=30):
            """
            Trims the list of combinations to be tested based on various conditions.
            - Checks if the ellipsoids created are separated enough to warrant separate seasons and weekday/weekend splits.
            - Checks if there are enough days in each season and weekday/weekend to warrant separate splits.

            Args:
                combo_list (list): List of combinations to be tested.
                split_min_days (int, optional): Minimum number of days required for a split. Defaults to 30.

            Returns:
                list: Trimmed list of combinations to be tested.
            """

            meter = self.df_meter
            allow_sep_summer = settings.allow_separate_summer
            allow_sep_shoulder = settings.allow_separate_shoulder
            allow_sep_winter = settings.allow_separate_winter
            allow_sep_weekday_weekend = settings.allow_separate_weekday_weekend

            if settings.reduce_splits_by_gaussian:
                allow_split = ellipsoid_split_filter(
                    self.df_meter, n_std=settings.reduce_splits_num_std
                )

                if allow_sep_summer and not allow_split["summer"]:
                    allow_sep_summer = False

                if allow_sep_shoulder and not allow_split["shoulder"]:
                    allow_sep_shoulder = False

                if allow_sep_winter and not allow_split["winter"]:
                    allow_sep_winter = False

                if allow_sep_weekday_weekend and not allow_split["weekday_weekend"]:
                    allow_sep_weekday_weekend = False

            we_days = self.combo_dictionary["we"]

            if (meter["season"].values == "summer").sum() < split_min_days:
                allow_sep_summer = False

            if (meter["season"].values == "shoulder").sum() < split_min_days:
                allow_sep_shoulder = False

            if (meter["season"].values == "winter").sum() < split_min_days:
                allow_sep_winter = False

            combo_list_trimmed = []
            for combo in combo_list:
                if "fw-su_sh_wi" == combo:  # always fit the full model with all data
                    combo_list_trimmed.append(combo)
                    continue
                elif "wd" in combo and not allow_sep_weekday_weekend:
                    continue

                banned_season_split = {
                    "su": not allow_sep_summer,
                    "sh": not allow_sep_shoulder,
                    "wi": not allow_sep_winter,
                }

                valid_combo = True
                components = [item[3:] for item in combo.split("__")]
                for component in components:
                    seasons = component.split("_")

                    if (len(seasons) == 1) and banned_season_split[component]:
                        valid_combo = False
                        break

                    we_count = 0
                    for season in seasons:
                        we_count += (
                            (meter["season"].values == self.combo_dictionary[season])
                            & meter["day_of_week"].isin(we_days).values
                        ).sum()

                    if we_count < split_min_days / 3.75:
                        valid_combo = False
                        break

                if valid_combo:
                    combo_list_trimmed.append(combo)

            return combo_list_trimmed

        def _remove_duplicate_permutations(combo_list):
            """
            Removes duplicate permutations from a list of strings.

            Args:
                combo_list (list): A list of strings representing permutations.

            Returns:
                list: A list of unique permutations.
            """

            unique_sorted_combos = []
            unique_combos = []
            for combo in combo_list:
                sorted_combo = "__".join(sorted(combo.split("__")))

                if sorted_combo not in unique_sorted_combos:
                    unique_sorted_combos.append(combo)
                    unique_combos.append(combo)

            return unique_combos

        combo_list = _get_combinations()
        combo_list = _remove_duplicate_permutations(combo_list)
        combo_list = _trim_combinations(combo_list)

        return combo_list

    def _meter_segment(self, component, meter=None):
        """
        Returns a meter segment based on the given component and meter data.

        Parameters:
            component (str): A string representing the component to filter the meter data by.
            meter (pandas.DataFrame, optional): A pandas DataFrame containing the meter data. Defaults to None.

        Returns:
            pandas.DataFrame: A pandas DataFrame containing the meter data filtered by the given component.
        """

        if meter is None:
            meter = self.df_meter

        season_list = component[3:].split("_")
        day_list = component[:2]

        seasons = [self.combo_dictionary[key] for key in season_list]
        days = self.combo_dictionary[day_list]

        meter_segment = meter[
            meter["season"].isin(seasons) & meter["day_of_week"].isin(days)
        ]

        return meter_segment

    # TODO: rename components to submodel or submodel to component? Likely the first
    def _components(self):
        """
        Returns a sorted list of unique components from the combinations attribute.
        """

        components = list(
            set([i for item in self.combinations for i in item.split("__")])
        )
        components = sorted(components, key=lambda x: (len(x), x))

        return components

    def _fit_components(self):
        """
        Fits initial models for each component using the meter segment data and component settings.

        If the alpha_final_type is "last", the settings are updated to disable the final bounds scalar and set alpha_final_type to None.

        Returns:
            dict: A dictionary containing the fitted components.
        """

        if self.settings.alpha_final_type == "last":
            settings_update = {
                "developer_mode": True,
                "alpha_final_type": None,
                "final_bounds_scalar": None,
            }

            self.component_settings = update_daily_settings(
                self.settings, settings_update
            )
        else:
            self.component_settings = self.settings

        fit_components = {item: None for item in self.components}
        for component in fit_components.keys():
            meter_segment = self._meter_segment(component)

            # Fit new models
            fit_components[component] = fit_initial_models_from_full_model(
                meter_segment, self.component_settings, print_res=False
            )

        return fit_components

    def _combination_selection_criteria(self, combination):
        """
        Calculates the selection criteria for a given combination of components.

        Parameters:
            combination (str): A string representing the combination of components.

        Returns:
            float: The selection criteria for the given combination.
        """
         
        components = combination.split("__")

        N = np.sum([self.fit_components[X].N for X in components])
        TSS = np.sum([self.fit_components[X].TSS for X in components])
        # starts as added penalties based on # splits
        # num_coeffs = 3*self.df_penalties[combination] # + np.sum([self.fit_components[X].num_coeffs for X in components])
        num_coeffs = len(components)

        if combination == "fw-su_sh_wi":
            wRMSE = self.wRMSE_base
        else:
            wRMSE = self._get_error_metrics(combination)[0]

        loss = wRMSE / self.wRMSE_base

        criteria_type = self.settings.split_selection_criteria.lower()
        penalty_multiplier = self.settings.split_selection_penalty_multiplier
        penalty_power = self.settings.split_selection_penalty_power

        criteria = selection_criteria(
            loss, TSS, N, num_coeffs, criteria_type, penalty_multiplier, penalty_power
        )

        return criteria

    def _best_combination(self, print_out=False):
        """
        Finds the best combination of parameters based on the selection criteria.

        Parameters:
            print_out (bool): Whether to print the combination and selection criteria for each iteration.

        Returns:
            str: The best combination of parameters as a string.
        """

        HoF = {"combination_str": None, "selection_criteria": np.inf}
        for combo in self.combinations:
            selection_criteria = self._combination_selection_criteria(combo)

            if selection_criteria < HoF["selection_criteria"]:
                HoF["combination_str"] = combo
                HoF["selection_criteria"] = selection_criteria

            if print_out:
                print(f"{combo:>40s} {selection_criteria:>8.1f}")

        if print_out:
            print(f"{HoF['combination_str']:>40s} {HoF['selection_criteria']:>8.1f}")

        return HoF["combination_str"]

    def _final_fit(self, combination):
        """
        Fits the final model for a given combination of components.

        Parameters:
            combination (str): A string representing the combination of components.

        Returns:
            dict: A dictionary containing the fitted models for each component in the combination.
        """

        model = {}
        for component in combination.split("__"):
            settings = self.settings
            prior_model = self.fit_components[component]

            if settings.alpha_final_type is None:
                if self.verbose:
                    print(f"{component}__{prior_model.model_name}")

                model[component] = prior_model
                continue

            settings_update = {"developer_mode": True, "regularization_alpha": 0.0}
            settings = update_daily_settings(self.settings, settings_update)

            # separate meter appropriately
            meter_segment = self._meter_segment(component)

            # Fit new models
            if self.verbose:
                print(f"{component}__{prior_model.model_name}")

            model[component] = fit_final_model(
                meter_segment, prior_model, settings, print_res=self.verbose
            )

            model[component].settings = self.settings  # overwrite to input settings

        return model

    def _get_error_metrics(self, combination):
        """
        Calculates the error metrics for a given combination of components.
        RMSE and MAE are calculated as the mean of the residuals, wRMSE is calculated as the weighted mean of the residuals.

        Parameters:
            combination (str): A string representing the combination of components to calculate error metrics for.
                If None, the best combination will be used.

        Returns:
            tuple: A tuple containing the calculated error metrics (wRMSE, RMSE, MAE).
        """

        if combination is None:
            combination = self.best_combination

        N = 0
        wSSE = 0
        resid = []
        for component in combination.split("__"):
            fit_component = self.fit_components[component]

            wSSE += fit_component.wSSE
            N += fit_component.N
            resid.append(fit_component.resid)

        resid = np.hstack(resid)

        wRMSE = np.sqrt(wSSE / N)
        RMSE = np.mean(resid**2) ** 0.5
        MAE = np.mean(np.abs(resid))

        return wRMSE, RMSE, MAE

    def evaluate(self, df_eval):
        """
        Evaluates the model on the given evaluation dataframe.

        Parameters:
            df_eval (pandas.DataFrame): The evaluation dataframe.

        Returns:
            pandas.DataFrame: The evaluation dataframe with model predictions added.
        """

        # initialize data to input dataframe
        df_eval = self._initialize_data(df_eval)

        df_all_models = []
        for component_key in self.model:
            eval_segment = self._meter_segment(component_key, df_eval)
            T = eval_segment["temperature"].values

            model, unc, hdd_load, cdd_load = self.model[component_key].eval(T)

            df_model = pd.DataFrame(
                data={
                    "model": model,
                    "model_unc": unc,
                    "heating_load": hdd_load,
                    "cooling_load": cdd_load,
                },
                index=eval_segment.index,
            )
            df_model["model_split"] = component_key
            df_model["model_type"] = self.model[component_key].model_name
            df_model["model_alpha"] = self.model[
                component_key
            ].loss_alpha  # TODO: remove?

            df_all_models.append(df_model)

        df_model_prediction = pd.concat(df_all_models, axis=0)

        df_eval = df_eval.join(df_model_prediction)

        return df_eval
