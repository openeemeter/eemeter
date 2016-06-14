import pandas as pd


class TraceModeler(object):

    freq_str = None

    def __init__(self, model, modeling_period, trace):
        self.model = model
        self.modeling_period = modeling_period
        self.trace = trace
        self.params = None
        self.invalid_fit = None

    def create_model_input(self, weather_source):
        '''Return a `DatetimeIndex`ed dataframe containing necessary model data
        '''
        trace_values = self.trace.data.value.resample(self.freq_str).sum()
        temp_values = weather_source.indexed_temperatures(trace_values.index,
                                                          "degF")

        return pd.DataFrame({"tempF": temp_values, "energy": trace_values},
                            columns=["energy", "tempF"])

    def create_demand_fixture(self, weather_fixture_source, index):
        temp_values = weather_fixture_source \
            .indexed_temperatures(index, "degF")

        return pd.DataFrame({"tempF": temp_values})

    def evaluate(self, df):
        try:
            params, cvrmse, r2 = self.model.fit(df)
        except ValueError:
            # fit failed, probably because of insufficient data.
            self.invalid_fit = True
            return None, None, None
        else:
            self.invalid_fit = False
            self.params = params
            return params, r2, cvrmse

    def predict(self, df):
        if self.invalid_fit is None:
            message = (
                'Please, run `modeler.evaluate()` before predicting.'
            )
            raise ValueError(message)
        elif self.invalid_fit:
            message = (
                'Fit invalid, please check input data.'
                ' Cannot call `.predict()` without a valid fit.'
            )
            raise ValueError(message)

        return self.model.predict(df, self.params)


class DailyModeler(TraceModeler):

    freq_str = "D"
