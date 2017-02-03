eemeter.modeling.formatters
---------------------------

The formatter classes are designed to provide a standard interface to model
fit and predict methods. The formatters add weather data to daily or monthly
energy data. The interface assumes that the model class will be responsible
for applying data sufficiency rules and additional formatting necessary for
performing model fits or predictions.

.. autoclass:: eemeter.modeling.formatters.ModelDataFormatter
    :members:

.. autoclass:: eemeter.modeling.formatters.ModelDataBillingFormatter
    :members:

eemeter.modeling.models
-----------------------

.. autoclass:: eemeter.modeling.models.seasonal.SeasonalElasticNetCVModel
    :members:

.. autoclass:: eemeter.modeling.models.billing.BillingElasticNetCVModel
    :members:

.. autoclass:: eemeter.modeling.models.caltrack.CaltrackMonthlyModel
    :members:
