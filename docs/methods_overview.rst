.. _methods-overview:

Methods Overview
----------------

The EEmeter provides multiple methods for calculating
:ref:`energy savings <glossary-energy-savings>`. All of these methods compare
:ref:`energy demand <glossary-energy-demand>` from a modeled counterfactual
pre-:ref:`intervention <glossary-intervention>`
:ref:`baseline <glossary-baseline>` to post-intervention energy
demand. Some of these methods, including the most conventional,
:ref:`weather normalize <glossary-weather-normalization>` energy demand.

These basic methods [#]_ rely on a :ref:`modeled <glossary-energy-model>`
relationship between weather patterns and energy demand. The particular models
used by the EEmeter are described more precisely in :ref:`modeling-overview`.

Modeling periods
^^^^^^^^^^^^^^^^

For any savings calculation, the period of time prior to the start of any
:ref:`interventions <glossary-intervention>` taking place as part of a project
we term the :ref:`baseline period <glossary-baseline-period>`.
This period is used to establish models of the relationship between
:ref:`energy demand <glossary-energy-demand>` and a set of factors that
represent or contribute to :ref:`end use demand <glossary-energy-demand>` (such as
weather, time of day, or day of week) for a particular building _prior_ to an
intervention. The :ref:`baseline <glossary-baseline>` becomes a reference
point from which to make comparisions to post-intervention energy performance.
The baseline period is one of two types of
:ref:`modeling period <glossary-modeling-period>` frequently occurring in
the EEmeter.

The second half of the savings calculation concerns what happens after an
intervention. Any post-intervention period for which energy savings is
calculated is called a :ref:`reporting period <glossary-reporting-period>`
because it is the period of time over which energy savings is reported. A
project generally has only one
:ref:`baseline period <glossary-baseline-period>`, but it might have multiple
:ref:`reporting periods <glossary-reporting-period>`. These are
the second type of :ref:`modeling period <glossary-modeling-period>` to
frequent occur in the EEmeter.

The extent of these periods will, in most cases, be determined by the
start and end dates of the interventions in a project. However, in some cases,
the intervention dates are not known, or are ongoing, and must be *modeled*
because they cannot be stated explicitly. We refer to models which account for
the latter scenario as
:ref:`structural change models <glossary-structural-change-model>`;
these are covered in greater detail in :ref:`modeling-overview`.

EEmeter structures which capture this logic can be found in the API documentation
for :ref:`eemeter-structures`.

.. figure:: project-timeline-illustration.png

    Pre-intervention baseline period and post-intervention reporting periods
    on a project timeline.

Trace modeling
^^^^^^^^^^^^^^

The relationship between energy demand and various external factors can differ
drastically from building to building, and (usually!) changes after an
intervention. Modeling these relationships properly with statistical confidence
is a core strength of the EEmeter.

As noted in the :ref:`background <background>`, we term a set of
energy data points a :ref:`trace <glossary-trace>`, and a building or project
might be associated with any number of traces. In order to calculate savings
models, each of these traces must be modeled.

Before modeling, traces are segmented into components which overlap each
baseline and reporting period of interest, then are modeled separately. [#]_
This creates up to :math:`n * m` models for a project with :math:`n` traces
and :math:`m` modeling periods.

Each of these models attempts to establish the relationship between
:ref:`energy demand <glossary-energy-demand>` and external factors as it performed during the
particular modeling period of interest. However, since the extent to which a
model successfully describes these relationships varies significantly, these
must be considered only in conjunction with model error and goodness of fit
metrics :ref:`modeling-overview`. Any estimate of energy demand given by any
model fitted by the EEmeter is associated with variance and confidence bounds.

In practice the number of models fitted for any particular project might be
fewer than :math:`n * m` due to missing or insufficient data
(see :ref:`data-sufficiency`). The EEmeter takes these failures into account
and considers them when building summaries of savings.

.. figure:: trace-segmenting-illustration.png

    An example of trace segmenting with two traces,
    one baseline period and one reporting period. **Trace 1** is segmented
    into just one component - the baseline component - because data for the
    reporting period is missing. **Trace 2** is segmented into one baseline
    component and one reporting component. The segments of **Trace 1** and
    **Trace 2** have different lengths, but models of their energy demand
    behavior can still be built.

Weather normalization
^^^^^^^^^^^^^^^^^^^^^

Once we have created a model, we can apply that model determine an estimate of
of energy demand during arbitrary weather scenarios. The two most common
weather scenarios for which the EEmeter will estimate demand are the
":ref:`normal <glossary-weather-normal>`" weather year and the observed
reporting period weather year. This is generally necessary because the data
observed in the baseline and reporting periods occurred during different
time periods with different weather -- and valid comparisons between them must
account for this. Estimating energy performance during the "normal" weather
attempts to reduce bias in the savings estimate by accounting for the
peculiarity (as compared to other years or seasons) of the relevant observed
weather.

In an attempt to reduce the number of arbitrary factors influencing results,
we only ever compare model estimates or data over that has occurred over the
same weather scenario and time period. This helps (in the aggregate) to ensure
equivalency of :ref:`end use demand <glossary-end-use-demand>` pre- and post-
intervention.

Savings
^^^^^^^

If the data and models show that
:ref:`energy demand <glossary-energy-demand>` is reduced relative to
equivalent :ref:`end use demand <glossary-end-use-demand>`
following an intervention, we say that there have been energy savings, or
equivalently, that energy performance has increased.

Energy savings is necessarily a difference; however, this difference must be
taken carefully, given missing data and model error, and is only taken *after*
the necessary :ref:`aggregation <aggregation>` steps.

The equation for savings is always:

    :math:`S_\text{total} = E_\text{b} - E_\text{r}`

or

    :math:`S_\text{percent} = \frac{E_\text{b} - E_\text{r}}{E_\text{b}}`

where

    - :math:`S_\text{total}` is aggregate total savings
    - :math:`S_\text{percent}` is aggregate percent savings
    - :math:`E_\text{b}` is aggregate energy demand as under baseline period conditions
    - :math:`E_\text{r}` is aggregate energy demand as under reporting period conditions

Depending on the type of energy savings desired, the values :math:`E_\text{b}`
and :math:`E_\text{r}` may be calculated differently. The following types of
savings are supported:

- :ref:`annualized-weather-normal`
- :ref:`gross-predicted`
- :ref:`gross-observed`

.. _annualized-weather-normal:

Annualized weather normal
"""""""""""""""""""""""""

The :ref:`annualized weather normal <glossary-annualized-weather-normal>`
estimates savings as it may have occurred during a
:ref:`"normal" weather <glossary-weather-normal>` year. It does this by
building models of both the baseline and reporting energy demand and using
each to weather-normalize the energy values.

    :math:`E_\text{b} = \text{M}_\text{b}\left(\text{X}_\text{normal}\right)`

    :math:`E_\text{r} = \text{M}_\text{r}\left(\text{X}_\text{normal}\right)`

where

    - :math:`\text{M}_\text{b}` is the model of energy demand as built using
      trace data segmented from the baseline period.

    - :math:`\text{M}_\text{r}` is the model of energy demand as built using
      trace data segmented from the reporting period.

    - :math:`\text{X}_\text{normal}` are temperature and other covariate
      values for the weather normal year.

.. _gross-predicted:

Gross predicted
"""""""""""""""

The :ref:`gross predicted <glossary-gross-predicted>` method
estimates savings that have occurred from the completion of the project
interventions up to the date of the meter run.

    :math:`E_\text{b} = \text{M}_\text{b}\left(\text{X}_\text{r}\right)`

    :math:`E_\text{r} = \text{M}_\text{r}\left(\text{X}_\text{r}\right)`

where

    - :math:`\text{M}_\text{b}` is the model of energy demand as built using
      trace data segmented from the baseline period.

    - :math:`\text{M}_\text{r}` is the model of energy demand as built using
      trace data segmented from the reporting period.

    - :math:`\text{X}_\text{r}` are temperature and other covariate
      values for reporting period.

.. _gross-observed:

Gross observed
""""""""""""""

The :ref:`gross observed <glossary-gross-observed>` method
estimates savings that have occurred from the completion of the project
interventions up to the date of the meter run.

    :math:`E_\text{b} = \text{M}_\text{b}\left(\text{X}_\text{r}\right)`

    :math:`E_\text{r} = \text{A}_\text{r}`

where

    - :math:`\text{M}_\text{b}` is the model of energy demand as built using
      trace data segmented from the baseline period.

    - :math:`\text{A}_\text{r}` are the actual observed energy demand values
      from the trace data segmented from the baseline period. If the actual
      data has missing values, these are interpolated using gross predicted
      values (i.e., :math:`\text{M}_\text{r}\left(\text{X}_\text{r}\right)`).

    - :math:`\text{X}_\text{r}` are temperature and other covariate
      values for reporting period.

.. _aggregation:

Aggregation rules
^^^^^^^^^^^^^^^^^

Because even an individual project may have multiple traces describing its
energy demand, we must be able to aggregate trace-level results before we can
obtain project-level or portfolio-level savings. Ideally, this aggregation is
a simple sum of trace-level values. However, trace-level results are often
littered with messy results which must be accounted for; some may be missing
data, have bad model fits, or have entirely failed model builds. The EEmeter
must successfully handle each of these cases, or risk invalidating results for
entire portfolios.

The aggregation steps are as follows:

1. Select scope (project, portfolio) and gather all trace data available in
   that scope
2. Select baseline and reporting period. For portfolio level aggregations in
   which baseline and reporting periods may not align, select reporting period
   type and use the default baseline period for each project.
3. Group traces by :ref:`interpretation <glossary-trace-interpretation>`
4. Compute :math:`E_\text{b}` and :math:`E_\text{r}`:

    a. Compute (or retrieve) :math:`E_\text{t,b}` and :math:`E_\text{t,r}` for
       each trace :math:`\text{t}`.
    b. Determine, for each :math:`E_\text{t,b}` and :math:`E_\text{t,r}` whether
       or not it meets :ref:`criteria <inclusion-criteria>` for
       inclusion in aggregation.
    c. Discard *both* :math:`E_\text{t,b}` and :math:`E_\text{t,r}` for any trace
       for which either :math:`E_\text{t,b}` or :math:`E_\text{t,r}` has been
       discarded.
    d. Compute :math:`E_\text{b} = \sum_{\text{t}}E_\text{t,b}`
       and :math:`E_\text{r} = \sum_{\text{t}}E_\text{t,r}` for remaining
       traces. Errors are propgated according to the principles in
       :ref:`error-propogation`.

5. Compute savings from :math:`E_\text{b}` and :math:`E_\text{r}` as usual.

.. _inclusion-criteria:

Inclusion criteria
""""""""""""""""""

For inclusion in aggregates, :math:`E_\text{t,b}` and :math:`E_\text{t,r}` must
meet the following criteria


1. If :code:`ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED`, which represents solar
   generation, is available, and if solar panels were installed as one of the
   project interventions, blank :math:`E_\text{t,b}` should be replaced with 0.
2. Model has been successfully built.

.. _error-propogation:

Error propogation
^^^^^^^^^^^^^^^^^

Errors are propgated as if they followed :math:`\chi^2` distributions.

.. _weather-data-matching:

Weather data matching
^^^^^^^^^^^^^^^^^^^^^

Since weather and temperature data is so central to the activity of the
EEmeter, the particulars of how weather data is obtained for a project is often
of interest. Weather data sources are determined automatically within the
EEmeter using an internal mapping [#]_ bewteen ZIP codes [#]_ and weather
stations. The source of the weather normal data may differ from the source of
the observed weather data.

There is a `jupyter <https://jupyter.org/>`_ notebook outlining the process of
constructing the weather data available
`here <https://github.com/openeemeter/eemeter/blob/master/scripts/weather_stations_zipcodes_climate_zones.ipynb>`_.


.. [#] Additional information on *why* this method is used in preference to
   other methods is described in the :ref:`introduction`.

.. [#] This is not quite true for
   :ref:`structural change models <glossary-structural-change-model>`. This is
   covered in more detail in :ref:`modeling-overview`.

.. [#] Available `on github <https://github.com/openeemeter/eemeter/tree/master/eemeter/resources>`_.

.. [#] The ZIP codes used in this mapping aren't strictly ZIP codes, they're
   actually :ref:`ZCTAs <glossary-zip-code-tabulation-area>`.
