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
might be associated with any number of these. They must each be modeled.
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

Weather normalization
^^^^^^^^^^^^^^^^^^^^^

Once we have created a model, we can apply that model determine an estimate of
of energy demand during arbitrary weather scenarios. The two most common
weather scenarios for which the EEmeter will estimate demand are the
":ref:`normal <glossary-weather-normal>`" weather year and the observed
reporting period weather year. This is generally necessary because the data
observed in the baseline and reporting periods occured during different
time periods with different weather -- and valid comparisons between them must
account for this. Estimating energy performance during the "normal" weather
attempts to reduce bias in the savings estimate by accounting for the
peculiarity (as compared to other years or seasons) of the relevant observed
weather.

In an attempt to reduce the number of arbitrary factors influencing results,
we only ever compare model estimates or data over that has occured over the
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

:math:`\text{total savings} = \text{baseline} - \text{reporting}`

:math:`\text{percent savings} = \frac{\text{baseline} - \text{reporting}}{\text{baseline}}`

Annualized weather normal
"""""""""""""""""""""""""

To calculate :ref:`annualized weather normal <glossary-annualized-weather-normal>`
energy demand for a particular trace, the EEmeter selects the segment of that
trace data occuring in the baseline period and the segment occuring in the
reporting period and builds two models. It then
:ref:`weather normalizes <glossary-weather-normalization>` the observed data
for each segment according to the :ref:`matched <weather-data-matching>`
weather normal.

Gross predicted
"""""""""""""""


Gross observed
""""""""""""""

To calculate gross observed :ref:`energy savings <glossary-energy-savings>`,
the EEmeter selects a sample of :ref:`trace <glossary-trace>` data prior to an
:ref:`intervention <glossary-intervention>` and estimates usage as it would
have occured under reporting period the reporting period, using observed
reporting period weather to establish a :ref:`baseline <glossary-baseline>`.
For the reporting period, observed data is used.

.. _aggregation:

Aggregation rules
"""""""""""""""""

...

.. _weather-data-matching:

Weather data matching
"""""""""""""""""""""



:ref:`ZCTAs <glossary-zip-code-tabulation-area>`

.. [#] Additional information on *why* this method is used in preference to
   other methods is described in the introduction.

.. [#] This is not quite true for
   :ref:`structural change models <glossary-structural-change-model>`. This is
   covered in more detail in :ref:`modeling-overview`.

