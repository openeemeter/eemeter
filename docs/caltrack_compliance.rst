CalTRACK Compliance
===================


Checklist for caltrack compliance:

Section 2.2.1
-------------

- **2.2.1.1**: :any:`eemeter.get_baseline_data` must set ``max_days=365``.
- **2.2.1.2**: :any:`eemeter.caltrack_sufficiency_criteria` must set ``min_fraction_daily_coverage=0.9``.
- **2.2.1.3**: Missing values in input data are represented as ``float('nan')``, ``np.nan``, or anything recognized as null by the method :any:`pandas.isnull`.
- **2.2.1.4**: Values of ``0`` in electricity data have been converted to ``np.nan``.


Section 2.2.2
-------------

- **2.2.2.1**: Input meter data should be appropriately downsampled to daily outside of EEmeter.
- **2.2.2.2**: Estimated reads in input data have been combined with subsequent reads.
- **2.2.2.3**: :any:`eemeter.merge_temperature_data` must set ``percent_hourly_coverage_per_day=0.5``.
