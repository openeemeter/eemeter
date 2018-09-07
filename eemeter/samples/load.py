import json
from pkg_resources import resource_stream

from dateutil.parser import parse as parse_date
import pytz

from eemeter import meter_data_from_csv, temperature_data_from_csv

__all__ = ("samples", "load_sample")


def _load_sample_metadata():
    with resource_stream("eemeter.samples", "metadata.json") as f:
        metadata = json.loads(f.read().decode("utf-8"))
    return metadata


def samples():
    """ Load a list of sample data identifiers.

    Returns
    -------
    samples : :any:`list` of :any:`str`
        List of sample identifiers for use with :any:`eemeter.load_sample`.
    """
    sample_metadata = _load_sample_metadata()
    return list(sorted(sample_metadata.keys()))


def load_sample(sample):
    """ Load meter data, temperature data, and metadata for associated with a
    particular sample identifier. Note: samples are simulated, not real, data.

    Parameters
    ----------
    sample : :any:`str`
        Identifier of sample. Complete list can be obtained with
        :any:`eemeter.samples`.

    Returns
    -------
    meter_data, temperature_data, metadata : :any:`tuple` of :any:`pandas.DataFrame`, :any:`pandas.Series`, and :any:`dict`
        Meter data, temperature data, and metadata for this sample identifier.
    """
    sample_metadata = _load_sample_metadata()
    metadata = sample_metadata.get(sample)
    if metadata is None:
        raise ValueError(
            "Sample not found: {}. Try one of these?\n{}".format(
                sample,
                "\n".join(
                    [" - {}".format(key) for key in sorted(sample_metadata.keys())]
                ),
            )
        )

    freq = metadata.get("freq")
    if freq not in ("hourly", "daily"):
        freq = None

    meter_data_filename = metadata["meter_data_filename"]
    with resource_stream("eemeter.samples", meter_data_filename) as f:
        meter_data = meter_data_from_csv(f, gzipped=True, freq=freq)

    temperature_filename = metadata["temperature_filename"]
    with resource_stream("eemeter.samples", temperature_filename) as f:
        temperature_data = temperature_data_from_csv(f, gzipped=True, freq="hourly")

    metadata["blackout_start_date"] = pytz.UTC.localize(
        parse_date(metadata["blackout_start_date"])
    )
    metadata["blackout_end_date"] = pytz.UTC.localize(
        parse_date(metadata["blackout_end_date"])
    )

    return meter_data, temperature_data, metadata
