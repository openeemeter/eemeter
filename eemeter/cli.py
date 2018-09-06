from pkg_resources import resource_stream
import json

import click

from eemeter import (
    caltrack_method,
    meter_data_from_csv,
    temperature_data_from_csv,
    merge_temperature_data,
)


@click.group()
def cli():
    """Example usage


Save output::


    Use CalTRACK methods on sample data:

    \b
        $ eemeter caltrack --sample=il-electricity-cdd-hdd-daily

    Save output:

    \b
        $ eemeter caltrack --sample=il-electricity-cdd-only-billing_monthly --output-file=/path/to/output.json

    Load custom data (see sample files for example format):

    \b
        $ eemeter caltrack --meter-file=/path/to/meter/data.csv --temperature-file=/path/to/temperature/data.csv

    Do not fit CDD models (intended for gas data):

    \b
        $ eemeter caltrack --sample=il-gas-hdd-only-billing_monthly --no-fit-cdd
    """
    pass  # pragma: no cover


def _get_data(
    sample, meter_file, temperature_file, heating_balance_points, cooling_balance_points
):

    if sample is not None:
        with resource_stream("eemeter.samples", "metadata.json") as f:
            metadata = json.loads(f.read().decode("utf-8"))
        if sample in metadata:
            click.echo("Loading sample: {}".format(sample))

            meter_file = resource_stream(
                "eemeter.samples", metadata[sample]["meter_data_filename"]
            )
            temperature_file = resource_stream(
                "eemeter.samples", metadata[sample]["temperature_filename"]
            )
        else:
            raise click.ClickException(
                "Sample not found. Try one of these?\n{}".format(
                    "\n".join([" - {}".format(key) for key in sorted(metadata.keys())])
                )
            )

    if meter_file is not None:
        gzipped = meter_file.name.endswith(".gz")
        meter_data = meter_data_from_csv(meter_file, gzipped=gzipped)
    else:
        raise click.ClickException("Meter data not specified.")

    if temperature_file is not None:
        gzipped = temperature_file.name.endswith(".gz")
        temperature_data = temperature_data_from_csv(
            temperature_file, gzipped=gzipped, freq="hourly"
        )
    else:
        raise click.ClickException("Temperature data not specified.")

    return merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=heating_balance_points,
        cooling_balance_points=cooling_balance_points,
    )


@cli.command()
@click.option("--sample", default=None, type=str)
@click.option("--meter-file", default=None, type=click.File("rb"))
@click.option("--temperature-file", default=None, type=click.File("rb"))
@click.option("--output-file", default=None, type=click.File("wb"))
@click.option("--show-candidates/--no-show-candidates", default=False, is_flag=True)
@click.option("--fit-cdd/--no-fit-cdd", default=True, is_flag=True)
def caltrack(
    sample, meter_file, temperature_file, output_file, show_candidates, fit_cdd
):
    # TODO(philngo): add project dates and baseline period/reporting period.
    # TODO(philngo): complain about temperature data that isn't daily

    heating_balance_points = range(55, 66)
    cooling_balance_points = range(65, 76)

    data = _get_data(
        sample,
        meter_file,
        temperature_file,
        heating_balance_points,
        cooling_balance_points,
    )
    model_results = caltrack_method(data, fit_cdd=fit_cdd)
    json_str = json.dumps(model_results.json(with_candidates=show_candidates), indent=2)

    if output_file is None:
        click.echo(json_str)
    else:
        output_file.write(json_str.encode("utf-8"))
        click.echo("Output written: {}".format(output_file.name))
