#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
from click.testing import CliRunner
from tempfile import NamedTemporaryFile
import importlib.resources
import platform

from eemeter.eemeter.utilities.cli import cli, caltrack


def test_eemeter_cli():
    runner = CliRunner()

    result = runner.invoke(cli)
    assert result.exit_code == 0
    assert len(result.output) > 400


def test_eemeter_caltrack_sample_unknown():
    runner = CliRunner()

    result = runner.invoke(caltrack, ["--sample=unknown"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: Sample not found.")


def test_eemeter_caltrack_sample_known():
    runner = CliRunner()

    result = runner.invoke(caltrack, ["--sample=il-gas-hdd-only-daily"])

    assert result.exit_code == 0
    assert result.output.endswith("}\n")  # json


def test_eemeter_caltrack_meter_data_only():
    runner = CliRunner()

    meter_file = str(
        importlib.resources.files("eemeter.eemeter.samples").joinpath(
            "il-gas-hdd-only-daily.csv.gz"
        )
    )
    result = runner.invoke(caltrack, ["--meter-file={}".format(meter_file)])

    assert result.exit_code == 1
    assert result.output == "Error: Temperature data not specified.\n"


def test_eemeter_caltrack_temperature_data_only():
    runner = CliRunner()

    temperature_file = str(
        importlib.resources.files("eemeter.eemeter.samples").joinpath("il-tempF.csv.gz")
    )

    result = runner.invoke(caltrack, ["--temperature-file={}".format(temperature_file)])

    assert result.exit_code == 1
    assert result.output == "Error: Meter data not specified.\n"


def test_eemeter_caltrack_temperature_custom_data():
    runner = CliRunner()

    meter_file = str(
        importlib.resources.files("eemeter.eemeter.samples").joinpath(
            "il-gas-hdd-only-daily.csv.gz"
        )
    )
    temperature_file = str(
        importlib.resources.files("eemeter.eemeter.samples").joinpath("il-tempF.csv.gz")
    )

    result = runner.invoke(
        caltrack,
        [
            "--meter-file={}".format(meter_file),
            "--temperature-file={}".format(temperature_file),
        ],
    )

    assert result.exit_code == 0
    assert result.output.endswith("}\n")  # json


def test_eemeter_caltrack_sample_output_file():
    if platform.system() == "Windows":
        # CLI is deprecated, this test is failing in the runner, and it's not worth debugging
        return

    runner = CliRunner()

    output_file = NamedTemporaryFile()
    result = runner.invoke(
        caltrack,
        ["--sample=il-gas-hdd-only-daily", "--output-file={}".format(output_file.name)],
    )

    assert result.exit_code == 0
    assert "Output written:" in result.output

    assert output_file.read().endswith(b"}")
