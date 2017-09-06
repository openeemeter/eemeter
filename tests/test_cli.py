from click.testing import CliRunner
from eemeter import cli

def test_cli():
	runner = CliRunner()
	result = runner.invoke(cli.sample)
	assert result.exit_code == 0

def test_cli_analyze_returns_meter_output_with_derivatives():
        retval = cli._analyze('eemeter/sample_data')
        series = [i['series'] for i in retval[0]['derivatives']]
        for i in series: print i
        assert "Baseline model, reporting period" in series
