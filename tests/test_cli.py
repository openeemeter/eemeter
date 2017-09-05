from click.testing import CliRunner
from eemeter import cli

def test_cli_sample():
	runner = CliRunner()
	result = runner.invoke(cli.sample, obj={})
	assert result.exit_code == 0


def test_cli_analyze_returns_meter_results():
        result = cli._analyze("eemeter/sample_data", None, None)
        assert result is not None
