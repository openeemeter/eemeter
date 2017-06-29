from click.testing import CliRunner
from eemeter import cli

def test_cli():
	runner = CliRunner()
	result = runner.invoke(cli.sample)
	assert result.exit_code == 0
