from click.testing import CliRunner
from eemeter import cli

def test_cli():
    runner = CliRunner()
    result = runner.invoke(cli.sample)
    assert result.exit_code == 0

    
def test_trace_builder():
    path = cli._get_sample_inputs_path()
    projects, trace_objects = cli._load_projects_and_traces(path)
    assert len(trace_objects[0].data) == 2*24*365 + 1

    
def test_cli_analyze_returns_meter_output_with_derivatives():
    path = cli._get_sample_inputs_path()
    retval = cli._analyze(path)
    series = [i['series'] for i in retval[0]['derivatives']]
    assert "Baseline model, reporting period" in series
    assert retval[0]
