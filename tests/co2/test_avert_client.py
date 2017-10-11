from eemeter.co2.clients import AVERTClient
import pandas as pd


def test_rdf_data():
    client = AVERTClient()
    co2_by_load, load_by_hour = client.read_rdf_file(2016, 'UMW')
    assert load_by_hour.dropna().shape == (366*24,)
    assert co2_by_load.dropna().shape[0] > 1
