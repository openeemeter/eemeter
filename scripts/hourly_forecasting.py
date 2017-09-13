"""
This is just sample script to demonstrate builing HourlyModel
"""
import pandas as pd
import numpy as np
import csv, os, datetime, dateutil
import eemeter

from eemeter.structures import ZIPCodeSite
from eemeter.modeling.models import DayOfWeekBasedLinearRegression
from eemeter.structures import EnergyTrace
from eemeter.modeling.formatters import ModelDataFormatter
import pytz
from eemeter.weather.location import (
    zipcode_to_usaf_station,
    zipcode_to_tmy3_station,
)
from eemeter.weather.noaa import ISDWeatherSource

from pandas.tseries.offsets import DateOffset
import json


def load_projects(projects_file):
    projects = {}
    with open(projects_file, 'r') as f:
        fcsv = csv.reader(f)
        fcsv.next()
        for row in fcsv:
            projects[row[0]] = {'project_id': row[0],
                                'zipcode': row[1],
                                'baseline_period_end': row[2],
                                'reporting_period_start': row[3]}

    return projects


site_id_to_trace = json.load(open("/vagrant/etl-natgrid-lime/test_data/maps/site_to_circuit.json"))
trace_to_site = {vv : kk for kk, vv in site_id_to_trace.items()}

def get_trace_file(base_dir, trace_id):
    site_id = trace_to_site[trace_id]
    file = os.path.join(base_dir, "new_" +site_id + ".csv")
    return file



def reindex(df):
    """

    Parameters
    ----------
    df

    Returns
    -------

    """
    new_df = df.resample("H")
    last_index = df.index[0]
    last_value = None
    df_list = []
    for index, row in df.iterrows():
        if index == last_index:
            last_value = row['energy']
            continue
        num_hours = (index - last_index).days * 24.0
        mean = last_value / num_hours
        range = pd.date_range(start=last_index, end=index, freq='H')
        temp_df = pd.DataFrame({
            'energy' : [mean for xx in range]
        }, index=range)
        df_list.append(temp_df)
        last_index = index
        last_value = row['energy']
    new_df = pd.concat(df_list)
    return new_df[~new_df.index.duplicated(keep='first')]

def build_model(trace_file,
                trace_id_to_project,
                projects):
    data = pd.read_csv(trace_file, index_col=3, parse_dates=True)
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data = data.rename(columns={'value' : 'energy', 'start' : 'index'})

    data = data.tz_localize('UTC', level=0).sort_index()

    trace_id = data['trace_id'].tolist()[0]
    if "_" in trace_id:
        trace_id = trace_id.split("_")[0]
    project_id = trace_id_to_project.get(trace_id, '')

    if not project_id:
        print 'Could not find project id for trace id ' + trace_id
        return

    zipcode = projects[project_id]['zipcode']
    station = zipcode_to_usaf_station(zipcode)

    data = data.drop(['estimated', 'interpretation', 'unit', 'trace_id'], axis=1)
    weather_source = ISDWeatherSource(station)
    data = data[~data.index.duplicated(keep='first')]
    data_idx = pd.DataFrame(index=data.index)
    idx_hr = data_idx.asfreq('H', method='ffill', fill_value=np.NaN)
    tempF = weather_source.indexed_temperatures(idx_hr.index,
                                                         'degF',allow_mixed_frequency=True)

    data = data.assign(tempF=tempF)
    model = DayOfWeekBasedLinearRegression()
    model.fit(data)
    return trace_id, project_id, data, model, zipcode

def forecast_next_x_days(start_time_stamp, model, xdays, zipcode):
    series = pd.date_range(start_time_stamp, periods= xdays * 24.0, freq='H')
    dummy_df = pd.DataFrame({'dummy_col' : [np.NaN for xx in series]} , index=series)
    station = zipcode_to_usaf_station(zipcode)
    weather_source = ISDWeatherSource(station)
    tempF = weather_source.indexed_temperatures(dummy_df.index,
                                                'degF',allow_mixed_frequency=True)

    dummy_df = dummy_df.assign(tempF=tempF)
    dummy_df = dummy_df.drop(['dummy_col'], axis=1)
    return model.predict(dummy_df)
if __name__ == '__main__':
  projects_file = '/vagrant/etl-natgrid-lime/test_data/projects/Lime_Metering_Extract_070617.csv'
  projects = load_projects(projects_file)
  ciruit_to_project_fname = '/vagrant/etl-natgrid-lime/test_data/maps/circuit_to_proj.json'
  trace_id_to_project = json.load(open(ciruit_to_project_fname))

  trace_dir = "/vagrant/etl-natgrid-lime/test_data/traces"
  list_out_dfs = []
  for file in os.listdir(trace_dir):
      if not file.startswith("new_"):
          continue

      full_path = os.path.join(trace_dir, file)

      model_out = build_model(full_path,
                              trace_id_to_project,
                              projects)

      if model_out is None:
          print "Insufficent data in trace file ", full_path
          continue

      trace_id, project_id, data, model, zipcode = model_out
      start_time = data.index[-1] + pd.DateOffset(hours=1)
      energy_forecast = forecast_next_x_days(start_time, model, 10, zipcode)
      out_df = pd.DataFrame({
          "project_id" : [project_id for xx in energy_forecast.index],
          "trace_id" :  [trace_id for xx in energy_forecast.index],
          "energy_forecast" : energy_forecast['energy_forecast']
      }, index=energy_forecast.index)
      list_out_dfs.append(out_df)

  result_df = pd.concat(list_out_dfs)
  result_df.to_csv('/vagrant/natgrid/forecast.csv'
                   , index_label='date')
