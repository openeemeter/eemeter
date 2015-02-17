#!/usr/bin/python

# Usage generate_portfolio [OUTDIR]
# This script will generate projects projects, customers and consumption data
# And output corresponding csv files in the specified directory

from eemeter.generator import ProjectGenerator
from eemeter.generator import generate_periods
from eemeter.weather import GSODWeatherSource
from eemeter.models import DoubleBalancePointModel
from eemeter.models import PRISMModel

from datetime import datetime
from datetime import timedelta

from scipy.stats import uniform
import pandas as pd

import os
import uuid
import random
import argparse

from itertools import chain, repeat


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate a portfolio of projects.')
    parser.add_argument('n_projects', type=int, help='number of projects in portfolio.')
    parser.add_argument('start_date', type=str, help='start date of earliest consumption period (YYYY-MM-DD).')
    parser.add_argument('outdir', type=str, help='directory in which to write CSV output.')
    parser.add_argument('weather_station', type=str, help='weather station from which to pull temperature data. (e.g. 725300)')
    args = parser.parse_args()

    electricity_model = DoubleBalancePointModel()
    gas_model = PRISMModel()

    electricity_param_distributions = (
            uniform(loc=1, scale=.5),
            uniform(loc=1, scale=.5),
            uniform(loc=5, scale=5),
            uniform(loc=62, scale=5),
            uniform(loc=2, scale=5))
    electricity_param_delta_distributions = (
            uniform(loc=-.2,scale=.3),
            uniform(loc=-.2, scale=.3),
            uniform(loc=-2, scale=3),
            uniform(loc=0, scale=0),
            uniform(loc=0, scale=0))
    gas_param_distributions = (
            uniform(loc=62, scale=3),
            uniform(loc=5, scale=5),
            uniform(loc=1, scale=.5))
    gas_param_delta_distributions = (
            uniform(loc=0, scale=0),
            uniform(loc=-2, scale=3),
            uniform(loc=-.2,scale=.3))

    generator = ProjectGenerator(electricity_model, gas_model,
                                 electricity_param_distributions,electricity_param_delta_distributions,
                                 gas_param_distributions,gas_param_delta_distributions)

    start_date = datetime.strptime(args.start_date,"%Y-%m-%d")
    n_days = (datetime.now() - start_date).days
    if n_days < 30:
        message = "start_date ({}) must be at least 30 days before today".format(start_date)
        raise ValueError(message)

    weather_source = GSODWeatherSource(args.weather_station,start_date.year,datetime.now().year)

    project_data = []
    consumption_data = []
    for _ in range(args.n_projects):
        elec_periods = generate_periods(start_date,datetime.now())
        gas_periods = generate_periods(start_date,datetime.now())

        retrofit_start_date = start_date + timedelta(days=random.randint(0,n_days-30))
        retrofit_completion_date = retrofit_start_date + timedelta(days=30)

        electricity_noise = None
        gas_noise = None

        elec_consumption, gas_consumption = generator.generate(weather_source, elec_periods, gas_periods,
                                                           retrofit_start_date, retrofit_completion_date,
                                                           electricity_noise,gas_noise)
        custom_id = uuid.uuid4()

        p_data = {
            "id": custom_id,
            "retrofit_start_date": retrofit_start_date,
            "retrofit_completion_date": retrofit_completion_date,
            }
        project_data.append(p_data)

        for c,unit_name in chain(zip(elec_consumption,repeat("kWh")),zip(gas_consumption,repeat("therm"))):
            c_data = {
                    "project_id": custom_id,
                    "fuel_type": c.fuel_type,
                    "unit_name": unit_name,
                    "quantity": c.to(unit_name),
                    "start_date": c.start,
                    "end_date": c.end,
                    }
            consumption_data.append(c_data)

    # write projects csv
    projects = pd.DataFrame({
        'id': [p["id"] for p in project_data],
        'retrofit_start_date': [p["retrofit_start_date"] for p in project_data],
        'retrofit_completion_date': [p["retrofit_completion_date"] for p in project_data],
        'weather_station': [args.weather_station for _ in project_data],
        })
    projects.to_csv(os.path.join(args.outdir, 'projects.csv'),index=False)

    # write consumptions.csv
    consumptions = pd.DataFrame(consumption_data)
    consumptions.to_csv(os.path.join(args.outdir, 'consumptions.csv'),index=False)
