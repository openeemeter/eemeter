#!/usr/bin/python

# Usage generate_portfolio [OUTDIR]
# This script will generate projects projects, customers and consumption data
# And output corresponding csv files in the specified directory

from eemeter import generator
from datetime import datetime
from eemeter.consumption import DatetimePeriod
from eemeter.weather import TMY3WeatherSource

import os
import itertools
from scipy.stats import uniform
import pandas as pd
import sys

outdir = sys.argv[1]

# initialize periods, projects, portolio
periods1 = [DatetimePeriod(datetime(2011,i,1), datetime(2011,i+1,1)) for i in range(1,11)] # monthly 2011
periods2 = [DatetimePeriod(datetime(2012,i,1), datetime(2012,i+1,1)) for i in range(1,11)] # monthly 2012

project1 = generator.ProjectGenerator("Pasadena", 5, "electricity", "J", "degF", 
                           uniform(loc=60, scale=10), uniform(loc=.5,scale=1), 
                           uniform(loc=75, scale=10), uniform(loc=.5, scale=1),
                           uniform(loc=5, scale=5))

project2 = generator.ProjectGenerator("Chicago", 10, "gas", "J", "degF", 
                           uniform(loc=60, scale=5), uniform(loc=.5,scale=1), 
                           uniform(loc=0, scale=0), uniform(loc=0, scale=0),
                           uniform(loc=5, scale=5))

portfolio = generator.PortfolioGenerator("My Project", [project1, project2])

# write projects csv
projects = pd.DataFrame({'name': [p.name for p in portfolio.projects],
              'fuel_type': [p.fuel_type for p in portfolio.projects],
              'consumption_unit_name': [p.consumption_unit_name for p in portfolio.projects]})
projects.index.name = 'id'

projects.to_csv(os.path.join(outdir, 'projects.csv'))

# write customers csv
project_customer_ids = []
project_ids = []

for i in range(len(portfolio.projects)):
    for j in range(portfolio.projects[i].n_homes):
        project_ids.append(i)
        project_customer_ids.append(j)
    
customers = pd.DataFrame({'project_customer_id': project_customer_ids, 'project_id' : project_ids})
customers.index.name = 'id'

customers.to_csv(os.path.join(outdir, 'customers.csv'))

# initialize weather
pasadena = TMY3WeatherSource('722880',os.environ.get("TMY3_DIRECTORY"))
chicago = TMY3WeatherSource('725300',os.environ.get("TMY3_DIRECTORY"))

consumptions = portfolio.generate([pasadena, chicago], [periods1, periods2], [None, None])

c = list(itertools.chain.from_iterable([[id, c.start, c.end, c.to("J")] for c in consumptions[id]] for id in range(len(consumptions))))

consumption = pd.DataFrame(c, columns=['customer_id', 'start', 'end', 'joules'])
consumption.index.name = 'id'
consumption.to_csv(os.path.join(outdir, 'consumption.csv'))