import xlrd
import datetime
import pandas as pd
from scipy import interpolate

# This code is based on data and algorithms provided by
# the Environmental Protection Agency, AVERT, and Synapse Energy.
# https://www.epa.gov/statelocalenergy/avoided-emissions-and-generation-tool-avert


def read_rdf_file(fname):
    '''Read in an XLS file produced by the DOE AVERT
    team to calculate carbon avoidance'''

    # Open the workbook
    wb = xlrd.open_workbook(fname)

    # Find the sheet with the actual data in it
    maxsheet, maxcol = -1, -1
    for idx in range(wb.nsheets):
        if wb.sheet_by_index(idx).ncols > maxcol:
            maxsheet, maxcol = idx, wb.sheet_by_index(idx).ncols
    sheet = wb.sheet_by_index(maxsheet)

    # Find the number of load bins
    nbins = maxcol - 15

    # Grab the load bins, and sum up the CO2 emissions per bin
    load_bins = sheet.row_values(1)[15:15+nbins]
    co2_sums = []
    for col in range(15, 15+nbins):
        this_co2_sum = sum([i for i in sheet.col_values(col)[10004:11999]
                            if i != ''])
        co2_sums.append(this_co2_sum)

    # Make the CO2 per bin into a series
    co2_by_load = pd.Series(co2_sums, index=load_bins)

    # Now read in the regional load by hour
    idx = 3
    timestamp, regional_load = [], []
    yrs = sheet.col_values(1)
    mos = sheet.col_values(2)
    das = sheet.col_values(3)
    hrs = sheet.col_values(4)
    rls = sheet.col_values(5)
    while True:
        if yrs[idx] == '':
            break
        yr = int(yrs[idx])
        mo = int(mos[idx])
        da = int(das[idx])
        hr = int(hrs[idx])
        rl = rls[idx]
        ts = datetime.datetime(year=yr, month=mo, day=da, hour=hr)
        timestamp.append(ts)
        regional_load.append(rl)
        idx = idx + 1

    # Convert it to a series
    load_by_hour = pd.Series(regional_load, index=timestamp)

    # And return the two series.
    return co2_by_load, load_by_hour


def calc_avoided_co2(resource_curve, co2_by_load, load_by_hour):
    '''Based on a resource curve and the data in an AVERT-produced RDF
    file, compute avoided CO2 emissions, in tons of CO2.'''

    # Calculate the pre-intervention CO2 emissions
    f = interpolate.interp1d(co2_by_load.index, co2_by_load.values)
    co2_pre = f(load_by_hour.values)

    # Calculate the post-internention load and CO2 emissions
    load_post = load_by_hour - resource_curve
    co2_post = f(load_post.values)

    # Return the savings
    return co2_pre - co2_post
