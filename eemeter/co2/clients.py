from io import BytesIO
import logging
from datetime import datetime

import pandas as pd

import requests
import zipfile
import xlrd

logger = logging.getLogger(__name__)

AVERT_NAMES = {
    'CA': ['California', 'california'],
    'EMW': ['Great Lakes - Mid-Atlantic', 'great_lakes_-_mid-atlantic'],
    'LMW': ['Lower Midwest', 'lower_midwest'],
    'NE': ['Northeast', 'northeast'],
    'NW': ['Northwest', 'northwest'],
    'RM': ['Rocky Mountains', 'rocky_mountains'],
    'SE': ['Southeast', 'southeast'],
    'SW': ['Southwest', 'southwest'],
    'TX': ['Texas', 'texas'],
    'UMW': ['Upper Midwest', 'upper_midwest']
}


class AVERTClient(object):

    def __init__(self, n_tries=3):
        self.n_tries = n_tries

    def _retrieve_from_zipfile(self, year, region):
        if (int(year) < 2007) or (int(year) > 2015):
            return None
        if region not in AVERT_NAMES.keys():
            return None

        url = 'https://www.epa.gov/sites/production/files/2017-07/' +\
              'avert_regional_data_files_{}_-_07-31-17.zip'.format(year)
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(BytesIO(r.content))
        fname_to_extract = None
        for fname in z.namelist():
            for name in AVERT_NAMES[region]:
                if fname.find(name) != -1:
                    fname_to_extract = fname

        return z.read(fname_to_extract)

    def _retrieve_file(self, year, region):
        if int(year) != 2016:
            return None
        if region not in AVERT_NAMES.keys():
            return None

        region_lcase = AVERT_NAMES[region][1]
        url = 'https://www.epa.gov/sites/production/files/2017-07/' +\
              'avert_rdf_2016_epa_netgen_pm25_{}.xlsx'.format(region_lcase)
        r = requests.get(url, stream=True)
        return r.content

    def read_rdf_file(self, year, region):
        '''Read in an XLS file produced by the DOE AVERT
        team to calculate carbon avoidance'''

        if year == 2016:
            streamdata = self._retrieve_file(year, region)
        else:
            streamdata = self._retrieve_from_zipfile(year, region)

        if not streamdata:
            # return empty series
            logging.error("Could not find weather data for Year: " +
                          str(year) + " and Region " + str(region))
            return pd.Series(), pd.Series()
        # Open the workbook
        wb = xlrd.open_workbook(file_contents=streamdata)

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
            ts = datetime(year=yr, month=mo, day=da, hour=hr)
            timestamp.append(ts)
            regional_load.append(rl)
            idx = idx + 1

        # Convert it to a series
        load_by_hour = pd.Series(regional_load, index=timestamp)

        # And return the two series.
        return co2_by_load, load_by_hour
