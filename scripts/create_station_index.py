import pandas as pd
import json
from collections import defaultdict
import ftplib
from io import BytesIO

if __name__ == "__main__":

    ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
    ftp.login()
    string = BytesIO()
    ftp.retrbinary('RETR /pub/data/noaa/isd-inventory.csv', string.write)
    ftp.quit()
    string.seek(0)

    df = pd.read_csv(string,
                     dtype={"USAF": str, "WBAN": str},
                     usecols=["USAF", "WBAN"])

    index = defaultdict(set)
    for usaf, full in zip(df.USAF, df.USAF + '-' + df.WBAN):
        if usaf is not '999999':
            index[usaf].add(full)
    index = {k: list(v) for k, v in index.items()}

    with open('GSOD-ISD_station_index.json', 'w') as f:
        json.dump(index, f)
