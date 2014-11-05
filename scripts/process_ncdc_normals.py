import argparse
import glob
import os
import datetime

def find_headings(lines):
    trn,trpn,prn,prpn = None,None,None,None
    for i,line in enumerate(lines):
        if "Temperature-Related Normals" == line:
            trn = i
        if "Temperature-Related Pseudonormals" == line:
            trpn = i
        if "Precipitation-Related Normals" == line:
            prn = i
        if "Precipitation-Related Pseudonormals" == line:
            prpn = i
    return trn,trpn,prn,prpn

def get_monthly(lines,start_index,n):
    months = [datetime.datetime(2000,month,1).strftime("%b") for month in range(1,13)]

    def get_line(line):
        return {month: datum for month,datum in zip(months,line.split()[1:])}

    return {lines[start_index + i].split()[0]: get_line(lines[start_index + i]) for i in range(2,2 + n)}

def get_daily(lines,start_index,n,offset=2,verbose=False):
    months = [datetime.datetime(2000,month,1).strftime("%b") for month in range(1,13)]
    days = [datetime.datetime(2000,1,day).strftime("%d") for day in range(1,32)]

    def get_block(lines):
        if verbose:
            print lines
        lines[0] = " ".join(lines[0].split()[1:])
        return { month:{ day:datum for day,datum in zip(days,line.split()[1:])}
                for month,line in zip(months,lines)}

    return {lines[start_index + i].split()[0]: get_block(lines[start_index + i:start_index + i + 12])
            for i in range(offset, 15*n + 3,15)}

def get_yearly(lines,start_index,n):
    def get_line(line):
        return line.split()[-1]
    return {lines[start_index + i].split()[0]: get_line(lines[start_index + i]) for i in range(n)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    filenames = glob.glob(os.path.join(args.data_dir,"*"))

    months = [datetime.datetime(2000,month,1).strftime("%b") for month in range(1,13)]
    days = [datetime.datetime(2000,1,day).strftime("%d") for day in range(1,32)]
    data = []
    for filename in filenames[:10]:
        with open(filename,'r') as f:
            lines = [line.strip() for line in f.readlines()]
        if lines == []:
            continue
        station_name = lines[0].split(":")[-1]
        station_id = lines[1].split(":")[-1]
        latitude = lines[2].split(":")[-1]
        longitude = lines[3].split(":")[-1]
        elevation = lines[4].split(":")[-1]

        trn,trpn,prn,prpn = find_headings(lines)
        trn_data, trpn_data,prn_data,prpn_data = None,None,None,None
        if trn is not None:
            trn_data = {"monthly":get_monthly(lines,trn + 2,39),
                        "daily":get_daily(lines,trn + 45,22),
                        "ann":get_yearly(lines,trn + 392,35),
                        "djf":get_yearly(lines,trn + 427,35),
                        "mam":get_yearly(lines,trn + 462,35),
                        "jja":get_yearly(lines,trn + 497,35),
                        "son":get_yearly(lines,trn + 532,35)}

        if trpn is not None:
            trpn_data = {"monthly":get_monthly(lines,trpn + 2,35),
                         "daily":get_daily(lines,trpn + 40,18,offset=3),
                         "ann":get_yearly(lines,trpn + 328,35),
                         "djf":get_yearly(lines,trpn + 363,35),
                         "mam":get_yearly(lines,trpn + 398,35),
                         "jja":get_yearly(lines,trpn + 433,35),
                         "son":get_yearly(lines,trpn + 468,35),
                         }

        if prn is not None:
            prn_data = {"monthly": get_monthly(lines,prn + 2,21),
                        "daily": get_daily(lines,prn + 26,26,offset=3),
                        "ann": get_yearly(lines,prn + 419,15),
                        "djf": get_yearly(lines,prn + 434,15),
                        "mam": get_yearly(lines,prn + 449,15),
                        "jja": get_yearly(lines,prn + 464,15),
                        "son": get_yearly(lines,prn + 479,15)}

        if prpn is not None:
            prpn_data = {"monthly": get_monthly(lines,prpn + 2,1),
                         "daily": get_daily(lines,prpn + 6,2,offset=3),
                         "ann": get_yearly(lines,prpn + 39,1),
                         "djf": get_yearly(lines,prpn + 40,1),
                         "mam": get_yearly(lines,prpn + 41,1),
                         "jja": get_yearly(lines,prpn + 42,1),
                         "son": get_yearly(lines,prpn + 43,1),
                         }

        data.append({"Temperature-Related Normals":trn_data,
                     "Temperature-Related Pseudonormals":trpn_data,
                     "Precipitation-Related Normals":prn_data,
                     "Precipitation-Related Pseudonormals":prpn_data})
