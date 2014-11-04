import argparse
import glob
import os
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()

    filenames = glob.glob(os.path.join(args.data_dir,"*"))

    months = [datetime.datetime(2000,month,1).strftime("%b") for month in range(1,13)]
    data = []
    for filename in filenames[:10]:
        with open(filename,'r') as f:
            lines = f.readlines()
        if lines == []:
            continue
        station_name = lines[0].split(":")[-1].strip()
        station_id = lines[1].split(":")[-1].strip()
        latitude = lines[2].split(":")[-1].strip()
        longitude = lines[3].split(":")[-1].strip()
        elevation = lines[4].split(":")[-1].strip()

        next_heading = lines[6].strip()
        if next_heading == "Temperature-Related Normals":
            print "TRN"
            mly_tmax_normal = {month: datum for month,datum in zip(months,lines[10].strip().split()[1:])}
            print lines[11]
            mly_tmax_normal = {month: datum for month,datum in zip(months,lines[11].strip().split()[1:])}
            print mly_tmax_normal
        elif next_heading == "Temperature-Related Pseudonormals":
            print "TRPN"
        elif next_heading == "Precipitation-Related Normals":
            print "PRN"
        elif next_heading == "Precipitation-Related Pseudonormals":
            print "PRPN"
