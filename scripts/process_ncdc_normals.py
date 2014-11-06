import argparse
import glob
import os
import datetime
import json

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

def get_monthly(lines,start_index):
    months = [datetime.datetime(2000,month,1).strftime("%b") for month in range(1,13)]

    def get_line(line):
        return {month: datum for month,datum in zip(months,line[1:])}

    data = {}
    index = start_index + 2

    while lines[index].strip() is not "":
        line = lines[index].split()
        data[line[0]] = get_line(line)
        index += 1
    return data, index + 1

def get_daily(lines,start_index,verbose=False):
    months = [datetime.datetime(2000,month,1).strftime("%b") for month in range(1,13)]
    days = [datetime.datetime(2000,1,day).strftime("%d") for day in range(1,32)]

    def get_block(lines):
        if verbose:
            print lines
        lines[0] = " ".join(lines[0].split()[1:])
        return { month:{ day:datum for day,datum in zip(days,line.split()[1:])}
                for month,line in zip(months,lines)}

    data = {}
    index = start_index + 3

    while len(lines[index].split()) > 30:
        data[lines[index].split()[0]] = get_block(lines[index:index+12])
        index += 15
    return data, index + 1

def get_yearly(lines,start_index):
    def get_line(line):
        return line.split()[-1]

    data = {}
    index = start_index + 0
    while True:
        try:
            if lines[index].strip() is "":
                break
        except:
            break
        data[lines[index].split()[0]] = get_line(lines[index])
        index += 1
    return data, index + 1

def get_data(lines,index):
    monthly,index = get_monthly(lines,index)
    daily,index = get_daily(lines,index)
    yearly,index = get_yearly(lines,index)
    return {"monthly": monthly,"daily": daily,"yearly": yearly}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    filenames = glob.glob(os.path.join(args.data_dir,"*"))

    for filename in filenames:
        print filename.split("/")[-1]
        with open(filename,'r') as f:
            lines = [line.strip() for line in f.readlines()]

        station_name = lines[0].split(":")[-1]
        station_id = lines[1].split(":")[-1]
        latitude = lines[2].split(":")[-1]
        longitude = lines[3].split(":")[-1]
        elevation = lines[4].split(":")[-1]

        trn,trpn,prn,prpn = find_headings(lines)

        trn_data, trpn_data, prn_data, prpn_data = None,None,None,None

        if trn is not None:
            trn_data = get_data(lines,trn + 2)
        if trpn is not None:
            trpn_data = get_data(lines,trpn + 2)
        if prn is not None:
            prn_data = get_data(lines,prn + 2)
        if prpn is not None:
            prpn_data = get_data(lines,prpn + 2)

        data = {"station_name": station_name,
                 "station_id": station_id,
                 "latitude": latitude,
                 "longitude": longitude,
                 "elevation": elevation,
                 "Temperature-Related Normals": trn_data,
                 "Temperature-Related Pseudonormals": trpn_data,
                 "Precipitation-Related Normals": prn_data,
                 "Precipitation-Related Pseudonormals": prpn_data}

        target_filename = os.path.join(args.out_dir,filename.split("/")[-1].split(".")[0] + ".normals.json")
        with open(target_filename,'w') as f:
            f.write(json.dumps(data))
