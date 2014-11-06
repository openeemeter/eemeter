##################
#
#   Or just use:
#
#       wget -r ftp://ftp.ncdc.noaa.gov/pub/data/normals/1981-2010/products/station/
#
##################


from ftplib import FTP
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("start_at")
    args = parser.parse_args()

    ftp = FTP('ftp.ncdc.noaa.gov')
    ftp.login()
    ftp.cwd('/pub/data/normals/1981-2010/products/station')

    filenames = []
    def callback(response):
        filename = response.split()[-1]
        filenames.append(filename)
    ftp.retrlines('LIST',callback)

    started = False
    for filename in filenames:
        if not started and filename == args.start_at:
            started = True
        if started:
            print filename
            ftp.retrbinary('RETR {}'.format(filename), open(os.path.join(args.data_dir,filename), 'wb').write)
    ftp.quit()
