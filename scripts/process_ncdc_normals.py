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
    days = [datetime.datetime(2000,1,day).strftime("%d") for day in range(1,32)]
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
            mly_tavg_normal = {month: datum for month,datum in zip(months,lines[11].strip().split()[1:])}
            mly_tmin_normal = {month: datum for month,datum in zip(months,lines[12].strip().split()[1:])}
            mly_dutr_normal = {month: datum for month,datum in zip(months,lines[13].strip().split()[1:])}
            mly_cldd_normal = {month: datum for month,datum in zip(months,lines[14].strip().split()[1:])}
            mly_htdd_normal = {month: datum for month,datum in zip(months,lines[15].strip().split()[1:])}

            mly_tmax_stddev = {month: datum for month,datum in zip(months,lines[16].strip().split()[1:])}
            mly_tavg_stddev = {month: datum for month,datum in zip(months,lines[17].strip().split()[1:])}
            mly_tmin_stddev = {month: datum for month,datum in zip(months,lines[18].strip().split()[1:])}
            mly_dutr_stddev = {month: datum for month,datum in zip(months,lines[19].strip().split()[1:])}

            mly_cldd_base45 = {month: datum for month,datum in zip(months,lines[20].strip().split()[1:])}
            mly_cldd_base50 = {month: datum for month,datum in zip(months,lines[21].strip().split()[1:])}
            mly_cldd_base55 = {month: datum for month,datum in zip(months,lines[22].strip().split()[1:])}
            mly_cldd_base57 = {month: datum for month,datum in zip(months,lines[23].strip().split()[1:])}
            mly_cldd_base60 = {month: datum for month,datum in zip(months,lines[24].strip().split()[1:])}
            mly_cldd_base70 = {month: datum for month,datum in zip(months,lines[25].strip().split()[1:])}
            mly_cldd_base72 = {month: datum for month,datum in zip(months,lines[26].strip().split()[1:])}

            mly_htdd_base40 = {month: datum for month,datum in zip(months,lines[27].strip().split()[1:])}
            mly_htdd_base45 = {month: datum for month,datum in zip(months,lines[28].strip().split()[1:])}
            mly_htdd_base50 = {month: datum for month,datum in zip(months,lines[29].strip().split()[1:])}
            mly_htdd_base55 = {month: datum for month,datum in zip(months,lines[30].strip().split()[1:])}
            mly_htdd_base57 = {month: datum for month,datum in zip(months,lines[31].strip().split()[1:])}
            mly_htdd_base60 = {month: datum for month,datum in zip(months,lines[32].strip().split()[1:])}

            mly_tmax_avgnds_grth040 = {month: datum for month,datum in zip(months,lines[33].strip().split()[1:])}
            mly_tmax_avgnds_grth050 = {month: datum for month,datum in zip(months,lines[34].strip().split()[1:])}
            mly_tmax_avgnds_grth060 = {month: datum for month,datum in zip(months,lines[35].strip().split()[1:])}
            mly_tmax_avgnds_grth070 = {month: datum for month,datum in zip(months,lines[36].strip().split()[1:])}
            mly_tmax_avgnds_grth080 = {month: datum for month,datum in zip(months,lines[37].strip().split()[1:])}
            mly_tmax_avgnds_grth090 = {month: datum for month,datum in zip(months,lines[38].strip().split()[1:])}
            mly_tmax_avgnds_grth100 = {month: datum for month,datum in zip(months,lines[39].strip().split()[1:])}
            mly_tmax_avgnds_lsth032 = {month: datum for month,datum in zip(months,lines[40].strip().split()[1:])}

            mly_tmin_avgnds_lsth000 = {month: datum for month,datum in zip(months,lines[41].strip().split()[1:])}
            mly_tmin_avgnds_lsth010 = {month: datum for month,datum in zip(months,lines[42].strip().split()[1:])}
            mly_tmin_avgnds_lsth020 = {month: datum for month,datum in zip(months,lines[43].strip().split()[1:])}
            mly_tmin_avgnds_lsth032 = {month: datum for month,datum in zip(months,lines[44].strip().split()[1:])}
            mly_tmin_avgnds_lsth040 = {month: datum for month,datum in zip(months,lines[45].strip().split()[1:])}
            mly_tmin_avgnds_lsth050 = {month: datum for month,datum in zip(months,lines[46].strip().split()[1:])}
            mly_tmin_avgnds_lsth060 = {month: datum for month,datum in zip(months,lines[47].strip().split()[1:])}
            mly_tmin_avgnds_lsth070 = {month: datum for month,datum in zip(months,lines[48].strip().split()[1:])}

            dly_tmax_normal = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(53,65))}
            dly_tavg_normal = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(68,80))}
            dly_tmin_normal = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(83,95))}
            dly_dutr_normal = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(98,110))}
            dly_cldd_normal = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(113,125))}
            dly_htdd_normal = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(128,140))}

            dly_tmax_stddev = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(143,155))}
            dly_tavg_stddev = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(158,170))}
            dly_tmin_stddev = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(173,185))}
            dly_dutr_stddev = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(188,200))}

            dly_cldd_base45 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(203,215))}
            dly_cldd_base50 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(218,230))}
            dly_cldd_base55 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(233,245))}
            dly_cldd_base57 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(248,260))}
            dly_cldd_base60 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(263,275))}
            dly_cldd_base70 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(278,290))}
            dly_cldd_base72 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(293,305))}

            dly_htdd_base40 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(308,320))}
            dly_htdd_base45 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(323,335))}
            dly_htdd_base50 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(338,350))}
            dly_htdd_base55 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(353,365))}
            dly_htdd_base57 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(368,380))}
            dly_htdd_base60 = { month:{ day:datum for day,datum in zip(days,lines[i].strip().split()[2:])}
                    for month,i in zip(months,range(383,395))}

            ann_tmax_normal = lines[398].strip().split()[-1]
            ann_tavg_normal = lines[399].strip().split()[-1]
            ann_tmin_normal = lines[400].strip().split()[-1]
            ann_dutr_normal = lines[401].strip().split()[-1]
            ann_cldd_normal = lines[402].strip().split()[-1]
            ann_htdd_normal = lines[403].strip().split()[-1]

            ann_cldd_base45 = lines[404].strip().split()[-1]
            ann_cldd_base50 = lines[405].strip().split()[-1]
            ann_cldd_base55 = lines[406].strip().split()[-1]
            ann_cldd_base57 = lines[407].strip().split()[-1]
            ann_cldd_base60 = lines[408].strip().split()[-1]
            ann_cldd_base70 = lines[409].strip().split()[-1]
            ann_cldd_base72 = lines[410].strip().split()[-1]

            ann_htdd_base40 = lines[411].strip().split()[-1]
            ann_htdd_base45 = lines[412].strip().split()[-1]
            ann_htdd_base50 = lines[413].strip().split()[-1]
            ann_htdd_base55 = lines[414].strip().split()[-1]
            ann_htdd_base57 = lines[415].strip().split()[-1]
            ann_htdd_base60 = lines[416].strip().split()[-1]

            ann_tmax_avgnds_grth040 = lines[417].strip().split()[-1]
            ann_tmax_avgnds_grth050 = lines[418].strip().split()[-1]
            ann_tmax_avgnds_grth060 = lines[419].strip().split()[-1]
            ann_tmax_avgnds_grth070 = lines[420].strip().split()[-1]
            ann_tmax_avgnds_grth080 = lines[421].strip().split()[-1]
            ann_tmax_avgnds_grth090 = lines[422].strip().split()[-1]
            ann_tmax_avgnds_grth100 = lines[423].strip().split()[-1]
            ann_tmax_avgnds_lsth032 = lines[424].strip().split()[-1]

            ann_tmin_avgnds_lsth000 = lines[425].strip().split()[-1]
            ann_tmin_avgnds_lsth010 = lines[426].strip().split()[-1]
            ann_tmin_avgnds_lsth020 = lines[427].strip().split()[-1]
            ann_tmin_avgnds_lsth032 = lines[428].strip().split()[-1]
            ann_tmin_avgnds_lsth040 = lines[429].strip().split()[-1]
            ann_tmin_avgnds_lsth050 = lines[430].strip().split()[-1]
            ann_tmin_avgnds_lsth060 = lines[431].strip().split()[-1]
            ann_tmin_avgnds_lsth070 = lines[432].strip().split()[-1]

            djf_tmax_normal = lines[433].strip().split()[-1]
            djf_tavg_normal = lines[434].strip().split()[-1]
            djf_tmin_normal = lines[435].strip().split()[-1]
            djf_dutr_normal = lines[436].strip().split()[-1]
            djf_cldd_normal = lines[437].strip().split()[-1]
            djf_htdd_normal = lines[438].strip().split()[-1]

            djf_cldd_base45 = lines[439].strip().split()[-1]
            djf_cldd_base50 = lines[440].strip().split()[-1]
            djf_cldd_base55 = lines[441].strip().split()[-1]
            djf_cldd_base57 = lines[442].strip().split()[-1]
            djf_cldd_base60 = lines[443].strip().split()[-1]
            djf_cldd_base70 = lines[444].strip().split()[-1]
            djf_cldd_base72 = lines[445].strip().split()[-1]

            djf_cldd_base40 = lines[446].strip().split()[-1]
            djf_cldd_base45 = lines[447].strip().split()[-1]
            djf_cldd_base50 = lines[448].strip().split()[-1]
            djf_cldd_base55 = lines[449].strip().split()[-1]
            djf_cldd_base57 = lines[450].strip().split()[-1]
            djf_cldd_base60 = lines[451].strip().split()[-1]

            djf_tmax_avgnds_grth040 = lines[452].strip().split()[-1]
            djf_tmax_avgnds_grth050 = lines[453].strip().split()[-1]
            djf_tmax_avgnds_grth060 = lines[454].strip().split()[-1]
            djf_tmax_avgnds_grth070 = lines[455].strip().split()[-1]
            djf_tmax_avgnds_grth080 = lines[456].strip().split()[-1]
            djf_tmax_avgnds_grth090 = lines[457].strip().split()[-1]
            djf_tmax_avgnds_grth100 = lines[458].strip().split()[-1]
            djf_tmax_avgnds_lsth032 = lines[459].strip().split()[-1]

            djf_tmin_avgnds_lsth000 = lines[460].strip().split()[-1]
            djf_tmin_avgnds_lsth010 = lines[461].strip().split()[-1]
            djf_tmin_avgnds_lsth020 = lines[462].strip().split()[-1]
            djf_tmin_avgnds_lsth032 = lines[463].strip().split()[-1]
            djf_tmin_avgnds_lsth040 = lines[464].strip().split()[-1]
            djf_tmin_avgnds_lsth050 = lines[465].strip().split()[-1]
            djf_tmin_avgnds_lsth060 = lines[466].strip().split()[-1]
            djf_tmin_avgnds_lsth070 = lines[467].strip().split()[-1]

            mam_tmax_normal = lines[468].strip().split()[-1]
            mam_tavg_normal = lines[469].strip().split()[-1]
            mam_tmin_normal = lines[470].strip().split()[-1]
            mam_dutr_normal = lines[471].strip().split()[-1]
            mam_cldd_normal = lines[472].strip().split()[-1]
            mam_htdd_normal = lines[473].strip().split()[-1]

            mam_cldd_base45 = lines[474].strip().split()[-1]
            mam_cldd_base50 = lines[475].strip().split()[-1]
            mam_cldd_base55 = lines[476].strip().split()[-1]
            mam_cldd_base57 = lines[477].strip().split()[-1]
            mam_cldd_base60 = lines[478].strip().split()[-1]
            mam_cldd_base70 = lines[479].strip().split()[-1]
            mam_cldd_base72 = lines[480].strip().split()[-1]

            mam_cldd_base40 = lines[481].strip().split()[-1]
            mam_cldd_base45 = lines[482].strip().split()[-1]
            mam_cldd_base50 = lines[483].strip().split()[-1]
            mam_cldd_base55 = lines[484].strip().split()[-1]
            mam_cldd_base57 = lines[485].strip().split()[-1]
            mam_cldd_base60 = lines[486].strip().split()[-1]

            mam_tmax_avgnds_grth040 = lines[487].strip().split()[-1]
            mam_tmax_avgnds_grth050 = lines[488].strip().split()[-1]
            mam_tmax_avgnds_grth060 = lines[489].strip().split()[-1]
            mam_tmax_avgnds_grth070 = lines[490].strip().split()[-1]
            mam_tmax_avgnds_grth080 = lines[491].strip().split()[-1]
            mam_tmax_avgnds_grth090 = lines[492].strip().split()[-1]
            mam_tmax_avgnds_grth100 = lines[493].strip().split()[-1]
            mam_tmax_avgnds_lsth032 = lines[494].strip().split()[-1]

            mam_tmin_avgnds_lsth000 = lines[495].strip().split()[-1]
            mam_tmin_avgnds_lsth010 = lines[496].strip().split()[-1]
            mam_tmin_avgnds_lsth020 = lines[497].strip().split()[-1]
            mam_tmin_avgnds_lsth032 = lines[498].strip().split()[-1]
            mam_tmin_avgnds_lsth040 = lines[499].strip().split()[-1]
            mam_tmin_avgnds_lsth050 = lines[500].strip().split()[-1]
            mam_tmin_avgnds_lsth060 = lines[501].strip().split()[-1]
            mam_tmin_avgnds_lsth070 = lines[502].strip().split()[-1]

            jja_tmax_normal = lines[503].strip().split()[-1]
            jja_tavg_normal = lines[504].strip().split()[-1]
            jja_tmin_normal = lines[505].strip().split()[-1]
            jja_dutr_normal = lines[506].strip().split()[-1]
            jja_cldd_normal = lines[507].strip().split()[-1]
            jja_htdd_normal = lines[508].strip().split()[-1]

            jja_cldd_base45 = lines[509].strip().split()[-1]
            jja_cldd_base50 = lines[510].strip().split()[-1]
            jja_cldd_base55 = lines[511].strip().split()[-1]
            jja_cldd_base57 = lines[512].strip().split()[-1]
            jja_cldd_base60 = lines[513].strip().split()[-1]
            jja_cldd_base70 = lines[514].strip().split()[-1]
            jja_cldd_base72 = lines[515].strip().split()[-1]

            jja_cldd_base40 = lines[516].strip().split()[-1]
            jja_cldd_base45 = lines[517].strip().split()[-1]
            jja_cldd_base50 = lines[518].strip().split()[-1]
            jja_cldd_base55 = lines[519].strip().split()[-1]
            jja_cldd_base57 = lines[520].strip().split()[-1]
            jja_cldd_base60 = lines[521].strip().split()[-1]

            jja_tmax_avgnds_grth040 = lines[522].strip().split()[-1]
            jja_tmax_avgnds_grth050 = lines[523].strip().split()[-1]
            jja_tmax_avgnds_grth060 = lines[524].strip().split()[-1]
            jja_tmax_avgnds_grth070 = lines[525].strip().split()[-1]
            jja_tmax_avgnds_grth080 = lines[526].strip().split()[-1]
            jja_tmax_avgnds_grth090 = lines[527].strip().split()[-1]
            jja_tmax_avgnds_grth100 = lines[528].strip().split()[-1]
            jja_tmax_avgnds_lsth032 = lines[529].strip().split()[-1]

            jja_tmin_avgnds_lsth000 = lines[530].strip().split()[-1]
            jja_tmin_avgnds_lsth010 = lines[531].strip().split()[-1]
            jja_tmin_avgnds_lsth020 = lines[532].strip().split()[-1]
            jja_tmin_avgnds_lsth032 = lines[533].strip().split()[-1]
            jja_tmin_avgnds_lsth040 = lines[534].strip().split()[-1]
            jja_tmin_avgnds_lsth050 = lines[535].strip().split()[-1]
            jja_tmin_avgnds_lsth060 = lines[536].strip().split()[-1]
            jja_tmin_avgnds_lsth070 = lines[537].strip().split()[-1]

            son_tmax_normal = lines[538].strip().split()[-1]
            son_tavg_normal = lines[539].strip().split()[-1]
            son_tmin_normal = lines[540].strip().split()[-1]
            son_dutr_normal = lines[541].strip().split()[-1]
            son_cldd_normal = lines[542].strip().split()[-1]
            son_htdd_normal = lines[543].strip().split()[-1]

            son_cldd_base45 = lines[544].strip().split()[-1]
            son_cldd_base50 = lines[545].strip().split()[-1]
            son_cldd_base55 = lines[546].strip().split()[-1]
            son_cldd_base57 = lines[547].strip().split()[-1]
            son_cldd_base60 = lines[548].strip().split()[-1]
            son_cldd_base70 = lines[549].strip().split()[-1]
            son_cldd_base72 = lines[550].strip().split()[-1]

            son_cldd_base40 = lines[551].strip().split()[-1]
            son_cldd_base45 = lines[552].strip().split()[-1]
            son_cldd_base50 = lines[553].strip().split()[-1]
            son_cldd_base55 = lines[554].strip().split()[-1]
            son_cldd_base57 = lines[555].strip().split()[-1]
            son_cldd_base60 = lines[556].strip().split()[-1]

            son_tmax_avgnds_grth040 = lines[557].strip().split()[-1]
            son_tmax_avgnds_grth050 = lines[558].strip().split()[-1]
            son_tmax_avgnds_grth060 = lines[559].strip().split()[-1]
            son_tmax_avgnds_grth070 = lines[560].strip().split()[-1]
            son_tmax_avgnds_grth080 = lines[561].strip().split()[-1]
            son_tmax_avgnds_grth090 = lines[562].strip().split()[-1]
            son_tmax_avgnds_grth100 = lines[563].strip().split()[-1]
            son_tmax_avgnds_lsth032 = lines[564].strip().split()[-1]

            son_tmin_avgnds_lsth000 = lines[565].strip().split()[-1]
            son_tmin_avgnds_lsth010 = lines[566].strip().split()[-1]
            son_tmin_avgnds_lsth020 = lines[567].strip().split()[-1]
            son_tmin_avgnds_lsth032 = lines[568].strip().split()[-1]
            son_tmin_avgnds_lsth040 = lines[569].strip().split()[-1]
            son_tmin_avgnds_lsth050 = lines[570].strip().split()[-1]
            son_tmin_avgnds_lsth060 = lines[571].strip().split()[-1]
            son_tmin_avgnds_lsth070 = lines[572].strip().split()[-1]

        elif next_heading == "Temperature-Related Pseudonormals":
            print "TRPN"
        elif next_heading == "Precipitation-Related Normals":
            print "PRN"
        elif next_heading == "Precipitation-Related Pseudonormals":
            print "PRPN"
