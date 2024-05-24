import pandas as pd
from rich import print
import time
import pymap3d
csv1 = "/home/local/KHQ/joseph.vanpelt/projects/haystac/data/test.csv"
csv2 = "/home/local/KHQ/joseph.vanpelt/projects/haystac/data/train.csv"
out1 = "/home/local/KHQ/joseph.vanpelt/projects/haystac/data/test_enu.csv"
out2 = "/home/local/KHQ/joseph.vanpelt/projects/haystac/data/train_enu.csv"
# central point in knoxville
_lat0 = 35.960443
_lon0 = -83.921263


def get_enu_from_ll(lat, lon):
    """
    This function takes a lat/lon and returns an East/North combo
    we are assuming height of zero
    """
    en = pymap3d.enu.geodetic2enu(lat, lon, 0, _lat0, _lon0, 0)
    return en[:2]


if __name__ == "__main__":

    ins = [csv1, csv2]
    outs = [out1, out2]

    for x in range(0, len(ins)):
        in_file = ins[x]
        out_file = outs[x]
        csvq = time.time()
        table1 = pd.read_csv(in_file, header=0)

        first_dist_stops = {}
        east = []
        north = []
        for index, row in table1.iterrows():
            agent = row['agent_id']
            if agent not in first_dist_stops.keys():
                first_dist_stops[agent] = {}
                first_dist_stops[agent]['stop_points'] = []
            coords = row['geometry']
            temp = coords.split("(")[-1]
            temp = temp.split(")")[0]
            lon, lat = temp.split(" ")
            # print(f"{lat}, {lon}")
            en = get_enu_from_ll(float(lat), float(lon))
            east.append(en[0])
            north.append(en[1])
            first_dist_stops[agent]['stop_points'].append(en)
        # print(first_dist_stops)

        table1.insert(4, 'east', east)
        table1.insert(5, 'north', north)

        table1.to_csv(out_file, sep=',', index=False, encoding='utf-8')
        print(f"csv process time: {time.time() - csvq}")
