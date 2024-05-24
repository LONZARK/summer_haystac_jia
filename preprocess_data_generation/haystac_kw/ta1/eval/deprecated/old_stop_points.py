import numpy as np
import geopandas as gpd
import pandas as pd


def rolling_stop(points, times, time_window=120, dist_thresh=75):
    start_idx = 0
    stop_idx = 1
    stopped = False
    stops = []
    if (times[stop_idx] - times[start_idx]) >= time_window:
        stopped = True
    while stop_idx < len(times):
        time_at_point = times[stop_idx] - times[stop_idx-1]
        if stopped:
            ndist = np.max(np.linalg.norm(points[start_idx:stop_idx] -
                                          points[stop_idx][None, :],
                                          axis=1))
            if ndist < dist_thresh:
                stop_idx += 1
            else:
                stops.append((start_idx, stop_idx))
                stopped = time_at_point >= time_window
                start_idx = stop_idx
                stop_idx += 1
                if stop_idx >= len(times):
                    break
                while (times[stop_idx] - times[start_idx]) < time_window and \
                        stop_idx < len(times):
                    stop_idx += 1
                    if stop_idx >= len(times):
                        break
        else:
            max_dist = np.max(np.linalg.norm(points[start_idx:stop_idx,
                                                    np.newaxis] -
                                             points[start_idx:stop_idx],
                                             axis=2))
            if max_dist < dist_thresh:
                stopped = True
                stop_idx += 1
            else:
                if time_at_point >= time_window:
                    stopped = True
                    start_idx = stop_idx
                else:
                    start_idx += 1
                    stop_idx = start_idx
                stop_idx += 1
                if stop_idx >= len(times):
                    break
                while (times[stop_idx] - times[start_idx]) < time_window and \
                        stop_idx < len(times):
                    stop_idx += 1
                    if stop_idx >= len(times):
                        break
    if stopped:
        stops.append((start_idx, len(points)-1))
    return stops


def calculate_stop_points(agent_id, df):
    tso = df['timestamp'].values
    ts_full = df['timestamp'].values

    # extract all coordinates in order
    coords_full = df[['longitude', 'latitude']].to_numpy()
    mask = np.concatenate([[True],
                           np.linalg.norm(coords_full[1:]-coords_full[:-1],
                                          axis=1) > 0])
    mask = mask | np.concatenate([np.linalg.norm(coords_full[:-1] -
                                  coords_full[1:],axis=1) > 0, [True]])
    sdf = df[mask]
    tso = tso[mask]

    gdf = gpd.GeoDataFrame(sdf,
                           geometry=gpd.points_from_xy(sdf.longitude,
                                                       sdf.latitude),
                           crs="EPSG:4326")
    crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(crs)

    x = [x.x for x in gdf.geometry.values] + [gdf.geometry.values[-1].x]
    y = [x.y for x in gdf.geometry.values] + [gdf.geometry.values[-1].y]
    ts = np.concatenate([gdf['timestamp'].values, [df.timestamp.values[-1]]])
    tso = np.concatenate([tso, tso[-1:]])
    coords = np.vstack([x, y]).T

    stop_times = rolling_stop(coords, ts)
    data = {'agent_id': [],
            'time_start': [],
            'time_stop': [],
            'x': [],
            'y': []
            }
    for stop_instance in stop_times:
        start, stop = stop_instance
        mask = (ts_full >= ts[start]) & (ts_full <= ts[stop])
        if start == stop:
            continue
        data['agent_id'].append(agent_id)
        data['time_start'].append(tso[start])
        data['time_stop'].append(tso[stop])
        c = np.average(coords_full[mask], axis=0)
        data['x'].append(c[0])
        data['y'].append(c[1])
    df = pd.DataFrame(data)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y),
                           crs='EPSG:4326')
    itms = ['agent_id', 'time_start', 'time_stop', 'geometry']
    gdf = gdf.to_crs("EPSG:4326").dropna()[itms]
    return gdf
