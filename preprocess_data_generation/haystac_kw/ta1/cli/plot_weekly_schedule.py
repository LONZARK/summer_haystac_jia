import os 
import glob 

import tqdm 
import argparse

from haystac_kw.utils.data_utils.data import examine_parquet_data
from haystac_kw.utils.viz.weekplot_schedules import load_stop_point_data, \
                                            build_schedule_txt, \
                                            generateWeekplotFig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Schedule Visualization',
                    description='Script responsible for generating \
                        WeekPlot visualizations of agent schedules.')
    
    parser.add_argument('-d', '--data', 
                        required=True, 
                        help="path of agent parquet file to visualize. \
                            Parquet file must describe stop points.")
    
    args = parser.parse_args()

    fig_dir = "weekplot_figs"
    os.makedirs(fig_dir, exist_ok=True)

    if ".parquet" in args.data:
        saveName = os.path.join(fig_dir, os.path.basename(args.data.replace(".parquet", "")))
        agent_id = os.path.basename(args.data.replace(".parquet", ""))

        try: # attempt to plot parquet data
            examine_parquet_data(args.data)
            schedule_data = load_stop_point_data(args.data)
            string_data = build_schedule_txt(schedule_data)
            generateWeekplotFig(string_data, agent_id=agent_id, name=saveName)
        
        except Exception as exc: # print error message if parquet is bad
            print(exc)
            print("failed to plot parquet: %s" % args.data)

    elif os.path.isdir(args.data):
        
        plot_failures = []

        parquet_list = glob.glob(os.path.join(args.data, "*.parquet"))

        pbar = tqdm.tqdm(parquet_list)
        for p in pbar:
            
            agent_id = os.path.basename(p.replace(".parquet", ""))

            try:
                saveName = os.path.join(fig_dir, agent_id)
                schedule_data = load_stop_point_data(p)
                string_data = build_schedule_txt(schedule_data)
                generateWeekplotFig(string_data, agent_id=agent_id, name=saveName)
                pbar.set_description("saving fig: %s.png" % saveName)
            
            except Exception as exc: # print error message if parquet is bad but continue
                print(exc)
                print("failed to plot parquet: %s" % p)
                print("examining parquet data.. ")
                plot_failures.append(p)

        print("\nThe following parquet files failed in plot generation:")
        for f in plot_failures: 
            print(f)
            examine_parquet_data(f)
            print()
        
    print("done.")