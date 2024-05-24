
''' This file contains functions for plotting weekplot schedules. 
These functions are implemented in addition to those from third_party_ext.weekplot functions.
'''

from typing import List
from math import ceil 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from haystac_kw.third_party_ext.weekplot import DAYS, DAYS_TRUNCATED, plotEvent, Event, getDay

SCHEDULE_COLORS = {
    'Primary' : 'cornflowerblue', 
    'Home' : 'lightgreen', 
    'Other' : "mediumpurple"
}


def _plotWeekplot(events : List[Event], earliest : int, latest : int, title : str, plotname : str="weekplot"):
    """Helper function for plotting weekplot. Shouldn't be directly called.

    Parameters
    ----------
    events : List[Event]
        parsed event from parseStr()
    earliest : int
        earliest time to plot; usually 0
    latest : int
        latest time to plot; usually 24
    title : str
        title of the figure to put on the plot
    plotname : str, optional
        name of the plot to save as (defaults to 'weekplot')
    """

    fig = plt.figure(figsize=(18, 9))

    plt.title(title, y=1, fontsize=14)
    plt.xticks([], [])
    plt.yticks([], [])

    ax=fig.add_subplot(1, 1, 1)
    ax.set_xlim(0.5, len(DAYS) + 0.5)
    ax.set_xticks(range(1, len(DAYS) + 1), DAYS)
    ax.set_ylim(latest, earliest)
    ax.set_yticks(range(ceil(earliest), ceil(latest)), ["{0}:00".format(h) for h in range(ceil(earliest), ceil(latest))])
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    for e in events:
        plotEvent(e, ax)

    # print("generating week plot fig: %s.png" % name)
    plt.savefig('{0}.png'.format(plotname), dpi=200, bbox_inches='tight')
    plt.close()
    

def generateWeekplotFig(data_str : str, agent_id : str = None, name : str ="weekplot_viz"):
    """Generate and save the weekplot visualization.
    data_str should contain the schedule information needed.

    Parameters
    ----------
    data_str : str: str
        string containing the schedule information
    agent_id : str, optional
        agent id to plot (defaults to None)
    name : str
        filename to save the plot as (defaults to 'weekplot_viz')
    """

    events, earliest, latest = parseStr(data_str)
    title_str = "Weekly Schedule" if agent_id is None else "Weekly Schedule: Agent %s" % agent_id
    _plotWeekplot(events, earliest, latest, title_str, plotname=name)


def build_schedule_txt(schedule_list : list) -> str:
    """Create the intermediary string to build figure with weekplot extension.

    Parameters
    ----------
    schedule_list : list
        list of event info. note this method should not be directly called and should rather be seen as a helper function

    Returns
    -------
    str
        schedule represented as a string readable by weekplot module
    """

    event_data = [] 

    for event in schedule_list:
        
        str_data = "\n".join(event)
        event_data.append(str_data)
    
    contents = "\n\n".join(event_data)

    return contents

def load_stop_point_data(path : str):
    """Parses parquet files from the given path and processes information
    to a suitable format for weekplot.

    Parameters
    ----------
    path : str
        path containing stop point parquet files

    Returns
    -------
    List (mixed types)
        schedule data in the weekplot data format
    """
    
    df = pd.read_parquet(path)

    schedule_data = [] 
    
    assert len(df) > 1, "agent never arrived at destination from home"

    # add the home data manually since an agent starts at home
    departed_row = df.iloc[0]
    assert departed_row["event_type"] == "departed"
    day_of_the_week = DAYS_TRUNCATED[departed_row["timestamp"].weekday()]
    hour_min_time_end = departed_row["timestamp"].strftime("%H:%M")
    
    if hour_min_time_end[0] == "0":
        hour_min_time_end = hour_min_time_end[1:]

    start_data = ["Home", day_of_the_week, "%s - %s" % ("0:00", hour_min_time_end), SCHEDULE_COLORS["Home"]]
    schedule_data.append(start_data)

    # last row of the dataframe, the agent arrives at home and does not depart again
    arrived_row = df.iloc[-1]
    
    if arrived_row["event_type"] == "departed":
        df = df[:-1]
        df.reset_index()
        arrived_row = df.iloc[-1]
        assert arrived_row["event_type"] == "arrived"

    day_of_the_week = DAYS_TRUNCATED[arrived_row["timestamp"].weekday()]
    hour_min_time_start = arrived_row["timestamp"].strftime("%H:%M")
    
    if hour_min_time_start[0] == "0":
        hour_min_time_start = hour_min_time_start[1:]

    end_data = ["Home", day_of_the_week, "%s - %s" % (hour_min_time_start, "23:59"), SCHEDULE_COLORS["Home"]]
    
    df = df.iloc[1:-1] # remove first departure from home

    for i, g in df.groupby(np.arange(len(df)) // 2):
        
        print(g)

        arrived_row = g.iloc[0]
        departed_row = g.iloc[1] if len(g) > 1 else None 

        assert arrived_row["event_type"] == "arrived"
        assert departed_row["event_type"] != "arrived"

        day_of_the_week = DAYS_TRUNCATED[arrived_row["timestamp"].weekday()]
        hour_min_time_start = arrived_row["timestamp"].strftime("%H:%M")
        hour_min_time_end = departed_row["timestamp"].strftime("%H:%M") if departed_row is not None else "23:59"

        day_of_the_week_end = DAYS_TRUNCATED[departed_row["timestamp"].weekday()]

        if hour_min_time_start[0] == "0":
            hour_min_time_start = hour_min_time_start[1:]
        if hour_min_time_end[0] == "0":
            hour_min_time_end = hour_min_time_end[1:]

        parse_event_label = arrived_row["event_label"].split(":")

        if len(parse_event_label) > 1:
            event_label = parse_event_label[0].capitalize()
        else: 
            event_label = "Other"
        
        if day_of_the_week != day_of_the_week_end: # stayed at location across days
            
            sample = [event_label, day_of_the_week, "%s - %s" % (hour_min_time_start, "23:59"), SCHEDULE_COLORS[event_label]]
            sample2 = [event_label, day_of_the_week_end, "%s - %s" % ("0:00", hour_min_time_end), SCHEDULE_COLORS[event_label]]
            schedule_data.append(sample) 
            schedule_data.append(sample2)

        else: 
            sample = [event_label, day_of_the_week, "%s - %s" % (hour_min_time_start, hour_min_time_end), SCHEDULE_COLORS[event_label]]
            schedule_data.append(sample)

    schedule_data.append(end_data)

    return schedule_data

def parseStr(data):
    """string version of parseTxt and parseYml from original weekplot code.
    This was implemented to prevent intermediary files from being generated,
    e.g. instead of outputting a txt file and then reading from that text file
    to generate the weekplot, use the string generated directly to generate
    the plot without saving the txt.

    Parameters
    ----------
    data : str
        string representation of weekly event schedules

    Returns
    -------
    List[Event], int, int
        - events: list of Event objects
        - earliest: earliest time to plot (typically 0)
        - latest: latest time to plot (typically 24)
    """

    lines = data.split("\n")

    index = 0
    latest = 24
    earliest = 0
    events = [Event('', '', '', '', '', '', '')]
    for line in lines:
        line = line.rstrip()
        index += 1
        if index == 1:
            events[-1].name = line
        elif index == 2:
            events[-1].days = [getDay(d) for d in line.replace(' ', '').split(',')]
        elif index == 3:
            hours = line.replace(' ', '').split('-')
            start = hours[0].split(':')
            end = hours[1].split(':')
            events[-1].startH = int(start[0])
            events[-1].startM = int(start[1])
            events[-1].endH = int(end[0])
            events[-1].endM = int(end[1])
            earliest = events[-1].startH if events[-1].startH < earliest else earliest
            latest = events[-1].endH + 1 if events[-1].endH > latest else latest
        elif index == 4:
            events[-1].color = line
        elif index == 5 and line == '':
            events.append(Event('', '', '', '', '', '', ''))
            index = 0
        else:
            raise UserWarning("Invalid text input format.")
    return events, earliest, latest + 1
