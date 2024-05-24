""" This file is a self-contained file which builds the web application 
for human-based HOS app generation. To run this and the HAS app, 
the user will need to download the HOS app dependency files. 
Details on these files can be found in the documentation for 
the HOS and HAS app. 

To launch the HOS app, you can run: 

python haystac_kw/utils/specifications/hos_app.py \
    --path [path/to/hos/app/files]
"""

import os
import json 
import uuid
import argparse 

from typing import List 

import plotly.express as px
import dash
from dash import callback
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import math 
from datetime import datetime, timedelta

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

from shapely import GeometryCollection
from shapely.ops import unary_union

POI_PICKLE = None 
ADMIN_PATH = None
DEFAULT_ZONE = "Choa Chu Kang"

constraint_data = []

num_to_day_map = {
    1: "Monday", 
    2: "Tuesday", 
    3: "Wednesday", 
    4: "Thursday", 
    5: "Friday", 
    6: "Saturday", 
    7: "Sunday"
}

day_to_date_map = {
    1: datetime(2023, 2, 27, 0, 0, 0, 0), 
    2: datetime(2023, 2, 28, 0, 0, 0, 0),
    3: datetime(2023, 3, 1, 0, 0, 0, 0), 
    4: datetime(2023, 3, 2, 0, 0, 0, 0), 
    5: datetime(2023, 3, 3, 0, 0, 0, 0), 
    6: datetime(2023, 3, 4, 0, 0, 0, 0), 
    7: datetime(2023, 3, 5, 0, 0, 0, 0), 
}

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

plotted_df = None

def build_app():
    """Main entrypoint for building the HOS web app.
    """

    fig, admin_df, poi_df = plot_map(DEFAULT_ZONE)
    gantt_fig = {}
    subzone_list = ["All"] + sorted(admin_df['Name'].unique())
    app = dash.Dash(external_stylesheets=[dbc.themes.MINTY], 
                    prevent_initial_callbacks="initial_duplicate")
    app.title = "Hide Objective Specification Generator"

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Hide Objective Specification Generator"),
                            html.H5("HAYSTAC"),
                        ],
                        width=True,
                    ),
                    # dbc.Col([
                    #     html.Img(src="assets/MIT-logo-red-gray-72x38.svg", alt="MIT Logo", height="30px"),
                    # ], width=1)
                ],
                align="end",
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Tabs([

                            dcc.Tab(label='Create Event', children=[
                                html.Div(
                                    [

                                        html.H5("Region Selection"),

                                        dcc.Dropdown(
                                            subzone_list,
                                            DEFAULT_ZONE,
                                            id='subzone'
                                        ),
                                        
                                        html.Br(),
                                        html.Hr(), 
                                        html.Br(),

                                        html.H5("Event Creation"),

                                        html.H6("Specify Agent IDs"),
                                        html.P("Provide a list of agent IDs. Separate with commas to define multiple."),
                                        dcc.Input(id="agent_ids", value="0", type="text"),
                                        
                                        html.Br(),
                                        html.Br(),
                                        html.H6("Specify Event Type"),
                                        html.P("Choose event type, which is either arrive or depart."),
                                        dcc.Dropdown(
                                            ["arrive", "depart"],
                                            'arrive',
                                            id='event_type'
                                        ),
                                        html.Br(),

                                        html.H6("Select Location"),
                                        html.P("Click buildings on the map to select a location."),
                                        html.Pre(id='click-data', style=styles['pre']),
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Create Event",
                                                    id="create_event",
                                                    color="primary",
                                                    style={"margin": "5px"},
                                                ), 
                                                html.P('', id='create_event_msg'),
                                            ]
                                        ),
                                        html.Br(),

                                        html.Hr(),

                                    ]), 
                            ]),
                            dcc.Tab(label='Add Constraints', children=[
                                html.Div(
                                    [   
                                        html.Br(),

                                        html.H5("Last Added Event"),
                                        html.P("Constraints added below will apply to this event."),
                                        html.Pre(id='event-data', style=styles['pre']),

                                        html.Br(),
                                        html.Hr(), 
                                        html.Br(),

                                        html.H5("Time Constraint Creation"),

                                        dcc.Slider(1, 7, value=3, step=1,
                                            id="constraint_day",
                                            marks={
                                                1: {'label': 'Monday'},
                                                2: {'label': 'Tuesday'},
                                                3: {'label': 'Wednesday'},
                                                4: {'label': 'Thursday'},
                                                5: {'label': 'Friday'},
                                                6: {'label': 'Saturday'},
                                                7: {'label': 'Sunday'},
                                            },
                                            included=False
                                        ),

                                        dcc.RangeSlider(
                                            id="time_frame",
                                            min=0,
                                            max=24,
                                            step=0.25,
                                            value=[0, 2],
                                            marks={
                                                0: {'label': '00:00', 'style': {'color': '#77b0b1'}},
                                                3: {'label': '03:00'},
                                                9: {'label': '09:00'},
                                                6: {'label': '06:00'},
                                                9: {'label': '09:00'},
                                                12: {'label': '12:00'},
                                                15: {'label': '15:00'},
                                                18: {'label': '18:00'},
                                                21: {'label': '21:00'},
                                                24: {'label': '24:00', 'style': {'color': '#f50'}}
                                            }
                                        ),
                                        html.Div(id='time-range-slider'),

                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Add Time Constraint",
                                                    id="add-time-constraint",
                                                    color="primary",
                                                    style={"margin": "5px"},
                                                ), 
                                                dbc.Button(
                                                    "Remove Last Time Constraint",
                                                    id="remove-time-constraint",
                                                    color="primary",
                                                    style={"margin": "5px"},
                                                ), 

                                            ]
                                        ),

                                        html.Br(),
                                        html.Hr(), 
                                        html.Br(),

                                        html.H5("Duration Constraint Creation"),
                                        html.P("Select the two events:"),
                                        dcc.Dropdown(
                                            [],
                                            '',
                                            id='duration_event1'
                                        ),
                                        dcc.Dropdown(
                                            [],
                                            '',
                                            id='duration_event2'
                                        ),
                                        
                                        html.P("Duration Minimum [m]:"),
                                        dcc.Input(id="duration_min", value=0, type="number"),

                                        html.P("Duration Maximum [m]:"),
                                        dcc.Input(id="duration_max", value=0, type="number"),

                                        html.P("Stay at Location? (boolean)"),
                                        dcc.Dropdown(
                                            ["true", "false"],
                                            'true',
                                            id='stay_dropdown'
                                        ),

                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Add Duration Constraint",
                                                    id="add-duration-constraint",
                                                    color="primary",
                                                    style={"margin": "5px"},
                                                ), 
                                                dbc.Button(
                                                    "Remove Last Duration Constraint",
                                                    id="remove-duration-constraint",
                                                    color="primary",
                                                    style={"margin": "5px"},
                                                ), 

                                                html.Pre(id='duration-constraint-data', style=styles['pre'])

                                            ]
                                        ),

                                    ]
                                ),
                            ]), 

                            dcc.Tab(label='Export JSON', children=[
                                html.Hr(),
                                html.H5("Current JSON Data"),
                                html.Pre(id='json-data', style=styles['pre']),
                                html.Div(
                                    [
                                        html.P("Make sure to define the objective narrative before generating the final JSON."),
                                        dcc.Input(id="narrative", value="", type="text"),
                                        dbc.Button(
                                            "Generate Final JSON",
                                            id="generate_json",
                                            color="primary",
                                            style={"margin": "5px"},
                                        )
                                    ]
                                ),
                                html.P('JSON not saved yet', id='json-export'),
                                html.Hr()
                            ])
                            ]),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Spinner(
                                dcc.Graph(figure=fig, id="display", style={"height": "80vh"}),
                                color="primary",
                            ), 
                            dbc.Spinner(
                                dcc.Graph(figure=gantt_fig, id="display2", style={"height": "80vh"}),
                                color="primary",
                            )
                        ],
                        width=8,
                    ),
                ]
            ),
            html.Hr(),
            html.P(
                [
                    html.A(
                        "Source code",
                        href="https://github.com/peterdsharpe/AeroSandbox-Interactive-Demo",
                    ),
                    ". Aircraft design tools powered by ",
                    html.A(
                        "AeroSandbox", href="https://peterdsharpe.github.com/AeroSandbox"
                    ),
                    ". Build beautiful UIs for your scientific computing apps with ",
                    html.A("Plot.ly ", href="https://plotly.com/"),
                    "and ",
                    html.A("Dash", href="https://plotly.com/dash/"),
                    "!",
                ]
            ),
        ],
        fluid=True,
    )

    app.run_server(debug=True)

def plot_map(subzone):
    """Plots the map as a Plotly express widget.

    Parameters
    ----------
    subzone : str
        Level 4 subzone to filter for.
        This is typically passed in through the UI dropdown menu.
        Subzone names can be found in the level 4 geojson file that
        this app is dependent on.

    Returns
    -------
    tuple(plotly.graph_objects.Figure, geopandas.GeoDataFrame, geopandas.GeoDataFrame)
        - fig
        - admin_df
        - poi_df
    """

    admin_df = gpd.read_file(ADMIN_PATH, 
                            geometry="geometry", 
                            crs='EPSG:4326')

    admin_df['LogDensity'] = np.log10(admin_df['Census Population'] / admin_df['Land Area'])

    if subzone.lower() == "all":
        row = admin_df
        subzone_geom = row["geometry"].values
        subzone_geom = unary_union(subzone_geom)
    else:  
        row = admin_df[admin_df['Name'] == subzone]
        subzone_geom = row["geometry"].values[0]

    subzone_geom_center = subzone_geom.centroid 

    poi_df = pd.read_pickle(POI_PICKLE)
    poi_df = gpd.GeoDataFrame(poi_df, 
                            geometry="building_centroid", 
                            crs='EPSG:4326')


    utm = poi_df.estimate_utm_crs() 

    poi_df = gpd.GeoDataFrame(poi_df, 
                            geometry="building_poly", 
                            crs=utm).to_crs('EPSG:4326')


    within_subzone = poi_df[poi_df["building_centroid"].within(subzone_geom)].reset_index()

    global plotted_df 
    plotted_df = within_subzone

    fig = px.choropleth_mapbox(within_subzone, 
                            geojson=within_subzone.geometry, locations=within_subzone.index,
                            mapbox_style="open-street-map",
                            zoom=12, 
                            # center = {"lat": 1.3521, "lon": 103.8198},
                            center = {'lat': subzone_geom_center.y, 'lon': subzone_geom_center.x},
                            opacity=0.6, 
                            color_continuous_scale="reds",
                            hover_data=[within_subzone.index, 'id_building', 'id_road', "length_along_edge",'h3'],
                )


    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(clickmode='event+select')

    return fig, admin_df, poi_df

def number_to_time(num : float) -> str:
    """Converts a number of hours to a formatted time of HH:MM:00.

    Parameters
    ----------
    num : float
        number of hours
        
    Returns
    -------
    str
        formatted time as HH:MM:00.
    """

    hours, minutes = number_to_hours_mins(num)
    formatted_time = "%02d:%02d:00" % (hours,minutes)
    return formatted_time

def number_to_hours_mins(num : float) -> tuple:
    """Converts a float number input in units of hours to hours and minutes.

    Parameters
    ----------
    num : float
        number of hours

    Returns
    -------
    tuple(int, int)
        - hours : number of hours
        - minute : number of minutes
    """

    minutes = int((num - math.floor(num)) * 60)
    hours = int(num)

    return hours, minutes

def initialize_empty_json() -> dict:
    """Initializes an empty JSON file representing the schema for HOS.

    Returns
    -------
    dict
        empty json data
    """

    json_data = {
        "schema_type": "HOS",
        "objective_uid": str(uuid.uuid4()),
        "narrative" : "",
        "events": [],
        "time_constraints": [],
        "duration_constraints": []
    }

    return json_data

def create_event(event_uid : str=str(uuid.uuid4()), agents : List[str] =[], event_type : str="depart", location : dict={}) -> dict:
    """Creates and returns an event dictionary according to the HOS event schema.

    Parameters
    ----------
    event_uid : uuid.uuid4
        unique event identifier
    agents : List[str]
        list of agent ids involved in the event
    event_type : str
        event type; either "depart" or "arrive"
    location : dict
        GEOJSON dictionary representing the location of the event

    Returns
    -------
    dict
        event data dictionary in according to the HOS schema
    """

    assert event_type in ["depart", "arrive"], "event_type must be either depart or arrive"

    global event_id 
    event_id = event_uid 

    event_data = { 
        "event_uid" : event_uid, 
        "agents" : agents, 
        "event_type" : event_type, 
        "location" : location
    } if location is not None else None

    return event_data

def create_time_constraint(event : str, time_window : dict) -> dict: 
    """Creates and returns an time constraint dictionary according to the HOS event schema.

    Parameters
    ----------
    event : str
        unique id of event concerning the time constraint
    time_window : dict
        dictionary of start and end times of constraint

    Returns
    -------
    dict
        time constraint dictionary item
    """

    assert "begin" in time_window.keys() and "end" in time_window.keys(), "time window must contain a begin and end"

    time_const_data = {
        "event" : event, 
        "time_window" : time_window
    }

    return time_const_data 

def create_duration_constraint(events : dict, duration_window : dict, stay : bool) -> dict:
    """Creates and returns a duration constraint dictionary according to the HOS event schema.

    Parameters
    ----------
    events : dict
        dictionary of two events concerning this duration constraint
    duration_window : dict
        dictionary of start and end times of constraint
    stay : bool 
        indicates whether the agent should remain within the location
        between events

    Returns
    -------
    dict
        duration constraint dictionary item
    """

    assert "first" in events.keys() and "second" in events.keys(), \
        "events dictionary arg must have a first and second event"
    assert "minimum" in duration_window.keys() and "maximum" in duration_window.keys(), \
        "duration_window arg must have a minimum and maximum"
    
    duration_const_data = {
        "events" : events, 
        "duration_window" : duration_window, 
        "stay" : True if stay.lower() == "true" else False 
    }

    return duration_const_data

def update_gantt():
    """Updates the Gantt plot of time constraints below the map.
        This graph is primarily used for debugging purposes.

    Returns
    -------
    plotly.graph_objects.Figure
        figure
    """
    global constraint_data

    df = pd.DataFrame(constraint_data)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Constraint", color="Agent", 
                      width=1200, height=900)
    fig.update_yaxes(autorange="reversed")

    return fig

## CALLBACK FUNCTIONS FOR PLOTLY BELOW

@callback(
    Output('click-data', 'children'),
    Input('display', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@callback(
    Output('json-export', 'children'),
    Input('generate_json', 'n_clicks'), 
    State("narrative", "value"))
def export_json(n_clicks, narrative):

    global json_data 
    json_data["narrative"] = narrative

    if n_clicks is None or n_clicks < 0: 
        return 'JSON not saved yet'

    filepath = 'hos_output.json'

    with open(filepath, 'w') as fp:
        json.dump(json_data, fp, indent=4)
    
    return 'Exported JSON to %s' % filepath

@callback(
    Output('json-data', 'children'), 
    Input('event-data', 'children'), 
    Input('display2', 'figure'), 
    Input('duration-constraint-data', 'children'))
def update_json_info(event_data, display2, duration_constraint_data): 
    global json_data
    return json.dumps(json_data, indent=2)

@callback(
    Output('event-data', 'children'),
    Output('duration_event1', 'options'), 
    Output('duration_event2', 'options'), 
    Output('create_event_msg', 'children'),
    Input('create_event', 'n_clicks'),
    State("agent_ids", "value"), 
    State("event_type", "value"), 
    State("click-data", "children"), 
    prevent_initial_call=True)
def add_event(n_clicks, agent_ids, event_type, location):
    
    agents = [int(x.strip()) for x in agent_ids.split(",")]

    location = json.loads(location)

    geometry_info_i = location["points"][0]["customdata"][0]

    global plotted_df
    location_polygon = plotted_df.iloc[geometry_info_i]["building_poly"]

    geojson_collection = GeometryCollection([location_polygon])

    geojson_dict = json.loads(gpd.GeoSeries(geojson_collection).set_crs('EPSG:4326').to_json())
    geojson_dict = geojson_dict['features'][0]['geometry']

    global current_event_data
    current_event_data = create_event(event_uid=str(uuid.uuid4()), agents=agents, event_type=event_type, location=geojson_dict)
    
    if current_event_data is not None and location is not None:
        global json_data 
        json_data["events"].append(current_event_data)

    event_options = [{'label': i["event_uid"], 'value': i["event_uid"]} for i in json_data["events"]]

    event_data_copy = current_event_data.copy()
    event_data_copy.pop('location', None)

    return json.dumps(event_data_copy, indent=2), event_options, event_options, "New event created with ID: %s" % (current_event_data["event_uid"])

@callback(
    Output('duration-constraint-data', 'children', allow_duplicate=True),
    Input('remove-duration-constraint', 'n_clicks'), 
    prevent_initial_call=True)
def remove_duration_constraint(n_clicks):

    global json_data 
    json_data["duration_constraints"] = json_data["duration_constraints"][:-1]

    return json_data["duration_constraints"]

@callback(
    Output('display2', 'figure', allow_duplicate=True),
    Input('remove-time-constraint', 'n_clicks'), 
    prevent_initial_call=True)
def remove_time_constraint(n_clicks):

    global json_data 
    json_data["time_constraints"] = json_data["time_constraints"][:-1]

    global constraint_data
    constraint_data = constraint_data[:-1]

    if len(constraint_data) > 0:
        fig = update_gantt()
    else: 
        fig = {}

    return fig

@callback(
    Output('time-range-slider', 'children'),
    Input('time_frame', 'value'),
    Input('constraint_day', 'value'))
def update_arrival_slider(time_frame, constraint_day):
    time_frame = [number_to_time(v) for v in time_frame]
    day_label = num_to_day_map[constraint_day]
    return 'Selected time range: %s, %s - %s' % (day_label, time_frame[0], time_frame[1])
        
@callback(
Output('display', 'figure'),
Input('subzone', 'value'))
def update_graph(subzone_name):

    fig, _, _ = plot_map(subzone_name)
    return fig

@callback(
    Output('duration-constraint-data', 'children'), 
    Input('add-duration-constraint', 'n_clicks'), 
    State('duration_event1', 'value'), 
    State('duration_event2', 'value'), 
    State('duration_min', 'value'), 
    State('duration_max', 'value'), 
    State('stay_dropdown', 'value'), 
    prevent_initial_call=True
)
def add_duration(n_clicks, duration_event1, duration_event2, duration_min, duration_max, stay_dropdown):

    if duration_event1 == duration_event2:
        return "Constraint failed; first and second event cannot be the same event."

    if duration_min > duration_max:
        return "Constraint failed; duration minimum cannot be greater than duration maximum."
        
    duration_min = pd.Timedelta(minutes=float(duration_min)).isoformat()
    duration_max = pd.Timedelta(minutes=float(duration_max)).isoformat()

    global json_data
    
    duration_const_data = {
        "events" : {
            "first" : duration_event1, 
            "second" : duration_event2, 
        }, 
        "duration_window" : {
            "minimum" : duration_min, 
            "maximum" : duration_max, 
        }, 
        "stay" : stay_dropdown
    }

    json_data["duration_constraints"].append(duration_const_data)

    return json.dumps(duration_const_data, indent=2)

@callback(
    Output('display2', 'figure'),
    Input('add-time-constraint', 'n_clicks'),
    State('time_frame', 'value'), 
    State('constraint_day', 'value'), 
    prevent_initial_call=True)
def add_tc(n_clicks, time_frame, constraint_day):

    global current_event_data
    global json_data

    if current_event_data is not None: 
        event_id = current_event_data["event_uid"]
        event_agents = current_event_data["agents"]
        event_type = current_event_data["event_type"]

        value = [number_to_hours_mins(v) for v in time_frame]
        day = day_to_date_map[constraint_day] # .isoformat()

        begin_date = day + timedelta(hours=value[0][0], minutes=value[0][1])
        end_date= day + timedelta(hours=value[1][0], minutes=value[1][1])
        
        begin_str = begin_date.isoformat() 
        end_str = end_date.isoformat() 

        time_window_data = {
            "begin": begin_str, 
            "end": end_str
        }

        tc_data = create_time_constraint(event=event_id, time_window=time_window_data)
        json_data["time_constraints"].append(tc_data)

        global constraint_data

        for agent in event_agents:
            cdata = {
                "Constraint" : "%d_%s" % (len(constraint_data), event_type),
                "Start" : begin_str, 
                "Finish" : end_str, 
                "Agent" : int(agent) 
            }

            constraint_data.append(cdata)
    
    if len(constraint_data) > 0:
        fig = update_gantt()
    else: 
        fig = {}

    return fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Hide Objective Specification Web Application',
                    description='Web application for generating sample \
                        hide objective specifications based on the JSON \
                        schema given to by the government for HAYSTAC.')
    
    parser.add_argument('-p', '--path', 
                        required=True, 
                        help="Path containing supporting map and building \
                            files for web app visualization.")
    
    args = parser.parse_args()

    POI_PICKLE = os.path.join(args.path, "poi.pkl")
    ADMIN_PATH = os.path.join(args.path, "singapore_planning_area_lvl4.geojson")

    global json_data 
    json_data = initialize_empty_json() 

    global current_event_data
    current_event_data = None 

    build_app()