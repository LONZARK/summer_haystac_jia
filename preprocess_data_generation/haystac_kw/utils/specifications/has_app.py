""" This file is a self-contained file which builds the web application 
for human-based HAS app generation. To run this and the HOS app, 
the user will need to download the HOS app dependency files. 
Details on these files can be found in the documentation for 
the HOS and HAS app. 

To launch the HOS app, you can run: 

python haystac_kw/utils/specifications/has_app.py \
    --path [path/to/hos/app/files]
"""

import os
import json 
import uuid
import argparse 
import base64
from datetime import date

from typing import List, Tuple

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

import plotly.graph_objects as go
from shapely import GeometryCollection, Polygon, Point
from shapely.ops import unary_union

import haystac_kw.utils.specifications.generate_random_has as bhj

POI_PICKLE = None 
ADMIN_PATH = None
HOS_FILE = None
DEFAULT_ZONE = "None"

TRANSPORT_MODES = [
    "personal_vehicle"
]

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
    """
        Main entrypoint for building the HOS web app. 

        :return:
        :rtype: None
    """
    HOS_FILE = None

    global json_data 
    json_data = None

    global has_data
    has_data = None

    global current_movement
    current_movement = None 

    global DEFAULT_ZONE
    if DEFAULT_ZONE == "None" and json_data is None: 
        DEFAULT_ZONE = "Choa Chu Kang"

    global event_data 
    event_data = None 

    global time_constraint_data 
    time_constraint_data = None 

    global duration_constraint_data 
    duration_constraint_data = None 

    global events_by_agent_data 
    events_by_agent_data = None 

    fig, admin_df, poi_df, subzone = plot_map(DEFAULT_ZONE)

    gantt_fig = {}
    subzone_list = ["All"] + sorted(admin_df['Name'].unique())
    app = dash.Dash(external_stylesheets=[dbc.themes.CERULEAN], 
                    prevent_initial_callbacks="initial_duplicate")
    app.title = "Hide Activity Specification Generator"

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Hide Activity Specification Generator"),
                            html.H5("HAYSTAC"),
                        ],
                        width=True,
                    ),
                ],
                align="end",
            ),
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Tabs([

                            dcc.Tab(label='1. Explore HOS', children=[
                                html.Div(
                                    [
                                        html.Hr(),
                                        html.H5("Specify Objective HOS Path"),
                                        dcc.Upload(
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select Files')
                                            ]),
                                            style={
                                                'width': '100%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin': '10px'
                                            },
                                            # Allow multiple files to be uploaded
                                            multiple=False, 
                                            id="objective-path-upload"
                                        ),
                                        html.P("Selected file: None", id='selected-hos-name-msg'),
                                        html.Hr(), 
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Create HAS for Objective",
                                                    id="create-movement-button",
                                                    color="primary",
                                                    style={"margin": "5px"},
                                                ), 
                                                html.P('Please define HOS Objective ID', id='create-movement-msg'),
                                            ]
                                        ),

                                        html.Hr(),
                                        html.H6("List of Unfulfilled Time Constraints"),
                                        html.Pre(id='unfulfilled-time-objectives-display', style=styles['pre']),

                                        html.Hr(), 

                                        html.H6("List of Unfulfilled Duration Constraints"),
                                        html.Pre(id='unfulfilled-duration-objectives-display', style=styles['pre']),

                                        html.Hr()

                                    ]), 
                            ]), 

                            dcc.Tab(label='2. Add Itinerary for Agent', children=[
                                
                                html.Hr(), 

                                html.H5("Agent Selection"),
                                dcc.Dropdown(
                                    id='agent-select-dropdown'
                                ),
                                
                                html.Hr(), 

                                dbc.Button(
                                    "Create New Itinerary for Agent",
                                    id="create-single-itinerary-button",
                                    color="primary",
                                    style={"margin": "5px"},
                                ), 

                                html.Hr(), 

                                html.H5("Current Itinerary"),
                                html.Pre(id='itinerary-data-display', style=styles['pre']),
                                html.Hr(), 
                                html.P("", id="itinerary-creation-msg"),
                                html.Hr(), 
                            ]
                            ), 

                            dcc.Tab(label='3. Add Agent Movement', children=[
                                
                                html.Hr(), 

                                html.H5("Select Location"),
                                html.P("Click buildings on the map to select a location."),
                                html.Pre(id='click-data', style=styles['pre']),

                                html.Hr(), 
                                
                                html.H5("Select Movement Type"),

                                dcc.Tabs([
                                    dcc.Tab(label='Start', children=[
                                        html.Hr(),
                                        html.H6("START Movement Description"),
                                        html.Hr(),
                                        html.P("State agent must reach before other instructions can occur. Must be the first and only first movement for any agent's itinerary."),
                                        
                                        html.H6("Select Location"),
                                        html.P("First, please select a location on the right to indicate START location."),
                                        
                                        html.H6("Select Date and Time"),
                                        html.P("Now, please select a date and time range to indicate the time frame which the agent must be at the location. "),
                                        dcc.DatePickerRange(
                                            id='date-picker-range',
                                            min_date_allowed=date(2020, 1, 1),
                                            max_date_allowed=date(2023, 12, 31),
                                            initial_visible_month=date(2023, 3, 1), 
                                            start_date=date(2023, 3, 1), 
                                            end_date=date(2023, 3, 1),
                                            minimum_nights=0
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        
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
                                        html.Br(),
                                        
                                        html.P(id='date-picker-range-output'),

                                        dbc.Button(
                                            "Add Start Movement to Current Itinerary",
                                            id="add-start-movement-button",
                                            color="primary",
                                            style={"margin": "5px"},
                                        ), 

                                    ]),
                                    dcc.Tab(label='Move', children=[
                                        html.Hr(),
                                        html.H6("MOVE Movement Description"),
                                        html.Hr(),
                                        html.P("Location agent must reach and mode of transportation to take."),
                                        html.H6("Select Location"), 
                                        html.P("First, please select a location on the map to the right to indicate where the agent must target. "),
                                        html.H6("Select Transportation Mode"),
                                        html.P("Next, please select the transportation mode used to reach the defined location. "),

                                        dcc.Dropdown(
                                            TRANSPORT_MODES, 
                                            TRANSPORT_MODES[0],
                                            id='transport-mode-dropdown'
                                        ),

                                        html.Br(), 

                                        dbc.Button(
                                            "Add Move Movement to Current Itinerary",
                                            id="add-move-movement-button",
                                            color="primary",
                                            style={"margin": "5px"},
                                        ), 
                                    ]),
                                    dcc.Tab(label='Stay', children=[
                                        html.Hr(),
                                        html.H6("STAY Movement Description"),
                                        html.Hr(), 

                                        html.P("Period of time to stay at current location; duration, end_time, or both can be provided; if both are provided, priority should also be provided."),
                                        
                                        html.H6("Add Time Constraint"),
                                        dcc.Checklist(
                                            ['Duration', 'End Time'],
                                            ['Duration'], 
                                            id='duration-endtime-priority-checklist'), 
                                        html.Br(), 
                                        html.H6("If both duration and end time are provided, which one has priority?"),
                                        dcc.Dropdown(['Duration', 'End Time'], 
                                                    'Duration', 
                                                    id='priority-dropdown'),
                                        
                                        html.Hr(),

                                        html.H6("Duration Selection"),
                                        html.P("Please input the duration (in minutes) of the stay below."),
                                        dcc.Input(id="duration-stay-input", type="number"),
                                        
                                        html.Hr(),

                                        html.H6("End Time Selection"),
                                        dcc.DatePickerSingle(
                                            id='end-time-date-picker-range',
                                            min_date_allowed=date(2020, 1, 1),
                                            max_date_allowed=date(2023, 12, 31),
                                            initial_visible_month=date(2023, 3, 1), 
                                            date=date(2023, 3, 1)
                                        ),

                                        html.Br(),
                                        html.Br(),

                                        dcc.Slider(
                                            id="stay-end-time-slider",
                                            min=0,
                                            max=24,
                                            step=0.25,
                                            value=0,
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

                                        dbc.Button(
                                            "Add Stay Movement to Current Itinerary",
                                            id="add-stay-movement-button",
                                            color="primary",
                                            style={"margin": "5px"},
                                        ), 
                                    ]),
                                ]), 

                                html.P(id='add-movement-msg'), 

                                html.Hr(), 

                            ]
                            ), 

                            dcc.Tab(label='4. Export JSON', children=[
                                html.Hr(),
                                html.H5("Current JSON Data"),
                                html.Pre(id='json-data-display', style=styles['pre']),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Save HAS JSON",
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
                            # html.Hr(),

                            html.H5("Region Selection", 
                                style={'padding-left': 80}),

                            dcc.Dropdown(
                                subzone_list,
                                subzone,
                                id='subzone', 
                                searchable=True,
                                style={'width': 600, 'padding-left': 80}
                            ),

                            dbc.Spinner(
                                dcc.Graph(figure=fig, id="display", style={"height": "80vh"}),
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

    """
        Plots the map as a Plotly express widget. 

        :param subzone: Level 4 subzone to filter for. 
                This is typically passed in through the UI dropdown menu. 
                Subzone names can be found in the level 4 geojson file that 
                this app is dependent on.
        :type subzone: str

        :return:
            - fig 
            - admin_df
            - poi_df
        :rtype: tuple(plotly.graph_objects.Figure, geopandas.GeoDataFrame, geopandas.GeoDataFrame)
    """

    if json_data is not None:
        shape_list = [] 

        for event in json_data["events"]:
            
            location_dict = event["location"]["geometries"]

            for shapes in location_dict:
            
                poly_data = np.array(shapes["coordinates"][0])
                hole_data = shapes["coordinates"][1:] if len(shapes["coordinates"]) > 1 else None

                shapely_poly = Polygon(poly_data, holes=hole_data)
                shape_list.append(shapely_poly)

        hos_poly_gpd = gpd.GeoDataFrame(shape_list, columns=["geometry"], crs='EPSG:4326')
        
        subzone = infer_subzone_from_poly(hos_poly_gpd.iloc[0]["geometry"])
        print("Inferred subzone: %s" % subzone)

    admin_df = gpd.read_file(ADMIN_PATH, 
                            geometry="geometry", 
                            crs='EPSG:4326')

    admin_df['LogDensity'] = np.log10(admin_df['Census Population'] / admin_df['Land Area'])

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
                            center = {'lat': subzone_geom_center.y, 'lon': subzone_geom_center.x},
                            opacity=0.6, 
                            color_continuous_scale="reds",
                            hover_data=[within_subzone.index, 'id_building', 'id_road', "length_along_edge",'h3'],
                )


    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(clickmode='event+select')

    if json_data is not None:
        for i, gdf_shape in hos_poly_gpd.iterrows():
            xs,ys = gdf_shape["geometry"].exterior.coords.xy
            xs = np.array(xs)
            ys = np.array(ys)
            fig.add_trace(go.Scattermapbox(lon=xs, lat=ys, mode="lines", fill="toself", line=dict(color='red')))


    return fig, admin_df, poi_df, subzone

def get_event_list():
    """
        Gets the list of events from HOS json data.

        :return: list of event uuids in string format
        :rtype: List[str]
    """

    event_options = [i["event_uid"] for i in json_data["events"]]
    return event_options

def load_hos_data():
    """
        Loads HOS data from globally defined HOS file. 

        :return: HOS data in Python dict format
        :rtype: Dict
    """

    assert HOS_FILE is not None, "HOS file not initialized"

    with open(HOS_FILE, 'r') as f:
        hos_data = json.load(f)

    return hos_data
    
def get_agent_list_from_hos():
    """
        Gets a list of agent IDs from HOS file. 

        :return: List of agent IDs (integers)
        :rtype: List[int]
    """

    global json_data 
    
    agents = [] 

    for event in json_data["events"]:

        agents = agents + event["agents"]
    
    agents = set(agents)
    agents = sorted(list(agents))
    
    return agents

def infer_subzone_from_poly(polygon : Polygon) -> str:
    """
    Infers the name of the subzone containing the provided polygon.

    :params num: query polygon
    :type num: shapely.Polygon

    :return: subzone name
    :rtype: str
    """

    admin_df = gpd.read_file(ADMIN_PATH, 
                    geometry="geometry", 
                    crs='EPSG:4326')
    
    for _, row in admin_df.iterrows():
        
        subzone_geom = row["geometry"]

        if polygon.within(subzone_geom): 

            return row["Name"]
    
    return None


def number_to_time(num : float) -> str:
    """
    Converts a number of hours to a formatted time of HH:MM:00.

    :params num: number of hours
    :type num: float

    :return: formatted time as HH:MM:00.
    :rtype: str
    """

    hours, minutes = number_to_hours_mins(num)
    formatted_time = "%02d:%02d:00" % (hours,minutes)
    return formatted_time

def number_to_hours_mins(num : float) -> Tuple[int]:
    """
        Converts a float number input in units of hours to hours and minutes.

        :param num: number of hours 
        :type num: float
        
        :returns:
            - hours: number of hours 
            - minute: number of minutes
        :rtype: tuple(int, int)
    """

    minutes = int((num - math.floor(num)) * 60)
    hours = int(num)

    return hours, minutes

def agent_itinerary_exists(agent_id):
    """
    Checks if an agent itinerary already exists and returns a boolean.

    :params agent_id: query agent ID, typically enumerated and not a UUID
    :type agent_id: int

    :return: True if itinerary already exists in HAS data, False if not
    :rtype: bool
    """

    global has_data 

    if has_data is None: 
        return False 
    else: 

        for mvmt in has_data["movements"]:

            if mvmt["agent"] == agent_id: 
                return True 

        return False

def get_geojson_from_clickdata(clickdata):
    """
    Gets GEOJSON data from Plotly Dash clickdata. 
    Clickdata is the data stored when a user clicks a polygon on the map.

    :params clickdata: data from Plotly interactive plot click
    :type clickdata: dict

    :return: geojson dictionary representing the polygon clicked
    :rtype: dict
    """

    location = json.loads(clickdata)

    if location is None: 
        return None 

    geometry_info_i = location["points"][0]["customdata"][0]

    global plotted_df
    location_polygon = plotted_df.iloc[geometry_info_i]["building_poly"]

    geojson_collection = GeometryCollection([location_polygon])

    geojson_dict = json.loads(gpd.GeoSeries(geojson_collection).set_crs('EPSG:4326').to_json())
    geojson_dict = geojson_dict['features'][0]['geometry']

    return geojson_dict

## CALLBACK FUNCTIONS FOR PLOTLY DASH BELOW

@callback(
    Output('add-movement-msg', 'children', allow_duplicate=True), 
    Input('add-stay-movement-button', 'n_clicks'), 
    State('duration-endtime-priority-checklist', 'value'), 
    State('priority-dropdown', 'value'), 
    State('end-time-date-picker-range', 'date'), 
    State('stay-end-time-slider', 'value'), 
    State('duration-stay-input', 'value'), 
    prevent_initial_call=True
)
def create_stay_movement(n_clicks, priority_checklist, priority_choice, endtime_date_picker, endtime_time_slider, duration):

    if len(priority_checklist) == 0: 
        return "Failed. Please select at least one choice [\"duration\", \"end_time\"] to define the time constraint for this action. "

    if duration is None and "Duration" in priority_checklist: 
        return "Failed. Duration was selected and no duration value was provided. Please try again. "
    
    instruction_uid = uuid.uuid4()

    global current_movement 

    if current_movement is None: 
        return "Failed. Please initialize an itinerary first, above."

    if len(current_movement['itineraries'][-1]['itinerary']) == 0:
        return "Failed. MOVE movements may not be the first movement in a single itinerary. Please add a START movement first."

    # process duration here; duration_iso
    duration_iso = pd.Timedelta(minutes=float(duration)).isoformat()

    # process end time here; endtime_iso 
    endtime_date = date.fromisoformat(endtime_date_picker)
    endtime_date = datetime.combine(endtime_date, datetime.min.time())
    hours, minutes = number_to_hours_mins(endtime_time_slider)
    endtime_date = endtime_date + timedelta(hours=hours, minutes=minutes)
    endtime_iso = endtime_date.isoformat()

    # process priority here: priority
    priority = priority_choice.lower().replace(" ", "_")
        
    stay_mvmt_data = {
        "stay": {
            "instruction_uid" : str(instruction_uid), 
        }
    }

    for item in priority_checklist:

        if item.lower() == "duration": 
            stay_mvmt_data["stay"]["duration"] = duration_iso 
        
        if item.lower() == "end time": 
            stay_mvmt_data["stay"]["end_time"] = endtime_iso

    if len(priority_checklist) == 2: 

        stay_mvmt_data["stay"]["priority"] = priority
    
    current_movement['itineraries'][-1]['itinerary'].append(stay_mvmt_data)
    
    return "STAY Movement created with ID %s" % str(instruction_uid)


@callback(
    Output('add-movement-msg', 'children', allow_duplicate=True), 
    Input('add-move-movement-button', 'n_clicks'), 
    State('click-data', 'children'), 
    State('transport-mode-dropdown', 'value'),
    prevent_initial_call=True
)
def create_move_movement(n_clicks, click_data, transportation_mode):
    
    if click_data is None: 
        return "Failed. Please select a location on the map to the right."
    
    instruction_uid = uuid.uuid4()

    location_data = get_geojson_from_clickdata(click_data)

    if location_data is None: 
        return "Failed. Please select a location on the map to the right."
    
    global current_movement 

    if current_movement is None: 
        return "Failed. Please initialize an itinerary first, above."

    if len(current_movement['itineraries'][-1]['itinerary']) == 0:
        return "Failed. MOVE movements may not be the first movement in a single itinerary. Please add a START movement first."

    move_mvmt_data = {
        "move": {
            "instruction_uid" : str(instruction_uid), 
            "location" : location_data, 
            "transportation_mode" : transportation_mode
        }
    }

    current_movement['itineraries'][-1]['itinerary'].append(move_mvmt_data)
    
    return "MOVE Movement created with ID %s" % str(instruction_uid)

@callback(
    Output('add-movement-msg', 'children', allow_duplicate=True), 
    Input('add-start-movement-button', 'n_clicks'), 
    State('click-data', 'children'), 
    State('date-picker-range', 'start_date'), 
    State('date-picker-range', 'end_date'), 
    State('time_frame', 'value'),
    prevent_initial_call=True
)
def create_start_movement(n_clicks, click_data, start_date, end_date, time_frame):
    
    if click_data is None: 
        return "Failed. Please select a location on the map to the right."

    instruction_uid = uuid.uuid4()

    location_data = get_geojson_from_clickdata(click_data)

    if location_data is None: 
        return "Failed. Please select a location on the map to the right."
    
    global current_movement 

    if current_movement is None: 
        return "Failed. Please initialize an itinerary first, above."

    if len(current_movement['itineraries'][-1]['itinerary']) > 0:
        return "Failed. START movements may only be the first movement in a single itinerary. The current itinerary is non-empty."

    start_day = date.fromisoformat(start_date)
    end_day = date.fromisoformat(end_date)

    start_day = datetime.combine(start_day, datetime.min.time())
    end_day = datetime.combine(end_day, datetime.min.time())

    value = [number_to_hours_mins(v) for v in time_frame]

    begin_date = start_day + timedelta(hours=value[0][0], minutes=value[0][1])
    end_date= end_day + timedelta(hours=value[1][0], minutes=value[1][1])

    begin_str = begin_date.isoformat("T","auto") 
    end_str = end_date.isoformat("T","auto") 

    time_window_data = {
        "begin": begin_str, 
        "end": end_str
    }

    start_mvmt_data = {
        "start": {
            "instruction_uid" : str(instruction_uid), 
            "location" : location_data, 
            "time_window" : time_window_data
        }
    }
    
    current_movement['itineraries'][-1]['itinerary'].append(start_mvmt_data)

    return "START Movement created with ID %s" % str(instruction_uid)

@callback(
    Output('date-picker-range-output', 'children'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('time_frame', 'value'),)
def update_output(start_date, end_date, time_frame):

    value = [number_to_hours_mins(v) for v in time_frame]

    start_date_object = date.fromisoformat(start_date)
    start_date_object = datetime.combine(start_date_object, datetime.min.time())
    start_date_object = start_date_object + timedelta(hours=value[0][0], minutes=value[0][1])

    start_date_string = start_date_object.strftime('%Hh:%Mm, %B %d, %Y')
    string_prefix = start_date_string + ' ==> '

    end_date_object = date.fromisoformat(end_date)
    end_date_object = datetime.combine(end_date_object, datetime.min.time())
    end_date_object = end_date_object + timedelta(hours=value[1][0], minutes=value[1][1])
    
    end_date_string = end_date_object.strftime('%Hh:%Mm, %B %d, %Y')
    string_prefix = string_prefix + end_date_string
    
    return string_prefix

@callback(
    Output('itinerary-creation-msg', 'children', allow_duplicate=True), 
    Input('create-single-itinerary-button', 'n_clicks'), 
    State('agent-select-dropdown', 'value'), 
    prevent_initial_call=True
)
def create_itinerary(n_clicks, agent_select_value):
    
    global has_data 
    global current_movement

    if has_data is None:
        return "HAS data has not been initialized yet; please initialize one in tab 1 above."
    if agent_select_value is None:
        return "Please select an agent from the dropdown."
    
    agent_id = int(agent_select_value)

    if agent_itinerary_exists(agent_id): 
        
        for mvmt in has_data["movements"]:
            if mvmt["agent"] == agent_id: 
                mvmt["itineraries"].append({"itinerary" : []})
                return "Appended new itinerary for agent %d" % agent_id

    else:
        movement = bhj.create_movement(agent_id)
        movement["itineraries"].append({"itinerary": []})
        has_data["movements"].append(movement)

        current_movement = has_data["movements"][-1]

        return "Initialized data and created first itinerary for agent %d" % agent_id

@callback(
    Output('selected-hos-name-msg', 'children'),
    Input('objective-path-upload', 'filename'),
)
def update_selected_hos_msg(filename):

    return "Selected file: %s" % filename

@callback(
    Output('itinerary-data-display', 'children'),
    Input('json-data-display', 'children'), 
    Input('add-movement-msg', 'children')
)
def update_itinerary_info(filename, mvmt_msg):

    global current_movement
    current_itinerary = None 

    if current_movement is not None:
        current_itinerary = current_movement['itineraries'][-1]['itinerary'].copy()

        for it in current_itinerary:
            
            if "start" in it.keys():
                it["start"]["location"] = {}
            if "move" in it.keys(): 
                it["move"]["location"] = {}

    return json.dumps(current_itinerary, indent=2)

@callback(
    Output('json-data-display', 'children', allow_duplicate=True),  
    Input('display', 'figure'), 
    Input('create-single-itinerary-button', 'n_clicks'), 
    Input('add-movement-msg', 'children'),
    prevent_initial_call=True)
def update_json_info(event_data, display2, mvmt_msg): 
    global has_data
    return json.dumps(has_data, indent=2)

@callback(
    Output('unfulfilled-time-objectives-display', 'children', allow_duplicate=True),  
    Output('unfulfilled-duration-objectives-display', 'children', allow_duplicate=True), 
    Input('add-movement-msg', 'children'),
    Input('json-data-display', 'children'),
    prevent_initial_call=True)
def update_constraint_info(mvmt_msg, upload_contents): 
    global events_by_agent_data
    global event_data
    global time_constraint_data
    global duration_constraint_data
    global json_data

    event_data = json_data["events"].copy() 
    time_constraint_data = json_data["time_constraints"].copy()
    duration_constraint_data = json_data["duration_constraints"].copy()
    
    # process event data by agent 
    agent_event_data = {}
    
    for event in event_data: 

        for agent in event["agents"]:

            if agent not in agent_event_data.keys():
                agent_event_data[agent] = [event["event_uid"]] 
            else: 
                agent_event_data[agent].append(event["event_uid"])


    df_data = []
    for key, value in agent_event_data.items():
        df_data.append({
            "agent" : key, 
            "events" : value
        })

    events_by_agent_data = pd.DataFrame.from_dict(df_data)

    # process time constraint data 
    new_time_constraint_data = []

    for data in time_constraint_data: 
        
        new_data = {}
        new_data["event_uid"] = data["event"]
        new_data["time_window_begin"] = data["time_window"]["begin"]
        new_data["time_window_end"] = data["time_window"]["end"]
        new_time_constraint_data.append(new_data)

    time_constraint_data = pd.DataFrame.from_dict(new_time_constraint_data)

    # process duration constraint data 
    new_duration_constraint_data = []

    for data in duration_constraint_data: 
        
        new_data = {}
        new_data["first_event_uid"] = data["events"]["first"]
        new_data["second_event_uid"] = data["events"]["second"]
        new_data["duration_window_minimum"] = data["duration_window"]["minimum"]
        new_data["duration_window_maximum"] = data["duration_window"]["maximum"]
        new_data["stay"] = data["stay"]
        new_duration_constraint_data.append(new_data)

    duration_constraint_data = pd.DataFrame.from_dict(new_duration_constraint_data)


    event_data = pd.DataFrame.from_dict(json_data["events"]) 

    return time_constraint_data.to_string(), duration_constraint_data.to_string()

@callback(
    Output('create-movement-msg', 'children'), 
    Output('json-data-display', 'children'), 
    Output('agent-select-dropdown', 'options'), 
    Output('subzone', 'value'),
    Input('create-movement-button', 'n_clicks'), 
    State('objective-path-upload', 'contents'), 
    State('objective-path-upload', 'filename'),
    prevent_initial_call=True
)
def initialize_movement(n_clicks, obj_contents, filename):
    
    try:
        global json_data
        content_type, content_string = obj_contents.split(",")
        decoded = base64.b64decode(content_string)
        json_data = json.loads(decoded)
    except: 
        return "HOS file (%s) is invalid! Please try again. " % filename, "", [], "None"

    objective_uid = json_data["objective_uid"]

    global has_data 
    has_data = bhj.initialize_empty_has(objective_uid) 

    agent_list = get_agent_list_from_hos() 

    return "Successfully initialized HAS with Objective ID: %s" % objective_uid, json.dumps(has_data, indent=2), agent_list, "None"

@callback(
    Output('click-data', 'children'),
    Input('display', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@callback(
    Output('json-export', 'children'),
    Input('generate_json', 'n_clicks'))
def export_json(n_clicks):

    global json_data 
    if json_data is None: 
        return "JSON Data is currently none, please create one and retry exporting" 

    if n_clicks is None or n_clicks < 0: 
        return 'JSON not saved yet'

    filepath = 'has_output.json'

    with open(filepath, 'w') as fp:
        json.dump(json_data, fp, indent=4)
    
    return 'Exported JSON to %s' % filepath

        
@callback(
Output('display', 'figure'),
Input('json-data-display', 'children'),
Input('subzone', 'value'), 
prevent_initial_call=True)
def update_graph(display_data, subzone_name):

    fig, _, _, _ = plot_map(subzone_name)
    return fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Hide Activity Specification Web Application',
                    description='Web application for generating sample \
                        hide activity specifications based on the JSON \
                        schema given to by the government for HAYSTAC.')
    
    parser.add_argument('-p', '--path', 
                        required=True, 
                        help="Path containing supporting map and building \
                            files for web app visualization.")
    
    args = parser.parse_args()

    POI_PICKLE = os.path.join(args.path, "poi.pkl")
    ADMIN_PATH = os.path.join(args.path, "singapore_planning_area_lvl4.geojson")
    HOS_FILE = None

    build_app()