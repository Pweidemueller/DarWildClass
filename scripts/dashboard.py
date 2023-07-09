import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import plotly.graph_objects as go

import pandas as pd
import numpy as np

def group_date_sigthings(sightings_df):
    # Convert the 'Timestamp' column to datetime format
    sightings_df['Timestamp'] = pd.to_datetime(sightings_df['Timestamp'])

    # Set the 'Timestamp' column as the DataFrame's index
    sightings_df.set_index('Timestamp', inplace=True)

    # Resample the data by day, grouping by 'Animal' and 'Camera' and counting the number of sightings
    daily_sightings = sightings_df.groupby(['Animal', 'Camera']).resample('D').size().reset_index(name='Count')

    # Rename the 'Timestamp' column to 'Date'
    daily_sightings.rename(columns={'Timestamp': 'Date'}, inplace=True)

    return(daily_sightings)

def generate_timeline_sightings(daily_sightings):
    # Get the unique cameras and animals
    cameras = daily_sightings['Camera'].unique()
    animals = daily_sightings['Animal'].unique()
    
    # Create a figure
    fig = go.Figure()

    # Create a trace for each animal from each camera
    for animal in animals:
        for camera in cameras:
            df = daily_sightings[(daily_sightings['Animal'] == animal) & (daily_sightings['Camera'] == camera)]
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Count'],
                name=animal,
                legendgroup=animal,
                visible='legendonly' if camera != cameras[0] else True,  # Show the first camera's data by default
                hovertemplate=f'{animal}<br>Camera: {camera}<br>Date: %{{x}}<br>Sightings: %{{y}}<extra></extra>'
            ))

    # Create the dropdown menu
    dropdown = [{'label': camera, 'method': 'update', 'args': [{'visible': [camera in fig.data[i].hovertemplate for i in range(len(fig.data))]}]} for camera in cameras]
    dropdown.append({'label': 'All Cameras', 'method': 'update', 'args': [{'visible': [True]*len(fig.data)}]})

    # Update the layout
    fig.update_layout(
        title='Animal Sightings by Camera',
        xaxis_title='Date',
        yaxis_title='Number of Sightings',
        # hovermode='x unified',  # Show a single hover label with all the information
        updatemenus=[{'type': 'dropdown', 'showactive': True, 'buttons': dropdown, 'x': 1.1, 'y': 1.2}]
    )

    
    # Return the figure
    return fig

app = dash.Dash(external_stylesheets=[dbc.themes.COSMO])
sightings_df = pd.read_csv('../data/dummysightings.csv')
daily_sightings = group_date_sigthings(sightings_df)
cameras = sorted(sightings_df.Camera.unique())
cameras = np.append(cameras, 'All Cameras')
animals = sorted(sightings_df.Animal.unique())

# fig = generate_timeline_sightings(sightings_df)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("DarWild Sightings", className="display-4"),
        html.Hr(),
        html.P(
            "Explore interactive plots of animal sightings in Darwin College.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Footer("Author: Paula Weidemüller")
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        home = html.Div(
            [
                html.H2("Welcome to DarWild Sightings Dashboard"),
                html.P("At Darwin college we have three wild life cameras, that capture footage every time they register movement. Every week the footage gets analysed and animals in the clips are classified automatically using neural networks."),
                dcc.Dropdown(
                    id='camera-dropdown',
                    options=[{'label': camera, 'value': camera} for camera in cameras],
                    value=cameras[0]
                ),
                dcc.Graph(id='sightings-graph')
            ]
        )
        return home
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )
@app.callback(
    Output('sightings-graph', 'figure'),
    [Input('camera-dropdown', 'value')]
)
def update_graph(selected_camera):
    fig = go.Figure()
    if selected_camera == 'All Cameras':
        tmp = daily_sightings.groupby(['Date', 'Animal'])['Count'].agg('sum').reset_index(name='Count')
        print(tmp.head())
    for animal in animals:
        if selected_camera == 'All Cameras':
            df = tmp[(tmp['Animal'] == animal)]
        else:
            df = daily_sightings[(daily_sightings['Animal'] == animal) & (daily_sightings['Camera'] == selected_camera)]
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Count'],
            name=animal
        ))
    
    fig.update_layout(
        title='Animal Sightings by Camera',
        xaxis_title='Date',
        yaxis_title='Number of Sightings',
        hovermode='x unified'  # Show a single hover label with all the information
    )
    
    return fig

if __name__ == "__main__":
    app.run_server()