import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import base64
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from PIL import Image

def get_my_format(sightings_df):
    # Convert the 'Timestamp' column to datetime format
    sightings_df['Timestamp'] = pd.to_datetime(sightings_df['Timestamp'])

    # Extract year and month from the timestamp
    sightings_df['Year'] = sightings_df['Timestamp'].dt.year
    sightings_df['Month'] = sightings_df['Timestamp'].dt.month

    sightings_df = sightings_df.set_index('Timestamp')

    return(sightings_df)

def group_daily_sigthings(sightings_df):

    # Resample the data by day, grouping by 'Animal' and 'Camera' and counting the number of sightings
    daily_sightings = sightings_df.groupby(['Animal', 'Camera']).resample('D').size().reset_index(name='Count')

    # Rename the 'Timestamp' column to 'Date'
    daily_sightings.rename(columns={'Timestamp': 'Date'}, inplace=True)

    return(daily_sightings)

# def generate_timeline_sightings(daily_sightings):
#     # Get the unique cameras and animals
#     cameras = daily_sightings['Camera'].unique()
#     animals = daily_sightings['Animal'].unique()
    
#     # Create a figure
#     fig = go.Figure()

#     # Create a trace for each animal from each camera
#     for animal in animals:
#         for camera in cameras:
#             df = daily_sightings[(daily_sightings['Animal'] == animal) & (daily_sightings['Camera'] == camera)]
#             fig.add_trace(go.Scatter(
#                 x=df['Date'],
#                 y=df['Count'],
#                 name=animal,
#                 legendgroup=animal,
#                 visible='legendonly' if camera != cameras[0] else True,  # Show the first camera's data by default
#                 hovertemplate=f'{animal}<br>Camera: {camera}<br>Date: %{{x}}<br>Sightings: %{{y}}<extra></extra>'
#             ))

#     # Create the dropdown menu
#     dropdown = [{'label': camera, 'method': 'update', 'args': [{'visible': [camera in fig.data[i].hovertemplate for i in range(len(fig.data))]}]} for camera in cameras]
#     dropdown.append({'label': 'All Cameras', 'method': 'update', 'args': [{'visible': [True]*len(fig.data)}]})

#     # Update the layout
#     fig.update_layout(
#         title='Animal Sightings by Camera',
#         xaxis_title='Date',
#         yaxis_title='Number of Sightings',
#         # hovermode='x unified',  # Show a single hover label with all the information
#         updatemenus=[{'type': 'dropdown', 'showactive': True, 'buttons': dropdown, 'x': 1.1, 'y': 1.2}]
#     )

    
#     # Return the figure
#     return fig

animal_images = {
    'Ducks': '../images/Ducks.png',
    'Otter': '../images/Otter.png',
    'Squirrel': '../images/Squirrel.png',
    'Pigeon': '../images/Pigeon.png',
    'Mouse': '../images/Mouse.png'
}
sightings_df = pd.read_csv('../data/dummysightings.csv')
sightings_df = get_my_format(sightings_df)
daily_sightings = group_daily_sigthings(sightings_df)
cameras = sorted(sightings_df.Camera.unique())
cameras = np.append(cameras, 'All Cameras')
animals = sorted(sightings_df.Animal.unique())
years = sorted(sightings_df.Year.unique())

cameras_islands = base64.b64encode(open('../Images/cameraplacement.png', 'rb').read())
otter_logo = base64.b64encode(open('../Images/logo.png', 'rb').read())

# Define the animal class options for the dropdown menu
animal_classes = list(animal_images.keys())
dropdown_options = [{'label': cls, 'value': cls} for cls in animal_classes]

# fig = generate_timeline_sightings(sightings_df)
app = dash.Dash(external_stylesheets=[dbc.themes.COSMO], title="DarWild Sightings")

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
        # html.H2("DarWild Sightings", className="display-4"),
        html.Img(src='data:image/png;base64,{}'.format(otter_logo.decode()), style={'width':200}),
        html.Hr(),
        html.P(
            "Explore interactive plots of animal sightings in Darwin College.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Sighting of the Week", href="/page-1", active="exact"),
                dbc.NavLink("More in Depth Analysis", href="/page-2", active="exact"),
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
                html.Img(src='data:image/png;base64,{}'.format(cameras_islands.decode()), style={'width':'60%'}),
                dcc.Dropdown(
                    id='camera-dropdown',
                    options=[{'label': camera, 'value': camera} for camera in cameras],
                    value=cameras[3]
                ),
                dcc.Graph(id='sightings-graph'),
                # dcc.Dropdown(
                #     id='aggregation-dropdown',
                #     options=[
                #         {'label': 'Week', 'value': 'W'},
                #         {'label': 'Month', 'value': 'M'},
                #         {'label': 'Year', 'value': 'Y'}
                #     ],
                #     placeholder='Select a time aggregation level'
                # ),
                # dcc.Dropdown(
                #     id='period-dropdown',
                #     placeholder='Select a specific time period'
                # ),
                # dcc.Graph(id='sightings-barplot')
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date_placeholder_text="Start Period",
                    end_date_placeholder_text="End Period",
                    calendar_orientation='vertical',
                    display_format='D/M/Y',
                    min_date_allowed=sightings_df.index.min().date(),
                    max_date_allowed=sightings_df.index.max().date(),
                ),
                dcc.Graph(id='period-bar-plot')
            ]
        )
        return home
    elif pathname == "/page-1":
        page1 = html.Div(
            [
                html.H2('Sightings of the Week'),
                html.P('Select your favourite animal.'),
                dcc.Dropdown(
                    id='animal-dropdown',
                    options=dropdown_options,
                    value=animal_classes[0]  # Set the default selected value
                ),
                html.Div(id='image-container')
            ]
        )
        return page1
    elif pathname == "/page-2":
        page2 = html.Div(
            [
                html.H2('In Depth Analysis'),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': i, 'value': i} for i in years],
                    value=np.max(years),
                    style={'width': '50%'}
                ),
                dcc.Graph(id='animal-month-comp'),
                dcc.DatePickerRange(
                    id='date-picker-radar',
                    start_date_placeholder_text="Start Period",
                    end_date_placeholder_text="End Period",
                    calendar_orientation='vertical',
                    display_format='D/M/Y',
                    min_date_allowed=sightings_df.index.min().date(),
                    max_date_allowed=sightings_df.index.max().date(),
                ),
                dcc.Graph(id='radar-plot')
            ]
        )
        return page2
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
    Output('period-bar-plot', 'figure'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graph(start_date, end_date):
    if start_date is not None and end_date is not None:
        mask = (sightings_df.index >= start_date) & (sightings_df.index <= end_date)
        filtered_df = sightings_df.loc[mask]
        counts = filtered_df['Animal'].value_counts()
        fig = go.Figure([go.Bar(x=counts.index, y=counts.values)])
    else:
        fig = go.Figure()
    fig.update_layout(
        title='Animal Sightings',
        xaxis_title='Animal',
        yaxis_title='Number of Sightings'
    )
    return fig

@app.callback(
    Output('radar-plot', 'figure'),
    [Input('date-picker-radar', 'start_date'),
     Input('date-picker-radar', 'end_date')]
)
def update_graph(start_date, end_date):
    if start_date is not None and end_date is not None:
        mask = (sightings_df.index >= start_date) & (sightings_df.index <= end_date)
        filtered_df = sightings_df.loc[mask]
    else:
        filtered_df = sightings_df

    hourly_sightings = filtered_df.groupby(filtered_df.index.hour)['Animal'].value_counts().unstack().fillna(0)
    print(hourly_sightings.index.max())
    fig = go.Figure()

    for animal in hourly_sightings.columns:
        fig.add_trace(go.Scatterpolar(
            r=hourly_sightings[animal],
            theta=hourly_sightings.index * 360 / 24,  # Convert hours to degrees,
            fill='toself',
            name=animal
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, hourly_sightings.values.max()]
            ),
            angularaxis=dict(  # Add this to set the radial labels to hours
                tickvals=list(range(0, 360, 15)),  # Every 15 degrees
                ticktext=list(range(0, 24, 1)),  # Every hour
                direction='clockwise',  # Change the direction of rotation
                rotation=90  # Rotate the plot to position 0-hour at the top
            )
            ),
        showlegend=True,
        title="Distribution of sightings over 24h" 
    )

    return fig

@app.callback(
    Output('animal-month-comp', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_month_comp(selected_year):
    dff = sightings_df[(sightings_df['Year'] == selected_year)]
    dff = dff.groupby(['Month', 'Animal']).size().reset_index(name='Count')
    
    fig = go.Figure()
    for animal in animals:
        df_animal = dff[dff['Animal'] == animal]
        fig.add_trace(go.Scatter(
            x=df_animal['Month'],
            y=df_animal['Count'],
            name=animal
        ))
    
    fig.update_layout(
        title=f'Animal Sightings in {selected_year}',
        xaxis_title='Month',
        yaxis_title='Number of Sightings',
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    )
    
    return fig

@app.callback(
    Output('sightings-graph', 'figure'),
    [Input('camera-dropdown', 'value')]
)
def update_camera_scatterplot(selected_camera):
    fig = go.Figure()
    if selected_camera == 'All Cameras':
        tmp = daily_sightings.groupby(['Date', 'Animal'])['Count'].agg('sum').reset_index(name='Count')
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
        # hovermode='x unified'  # Show a single hover label with all the information
    )
    
    return fig

@app.callback(
    Output('period-dropdown', 'options'),
    [Input('aggregation-dropdown', 'value')]
)
def update_period_dropdown(selected_aggregation):
    if selected_aggregation is None:
        return []
    else:
        # Resample the data by the selected aggregation level
        aggregated_sightings = sightings_df.groupby('Animal').resample(selected_aggregation).size()
        
        # Generate the dropdown options based on the unique periods
        return [{'label': str(period), 'value': str(period)} for period in aggregated_sightings.index.get_level_values(1).unique()]

@app.callback(
    Output('sightings-barplot', 'figure'),
    [Input('aggregation-dropdown', 'value'),
     Input('period-dropdown', 'value')]
)
def update_barplot(selected_aggregation, selected_period):
    if selected_aggregation is None or selected_period is None:
        # If no aggregation level or period is selected, show the total number of sightings per animal
        counts = sightings_df['Animal'].value_counts()
        fig = go.Figure([go.Bar(x=counts.index, y=counts.values)])
    else:
        # Resample the data by the selected aggregation level
        aggregated_sightings = sightings_df.groupby('Animal').resample(selected_aggregation).size().reset_index(name='Count')

        # Filter the data for the selected period
        filtered_sightings = aggregated_sightings.loc[aggregated_sightings['Timestamp']==selected_period, :]

        # Create the bar plot
        fig = go.Figure([go.Bar(x=filtered_sightings.Animal, y=filtered_sightings.Count)])
    
    fig.update_layout(
        title='Animal Sighting in Selected Timeframe',
        xaxis_title='Animal',
        yaxis_title='Number of Sightings'
    )
    
    return fig

# Define the callback function to update the displayed image based on the dropdown selection
@app.callback(
    Output('image-container', 'children'),
    [Input('animal-dropdown', 'value')]
)
def update_image(animal_class):
    # Get the corresponding image for the selected animal class
    image = animal_images[animal_class]

    # Convert the numpy array to a PIL Image
    pil_image = Image.open(image)

    # Create the image element to display in the dashboard
    image_element = html.Img(src=pil_image, style={'width': '50%'})

    return image_element

if __name__ == "__main__":
    app.run_server()