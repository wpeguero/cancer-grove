"""The third Iteration of the Capstone Project Application.

After failing to create the offline GUI through the use of
Tkinter library and matplotlib, an online version of the
same GUI has been replicated using the Dash library. This
library uses the plotly library on the backend to develop
plots and display them within html that the Dash library
develops.
"""
import base64
import datetime
import io

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State

import plotly.express as px
from plotly.subplots import make_subplots
import pydicom as pdc
from pandas import DataFrame


from src.pipeline import extract_data, transform_data, predict



app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
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
    multiple=True
    ),
    html.Div(id='output-data-upload')
])

def parse_contents(contents, filename, date):
    """Load the content.

    Function set to load the uploaded content and
    transform the data into a readable format for
    the machine learning model to make predictions.

    Parameters
    ----------
    contents : Unkown
        Contains the uploaded file encoded into its
        64-bit version. This is then decoded and read
        using the pydicom library for data processing.

    filename : string
        The name of the uploaded file. this does
        not impact the data processing aspect or
        the predictions, but it is necessary to
        point out which file the user is looking
        at.

    date : datetime
        Used to let the user know the date in which
        the file was uploaded. Will be used within
        report to download.
    """
    content_type, content = contents.split(',')
    decoded = base64.b64decode(content)
    try:
        ds = pdc.dcmread(io.BytesIO(decoded))
        datapoint = extract_data(ds)
        datapoint = transform_data(datapoint)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return datapoint


@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates): #Need to change this to collect all data.
    """Load the main dashboard."""
    if list_of_contents is not None:
        data = DataFrame([
            parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ])
        data = predict(data, "models/tclass_VGG14")
        first_column = data.pop('Subject ID')
        data.insert(0, 'Suject ID', first_column)
        data['score'] = data['score'].astype(str)
        #Plotting the data.
        fig = make_subplots(
            rows=2, cols=2
        )
        fig1 = px.pie(data, names='pred_class')
        fig2 = px.pie(data, names='sex')
        fig3 = px.pie(data, names='side')
        fig4 = px.histogram(data, x='age')
        fig5 = px.histogram(data, x='pred_class')
        fig6 = px.histogram(data, x='side')
        return html.Div([
            html.H1(children='Main Dashboard', style={'textAlign': 'center'}),
            html.H5('Data Table'),
            html.H6(datetime.datetime.now()),
            html.Div([
                html.Div([
                    dcc.Graph(id='g1', figure=fig1, style={'display': 'inline-block', 'width':'33vw'}),
                    dcc.Graph(id='g2', figure=fig2, style={'display': 'inline-block', 'width':'33vw'}),
                    dcc.Graph(id='g3', figure=fig3, style={'display': 'inline-block', 'width':'33vw'}),
                ]),
                html.Div(children=[
                    dcc.Graph(id='g4', figure=fig4, style={'display': 'inline-block', 'width': '33vw'}),
                    dcc.Graph(id='g5', figure=fig5, style={'display': 'inline-block', 'width': '33vw'}),
                    dcc.Graph(id='g6', figure=fig6, style={'display': 'inline-block', 'width': '33vw'}),
                ])
            ], className="row"),
            #dcc.Graph(figure=fig1, style={'display': 'inline-block'}),
            #dcc.Graph(figure=fig2, style={'display': 'inline-block'}),

            dash_table.DataTable(
                data.to_dict('records'),
                [{'name':i, 'id':i} for i in data.columns],
                export_format="csv"
            ),
            html.Hr(),
        ])

server = app.server

if __name__ == '__main__':
    app.run_server(debug=False, port='8050')
