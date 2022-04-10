
import pickle
import argparse
import numpy as np 
import pandas as pd
import plotly.graph_objs as go 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dash import Dash, dcc, html, Input, Output, no_update


EMBEDS_FILE = "atari_viz_data.csv"
df = pd.read_csv(EMBEDS_FILE)

pca_cols = [c for c in df.columns if c.startswith('pca')]
tsne_cols = [c for c in df.columns if c.startswith('tsne')]
x_pca = df[pca_cols].to_numpy()
x_tsne = df[tsne_cols].to_numpy()

print("Starting app...")
# App layout 
fig1 = go.Figure(data=[
    go.Scatter(
        x=x_pca[:, 0].reshape(-1), y=x_pca[:, 1].reshape(-1),
        mode='markers',
        marker=dict(
            colorscale='jet',
            color=df['action'],
            size=10,
            sizemode='diameter',
            opacity=0.8
        )
    )
])    
fig1.update_traces(hoverinfo='none', hovertemplate=None)
fig1.update_layout(
    plot_bgcolor='rgba(200,200,200,0.2)',
    title='PCA features',
)

fig2 = go.Figure(data=[
    go.Scatter(
        x=x_tsne[:, 0].reshape(-1), y=x_tsne[:, 1].reshape(-1),
        mode='markers',
        marker=dict(
            colorscale='jet',
            color=df['action'],
            size=10,
            sizemode='diameter',
            opacity=0.8
        )
    )
])    
fig2.update_traces(hoverinfo='none', hovertemplate=None)
fig2.update_layout(
    plot_bgcolor='rgba(200,200,200,0.2)',
    title='TSNE features',
)

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id="graph-basic-1", figure=fig1, clear_on_unhover=True,
              style={'width': '90vh', 'height': '90vh'}),
        dcc.Tooltip(id="graph-tooltip-1"),
    ], style={'width': '50%', 'float': 'left'}),
    html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig2, clear_on_unhover=True,
              style={'width': '90vh', 'height': '90vh'}),
        dcc.Tooltip(id="graph-tooltip-2")
    ], style={'width': '50%', 'float': 'left'})
], style={'width': '100%'})

@app.callback(
    Output("graph-tooltip-1", "show"),
    Output("graph-tooltip-1", "bbox"),
    Output("graph-tooltip-1", "children"),
    Input("graph-basic-1", "hoverData"),
)
def display_hover_fig1(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = df_row['impath']

    children = [
        html.Div([
            html.Img(src=app.get_asset_url(img_src), style={"width": "100%"}),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


@app.callback(
    Output("graph-tooltip-2", "show"),
    Output("graph-tooltip-2", "bbox"),
    Output("graph-tooltip-2", "children"),
    Input("graph-basic-2", "hoverData"),
)
def display_hover_fig2(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    df_row = df.iloc[num]
    img_src = df_row['impath']

    children = [
        html.Div([
            html.Img(src=app.get_asset_url(img_src), style={"width": "100%"}),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True)