from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_shape(root, name, num_points=1024):
    cls = name.rsplit("_", 1)[0]
    return np.loadtxt(Path(root) / cls / f"{name}.txt", delimiter=",", max_rows=num_points, dtype=np.float32)


def show_modelnet(root, split="train", per_class=4, num_points=1024, height=450):
    root = Path(root)
    classes = (root / "modelnet40_shape_names.txt").read_text().split()
    names = (root / f"modelnet40_{split}.txt").read_text().split()

    fig = make_subplots(rows=1, cols=per_class, specs=[[{"type": "scene"}] * per_class], horizontal_spacing=0.01)
    owner, titles = [], []
    for c, cls in enumerate(classes):
        picked = [name for name in names if name.rsplit("_", 1)[0] == cls][:per_class]
        for i, name in enumerate(picked):
            xyz = load_shape(root, name, num_points)[:, :3]
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    marker=dict(size=1.5, color=xyz[:, 1], colorscale="Viridis"),
                    name=name,
                    visible=c == 0,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=i + 1,
            )
            owner.append(c)
        titles.append(f"{cls}: {', '.join(picked)}")

    buttons = [
        dict(label=cls, method="update", args=[{"visible": [o == c for o in owner]}, {"title.text": titles[c]}])
        for c, cls in enumerate(classes)
    ]
    axis = dict(color="white", gridcolor="#444", zerolinecolor="#aaa", showbackground=False, range=[-1, 1])
    fig.update_scenes(
        xaxis=dict(axis, title="x"),
        yaxis=dict(axis, title="y"),
        zaxis=dict(axis, title="z"),
        aspectmode="cube",
        bgcolor="black",
        camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.8, y=1.2, z=2.4)),
    )
    fig.update_layout(
        height=height,
        title=dict(text=titles[0], font=dict(color="white"), x=0.5),
        paper_bgcolor="black",
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0, y=1, xanchor="left", yanchor="top")],
    )
    return fig
