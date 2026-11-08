from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML
from plotly.subplots import make_subplots


def load_shape(root, name, num_points=1024):
    cls = name.rsplit("_", 1)[0]
    return np.loadtxt(Path(root) / cls / f"{name}.txt", delimiter=",", max_rows=num_points, dtype=np.float32)


def modelnet_figure(root, split="train", per_class=4, num_points=1024, height=780):
    if per_class < 1:
        raise ValueError("per_class должен быть положительным")

    root = Path(root)
    classes = (root / "modelnet40_shape_names.txt").read_text().split()
    names = (root / f"modelnet40_{split}.txt").read_text().split()
    samples = [[name for name in names if name.rsplit("_", 1)[0] == cls][:per_class] for cls in classes]

    cols = min(per_class, 2)
    rows = (per_class + cols - 1) // cols
    specs = [
        [{"type": "scene"} if row * cols + col < per_class else None for col in range(cols)]
        for row in range(rows)
    ]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=samples[0],
        horizontal_spacing=0.06 if cols > 1 else 0,
        vertical_spacing=0.08 / (rows - 1) if rows > 1 else 0,
    )

    owner = []
    for c, picked in enumerate(samples):
        for i, name in enumerate(picked):
            xyz = load_shape(root, name, num_points)[:, :3]
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=xyz[:, 1],
                        colorscale="Viridis",
                        cmin=-1,
                        cmax=1,
                        showscale=False,
                    ),
                    name=name,
                    visible=c == 0,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=i // cols + 1,
                col=i % cols + 1,
            )
            owner.append(c)

    buttons = [
        dict(
            label=cls,
            method="update",
            args=[
                {"visible": [o == c for o in owner]},
                {
                    "title.text": f"ModelNet40 · {cls}",
                    **{f"annotations[{i}].text": name for i, name in enumerate(samples[c])},
                },
            ],
        )
        for c, cls in enumerate(classes)
    ]
    axis = dict(
        range=[-1.05, 1.05],
        autorange=False,
        showgrid=True,
        gridcolor="#40516c",
        gridwidth=1,
        showbackground=True,
        backgroundcolor="#121d30",
        zeroline=True,
        zerolinecolor="#71839d",
        showline=True,
        linecolor="#71839d",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        tickfont=dict(size=10),
        showspikes=False,
    )
    fig.update_scenes(
        xaxis=dict(axis, title="x"),
        yaxis=dict(axis, title="y"),
        zaxis=dict(axis, title="z"),
        aspectmode="cube",
        bgcolor="#0b1120",
        dragmode="orbit",
        uirevision="modelnet",
        camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.8, y=1.2, z=2.4)),
    )
    fig.update_annotations(font=dict(size=12, color="#e2e8f0"))
    fig.update_layout(
        height=height,
        title=dict(text=f"ModelNet40 · {classes[0]}", x=0.02, y=0.97),
        font=dict(family="Arial, sans-serif", color="#e2e8f0"),
        paper_bgcolor="#0b1120",
        margin=dict(l=10, r=10, t=85, b=10),
        dragmode="orbit",
        uirevision="modelnet",
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=1,
                y=1.12,
                xanchor="right",
                yanchor="top",
                bgcolor="#1e293b",
                bordercolor="#475569",
                font=dict(color="#e2e8f0"),
            )
        ],
    )
    return fig


def show_modelnet(root, split="train", per_class=4, num_points=1024, height=780):
    fig = modelnet_figure(root, split, per_class, num_points, height)
    plot = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config=dict(
            responsive=True,
            staticPlot=False,
            scrollZoom=True,
            displayModeBar=True,
            displaylogo=False,
            modeBarButtons=[
                ["orbitRotation", "pan3d", "zoom3d"],
                ["resetCameraDefault3d", "toImage"],
            ],
        ),
    )
    document = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<style>html,body{margin:0;padding:0;overflow:hidden;background:#0b1120}</style>"
        f"</head><body>{plot}</body></html>"
    )
    return HTML(
        '<p style="margin:0 0 8px;font:13px system-ui;color:#64748b">'
        "Вращение — перетаскивание мышью; масштаб — колесо; "
        "перемещение и сброс камеры — кнопки над графиком. Каждый объект управляется отдельно."
        "</p>"
        '<iframe title="ModelNet40: облака точек" '
        f'style="display:block;width:100%;height:{height}px;border:0" '
        f'srcdoc="{escape(document, quote=True)}"></iframe>'
    )
