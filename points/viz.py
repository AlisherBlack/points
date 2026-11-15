from html import escape

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML
from plotly.subplots import make_subplots


def modelnet_figure(cache, split="train", per_class=4, num_points=1024, height=780):
    if per_class < 1:
        raise ValueError("per_class должен быть положительным")

    data = np.load(cache)
    classes = [str(cls) for cls in data["classes"]]
    points, labels = data[f"{split}_points"], data[f"{split}_labels"]
    picked = [np.flatnonzero(labels == c)[:per_class] for c in range(len(classes))]
    samples = [[f"{cls} · {i + 1}" for i in range(len(idx))] for cls, idx in zip(classes, picked)]

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
    for c, names in enumerate(samples):
        for i, name in enumerate(names):
            xyz = points[picked[c][i], :num_points, :3]
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


def critical_points_figure(xyz, critical, title="", height=560):
    xyz = np.asarray(xyz)[:, :3]
    mask = np.zeros(len(xyz), dtype=bool)
    mask[np.asarray(critical)] = True
    fig = go.Figure(
        [
            go.Scatter3d(
                x=xyz[~mask, 0],
                y=xyz[~mask, 1],
                z=xyz[~mask, 2],
                mode="markers",
                marker=dict(size=2, color="#64748b"),
                name=f"остальные: {(~mask).sum()}",
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=xyz[mask, 0],
                y=xyz[mask, 1],
                z=xyz[mask, 2],
                mode="markers",
                marker=dict(size=4, color="#f97316"),
                name=f"критические: {mask.sum()}",
                hoverinfo="skip",
            ),
        ]
    )
    hidden = dict(visible=False)
    fig.update_scenes(
        xaxis=hidden,
        yaxis=hidden,
        zaxis=hidden,
        aspectmode="data",
        bgcolor="#0b1120",
        dragmode="orbit",
        camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.2, y=0.8, z=1.6)),
    )
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.02, y=0.97),
        font=dict(family="Arial, sans-serif", color="#e2e8f0"),
        paper_bgcolor="#0b1120",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(x=0.02, y=0.9),
    )
    return fig


def show_critical_points(xyz, critical, title="", height=560):
    fig = critical_points_figure(xyz, critical, title, height)
    return iframe(
        fig,
        height,
        "PointNet: критические точки",
        "Оранжевые — критические точки: они дают максимум хотя бы в одном канале глобального признака. "
        "Вращение — перетаскивание мышью; масштаб — колесо.",
    )


def show_modelnet(cache, split="train", per_class=4, num_points=1024, height=780):
    fig = modelnet_figure(cache, split, per_class, num_points, height)
    return iframe(
        fig,
        height,
        "ModelNet40: облака точек",
        "Вращение — перетаскивание мышью; масштаб — колесо; "
        "перемещение и сброс камеры — кнопки над графиком. Каждый объект управляется отдельно.",
    )


def iframe(fig, height, title, hint):
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
        f'<p style="margin:0 0 8px;font:13px system-ui;color:#64748b">{escape(hint)}</p>'
        f'<iframe title="{escape(title, quote=True)}" '
        f'style="display:block;width:100%;height:{height}px;border:0" '
        f'srcdoc="{escape(document, quote=True)}"></iframe>'
    )
