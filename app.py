# ============================================================
# 1. IMPORTACIONES Y CONFIGURACIÓN
# ============================================================

from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output

pd.set_option("display.max_columns", 50)

DATA_URL = (
    "https://raw.githubusercontent.com/owid/co2-data/"
    "master/owid-co2-data.csv"
)

LOCAL_DATA_FILE = Path("owid-co2-data.csv")

START_YEAR = 1990
END_YEAR = 2022
YEARS = END_YEAR - START_YEAR

print("Importaciones completadas.")

# ============================================================
# 2. CARGA Y REVISIÓN INICIAL DE LOS DATOS
# ============================================================

# Si el CSV está cargado en Colab se usa ese archivo.
# En caso contrario, se descarga directamente desde OWID.
data_source = LOCAL_DATA_FILE if LOCAL_DATA_FILE.exists() else DATA_URL

df = pd.read_csv(data_source)

required_columns = [
    "country",
    "iso_code",
    "year",
    "population",
    "gdp",
    "co2",
    "co2_per_capita",
    "co2_per_gdp",
    "primary_energy_consumption",
]

missing_columns = sorted(set(required_columns) - set(df.columns))

if missing_columns:
    raise ValueError(
        "Faltan columnas requeridas: "
        + ", ".join(missing_columns)
    )

print("Dimensiones de la base:", df.shape)
print("Periodo disponible:", df["year"].min(), "-", df["year"].max())
print("Entidades disponibles:", df["country"].nunique())

df[required_columns].head()

# ============================================================
# 3. PREPARACIÓN DE LOS DATOS
# ============================================================

data = df[required_columns].copy()

# Conservar únicamente países con código ISO de tres letras.
country_mask = (
    data["iso_code"]
    .fillna("")
    .astype(str)
    .str.fullmatch(r"[A-Z]{3}")
)

analysis_df = data.loc[country_mask].copy()

# Limitar el análisis al periodo 1990-2022.
analysis_df = analysis_df[
    analysis_df["year"].between(START_YEAR, END_YEAR)
].copy()

# Validar duplicados por país y año.
duplicate_count = analysis_df.duplicated(
    subset=["country", "year"]
).sum()

if duplicate_count > 0:
    raise ValueError(
        f"Se encontraron {duplicate_count} duplicados por país y año."
    )

print("Países incluidos:", analysis_df["country"].nunique())
print("Registros incluidos:", len(analysis_df))
print("Valores faltantes principales (%):")

display(
    analysis_df.isna()
    .mean()
    .mul(100)
    .round(2)
    .sort_values(ascending=False)
    .to_frame("porcentaje_faltante")
    .head(10)
)

# ============================================================
# 4. CÁLCULO DEL CAGR Y CLASIFICACIÓN DE DESACOPLAMIENTO
# ============================================================

def cagr(initial_value, final_value, years):
    """Calcula la tasa de crecimiento anual compuesta en porcentaje."""

    if (
        pd.isna(initial_value)
        or pd.isna(final_value)
        or initial_value <= 0
        or final_value <= 0
        or years <= 0
    ):
        return np.nan

    return (
        (final_value / initial_value) ** (1 / years) - 1
    ) * 100


def classify_decoupling(row):
    """Clasifica el comportamiento conjunto del PIB y el CO₂."""

    gdp_growth = row["gdp_cagr"]
    co2_growth = row["co2_cagr"]

    if pd.isna(gdp_growth) or pd.isna(co2_growth):
        return "Insufficient data"

    if gdp_growth < 0:
        return "Economic contraction"

    if co2_growth < 0:
        return "Absolute decoupling"

    if co2_growth < gdp_growth:
        return "Relative decoupling"

    return "No decoupling"


base = (
    analysis_df[analysis_df["year"] == START_YEAR][
        ["country", "iso_code", "gdp", "co2"]
    ]
    .rename(
        columns={
            "gdp": "gdp_1990",
            "co2": "co2_1990",
        }
    )
)

final = (
    analysis_df[analysis_df["year"] == END_YEAR][
        ["country", "iso_code", "gdp", "co2"]
    ]
    .rename(
        columns={
            "gdp": "gdp_2022",
            "co2": "co2_2022",
        }
    )
)

summary = base.merge(
    final,
    on=["country", "iso_code"],
    how="inner",
)

summary["gdp_cagr"] = summary.apply(
    lambda row: cagr(
        row["gdp_1990"],
        row["gdp_2022"],
        YEARS,
    ),
    axis=1,
)

summary["co2_cagr"] = summary.apply(
    lambda row: cagr(
        row["co2_1990"],
        row["co2_2022"],
        YEARS,
    ),
    axis=1,
)

summary["decoupling_score"] = (
    summary["gdp_cagr"] - summary["co2_cagr"]
)

summary["category"] = summary.apply(
    classify_decoupling,
    axis=1,
)

dashboard_timeseries = analysis_df[
    [
        "country",
        "iso_code",
        "year",
        "population",
        "gdp",
        "co2",
        "co2_per_capita",
        "co2_per_gdp",
        "primary_energy_consumption",
    ]
].copy()

dashboard_summary = summary[
    [
        "country",
        "iso_code",
        "gdp_1990",
        "gdp_2022",
        "co2_1990",
        "co2_2022",
        "gdp_cagr",
        "co2_cagr",
        "decoupling_score",
        "category",
    ]
].copy()

display(dashboard_summary.head())

# ============================================================
# 5. VALIDACIONES Y HALLAZGOS
# ============================================================

assert not dashboard_timeseries.duplicated(
    subset=["country", "year"]
).any()

assert dashboard_summary["country"].is_unique

assert not np.isinf(
    dashboard_summary["gdp_cagr"].dropna()
).any()

assert not np.isinf(
    dashboard_summary["co2_cagr"].dropna()
).any()

valid_categories = {
    "Absolute decoupling",
    "Relative decoupling",
    "No decoupling",
    "Economic contraction",
    "Insufficient data",
}

assert set(
    dashboard_summary["category"].dropna().unique()
).issubset(valid_categories)

# Prueba controlada:
assert np.isclose(cagr(100, 121, 2), 10.0)

# Validación manual para México.
mexico = dashboard_summary[
    dashboard_summary["country"] == "Mexico"
]

if not mexico.empty:
    row = mexico.iloc[0]

    manual_mexico_gdp_cagr = (
        (row["gdp_2022"] / row["gdp_1990"]) ** (1 / YEARS)
        - 1
    ) * 100

    assert np.isclose(
        row["gdp_cagr"],
        manual_mexico_gdp_cagr,
    )

scatter_data = dashboard_summary.dropna(
    subset=[
        "gdp_cagr",
        "co2_cagr",
        "category",
    ]
).copy()

# Participación mundial del Top 10.
world_2022 = df.loc[
    (df["country"] == "World")
    & (df["year"] == END_YEAR),
    "co2",
]

if world_2022.empty:
    raise ValueError(
        "No se encontró el agregado World para 2022."
    )

top_10_emitters = (
    analysis_df[analysis_df["year"] == END_YEAR]
    .dropna(subset=["co2"])
    .nlargest(10, "co2")
)

top_10_co2_share = (
    top_10_emitters["co2"].sum()
    / world_2022.iloc[0]
    * 100
)

absolute_decoupling_count = (
    scatter_data["category"]
    .eq("Absolute decoupling")
    .sum()
)

total_decoupled_count = (
    scatter_data["category"]
    .isin(
        [
            "Absolute decoupling",
            "Relative decoupling",
        ]
    )
    .sum()
)

total_analyzed_countries = scatter_data["country"].nunique()

decoupled_country_share = (
    total_decoupled_count
    / total_analyzed_countries
    * 100
)

valid_ranking = (
    scatter_data[
        scatter_data["gdp_cagr"] > 0
    ]
    .sort_values(
        "decoupling_score",
        ascending=False,
    )
)

best_decoupling_country = valid_ranking.iloc[0]["country"]
best_decoupling_score = valid_ranking.iloc[0]["decoupling_score"]

print("Todas las validaciones fueron superadas.")
print(
    f"Top 10 emisores: {top_10_co2_share:.2f}% "
    "de las emisiones mundiales."
)
print(
    f"Desacoplamiento absoluto: "
    f"{absolute_decoupling_count} países."
)
print(
    f"Países desacoplados: "
    f"{decoupled_country_share:.1f}%."
)
print(
    "Mayor índice de desacoplamiento:",
    best_decoupling_country,
    f"({best_decoupling_score:.2f})",
)

# ============================================================
# 6. DASHBOARD FINAL
# ============================================================

dashboard_summary_app = dashboard_summary.copy()
dashboard_timeseries_app = dashboard_timeseries.copy()

CATEGORY_TRANSLATION = {
    "Absolute decoupling": "Desacoplamiento absoluto",
    "Relative decoupling": "Desacoplamiento relativo",
    "No decoupling": "Sin desacoplamiento",
    "Economic contraction": "Contracción económica",
    "Insufficient data": "Datos insuficientes",
}

CATEGORY_COLORS_ES = {
    "Desacoplamiento absoluto": "#2ca25f",
    "Desacoplamiento relativo": "#f0a202",
    "Sin desacoplamiento": "#d73027",
    "Contracción económica": "#8c8c8c",
    "Datos insuficientes": "#d9d9d9",
}

dashboard_summary_app["categoria_es"] = (
    dashboard_summary_app["category"]
    .map(CATEGORY_TRANSLATION)
    .fillna(dashboard_summary_app["category"])
)

scatter_data = dashboard_summary_app.dropna(
    subset=[
        "gdp_cagr",
        "co2_cagr",
        "category",
    ]
).copy()

scatter_data["categoria_es"] = (
    scatter_data["category"]
    .map(CATEGORY_TRANSLATION)
    .fillna(scatter_data["category"])
)

countries = sorted(
    dashboard_timeseries_app["country"]
    .dropna()
    .unique()
)

default_country = (
    "Mexico"
    if "Mexico" in countries
    else countries[0]
)

GRAPH_CARD_STYLE = {
    "backgroundColor": "white",
    "borderRadius": "12px",
    "boxShadow": "0 1px 5px rgba(0, 0, 0, 0.08)",
    "overflow": "hidden",
}

INSIGHT_CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "20px",
    "borderRadius": "12px",
    "boxShadow": "0 1px 5px rgba(0, 0, 0, 0.08)",
    "minHeight": "155px",
}


def format_percentage(value):
    if pd.isna(value):
        return "N/D"
    return f"{value:,.2f}%"


def create_kpi_card(title, value, subtitle=None):
    card_content = [
        html.P(
            title,
            style={
                "margin": "0",
                "fontSize": "14px",
                "fontWeight": "600",
                "color": "#64748b",
            },
        ),
        html.H2(
            value,
            style={
                "margin": "8px 0 0 0",
                "fontSize": "27px",
                "color": "#0f172a",
            },
        ),
    ]

    if subtitle is not None:
        card_content.append(
            html.P(
                subtitle,
                style={
                    "margin": "5px 0 0 0",
                    "fontSize": "12px",
                    "color": "#94a3b8",
                },
            )
        )

    return html.Div(
        card_content,
        style={
            "backgroundColor": "white",
            "padding": "18px",
            "borderRadius": "12px",
            "boxShadow": "0 1px 5px rgba(0, 0, 0, 0.08)",
            "minHeight": "105px",
        },
    )


def create_decoupling_scatter(selected_country=None):
    fig = px.scatter(
        scatter_data,
        x="gdp_cagr",
        y="co2_cagr",
        color="categoria_es",
        color_discrete_map=CATEGORY_COLORS_ES,
        hover_name="country",
        hover_data={
            "iso_code": True,
            "gdp_cagr": ":.2f",
            "co2_cagr": ":.2f",
            "decoupling_score": ":.2f",
            "categoria_es": False,
        },
        labels={
            "gdp_cagr": "Crecimiento anual del PIB (%)",
            "co2_cagr": "Crecimiento anual del CO₂ (%)",
            "categoria_es": "Clasificación",
            "decoupling_score": "Índice de desacoplamiento",
            "iso_code": "Código ISO",
        },
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#64748b",
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#64748b",
    )

    values = pd.concat(
        [
            scatter_data["gdp_cagr"],
            scatter_data["co2_cagr"],
        ]
    ).dropna()

    fig.add_trace(
        go.Scatter(
            x=[values.min(), values.max()],
            y=[values.min(), values.max()],
            mode="lines",
            line={
                "color": "#94a3b8",
                "width": 1,
                "dash": "dot",
            },
            name="PIB y CO₂ crecen al mismo ritmo",
            hoverinfo="skip",
        )
    )

    if selected_country is not None:
        row = scatter_data[
            scatter_data["country"] == selected_country
        ]

        if not row.empty:
            fig.add_trace(
                go.Scatter(
                    x=row["gdp_cagr"],
                    y=row["co2_cagr"],
                    mode="markers+text",
                    text=row["country"],
                    textposition="top center",
                    marker={
                        "size": 18,
                        "color": "white",
                        "line": {
                            "color": "#0f172a",
                            "width": 3,
                        },
                    },
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title={
            "text": (
                "Crecimiento económico frente al crecimiento "
                "de las emisiones"
            ),
            "x": 0.02,
        },
        template="plotly_white",
        height=570,
        legend_title_text="Clasificación",
        hovermode="closest",
        margin={"l": 60, "r": 25, "t": 80, "b": 60},
    )

    return fig


def create_ranking_chart():
    ranking_data = (
        scatter_data[scatter_data["gdp_cagr"] > 0]
        .sort_values(
            "decoupling_score",
            ascending=False,
        )
        .head(10)
        .sort_values("decoupling_score")
    )

    fig = px.bar(
        ranking_data,
        x="decoupling_score",
        y="country",
        orientation="h",
        color="categoria_es",
        color_discrete_map=CATEGORY_COLORS_ES,
        text="decoupling_score",
        labels={
            "decoupling_score": (
                "PIB CAGR − CO₂ CAGR "
                "(puntos porcentuales)"
            ),
            "country": "",
            "categoria_es": "Clasificación",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    fig.update_layout(
        title={
            "text": "Top 10 en desempeño de desacoplamiento",
            "x": 0.02,
        },
        template="plotly_white",
        height=570,
        showlegend=False,
        margin={"l": 110, "r": 50, "t": 80, "b": 60},
    )

    return fig


def create_top_co2_chart():
    top_co2 = (
        dashboard_summary_app
        .dropna(subset=["co2_2022"])
        .nlargest(10, "co2_2022")
        .sort_values("co2_2022")
    )

    fig = px.bar(
        top_co2,
        x="co2_2022",
        y="country",
        orientation="h",
        text="co2_2022",
        labels={
            "co2_2022": "Emisiones de CO₂ en 2022 (Mt)",
            "country": "",
        },
    )

    fig.update_traces(
        texttemplate="%{text:,.1f}",
        textposition="outside",
    )

    fig.update_layout(
        title={
            "text": "Top 10 países con mayores emisiones de CO₂",
            "x": 0.02,
        },
        template="plotly_white",
        height=510,
        showlegend=False,
        margin={"l": 110, "r": 70, "t": 75, "b": 60},
    )

    return fig


def create_top_gdp_chart():
    top_gdp = (
        dashboard_summary_app
        .dropna(subset=["gdp_2022"])
        .nlargest(10, "gdp_2022")
        .sort_values("gdp_2022")
        .copy()
    )

    top_gdp["gdp_billones"] = (
        top_gdp["gdp_2022"] / 1e12
    )

    fig = px.bar(
        top_gdp,
        x="gdp_billones",
        y="country",
        orientation="h",
        text="gdp_billones",
        labels={
            "gdp_billones": (
                "PIB en 2022 "
                "(billones de dólares internacionales)"
            ),
            "country": "",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    fig.update_layout(
        title={
            "text": "Top 10 países con mayor PIB",
            "x": 0.02,
        },
        template="plotly_white",
        height=510,
        showlegend=False,
        margin={"l": 110, "r": 70, "t": 75, "b": 60},
    )

    return fig


def create_timeseries_chart(selected_country):
    country_data = (
        dashboard_timeseries_app[
            dashboard_timeseries_app["country"]
            == selected_country
        ]
        .sort_values("year")
        .copy()
    )

    fig = go.Figure()

    variables = {
        "PIB": "gdp",
        "Emisiones de CO₂": "co2",
        "CO₂ per cápita": "co2_per_capita",
        "Consumo de energía primaria": (
            "primary_energy_consumption"
        ),
    }

    for label, column in variables.items():
        valid = country_data[
            ["year", column]
        ].dropna().copy()

        if valid.empty:
            continue

        initial_value = valid[column].iloc[0]

        if initial_value == 0:
            continue

        valid["indice_100"] = (
            valid[column] / initial_value
        ) * 100

        fig.add_trace(
            go.Scatter(
                x=valid["year"],
                y=valid["indice_100"],
                mode="lines",
                name=label,
                customdata=valid[[column]],
                hovertemplate=(
                    f"{label}<br>"
                    "Año: %{x}<br>"
                    "Índice: %{y:.1f}<br>"
                    "Valor original: %{customdata[0]:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="#94a3b8",
    )

    fig.update_layout(
        title={
            "text": (
                f"Evolución histórica de {selected_country} "
                "(índice base 100)"
            ),
            "x": 0.02,
        },
        template="plotly_white",
        height=500,
        hovermode="x unified",
        xaxis_title="Año",
        yaxis_title="Índice, primer año disponible = 100",
        margin={"l": 65, "r": 30, "t": 80, "b": 60},
    )

    return fig


app = Dash(__name__)
server = app.server
app.title = "Dashboard de CO₂ y crecimiento económico"

top_10_share_text = f"{top_10_co2_share:.2f}%"
decoupled_share_text = f"{decoupled_country_share:.1f}%"
best_score_text = (
    f"{best_decoupling_score:.2f} puntos porcentuales"
)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Crecimiento económico y emisiones de CO₂",
                    style={
                        "margin": "0",
                        "fontSize": "32px",
                        "color": "#0f172a",
                    },
                ),
                html.P(
                    "Análisis internacional de desacoplamiento, 1990–2022",
                    style={
                        "margin": "7px 0 0 0",
                        "fontSize": "16px",
                        "color": "#64748b",
                    },
                ),
            ],
            style={"marginBottom": "22px"},
        ),

        html.Div(
            [
                html.Label(
                    "Selecciona un país",
                    style={
                        "fontWeight": "600",
                        "color": "#334155",
                    },
                ),
                dcc.Dropdown(
                    id="country-dropdown",
                    options=[
                        {"label": country, "value": country}
                        for country in countries
                    ],
                    value=default_country,
                    clearable=False,
                    searchable=True,
                    style={"marginTop": "7px"},
                ),
            ],
            style={
                "backgroundColor": "white",
                "padding": "16px",
                "borderRadius": "12px",
                "boxShadow": "0 1px 5px rgba(0, 0, 0, 0.08)",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            id="kpi-container",
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(175px, 1fr))"
                ),
                "gap": "14px",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            [
                html.Div(
                    dcc.Graph(id="decoupling-scatter"),
                    style=GRAPH_CARD_STYLE,
                ),
                html.Div(
                    dcc.Graph(id="ranking-chart"),
                    style=GRAPH_CARD_STYLE,
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(480px, 1fr))"
                ),
                "gap": "18px",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            [
                html.Div(
                    dcc.Graph(id="top-co2-chart"),
                    style=GRAPH_CARD_STYLE,
                ),
                html.Div(
                    dcc.Graph(id="top-gdp-chart"),
                    style=GRAPH_CARD_STYLE,
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(480px, 1fr))"
                ),
                "gap": "18px",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            [
                html.H2(
                    "Hallazgos destacados",
                    style={
                        "margin": "0 0 15px 0",
                        "fontSize": "22px",
                        "color": "#0f172a",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    top_10_share_text,
                                    style={
                                        "fontSize": "38px",
                                        "fontWeight": "700",
                                        "color": "#b91c1c",
                                        "marginBottom": "8px",
                                    },
                                ),
                                html.H3(
                                    "Concentración mundial de las emisiones",
                                    style={"margin": "0 0 8px 0"},
                                ),
                                html.P(
                                    (
                                        "Los diez países con mayores "
                                        "emisiones de CO₂ representan el "
                                        f"{top_10_share_text} de las "
                                        "emisiones mundiales en 2022."
                                    ),
                                    style={
                                        "margin": "0",
                                        "lineHeight": "1.6",
                                        "color": "#475569",
                                    },
                                ),
                            ],
                            style={
                                **INSIGHT_CARD_STYLE,
                                "borderTop": "5px solid #b91c1c",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    str(absolute_decoupling_count),
                                    style={
                                        "fontSize": "38px",
                                        "fontWeight": "700",
                                        "color": "#15803d",
                                        "marginBottom": "8px",
                                    },
                                ),
                                html.H3(
                                    "Desacoplamiento absoluto",
                                    style={"margin": "0 0 8px 0"},
                                ),
                                html.P(
                                    (
                                        f"{absolute_decoupling_count} países "
                                        "aumentaron su PIB mientras redujeron "
                                        "sus emisiones de CO₂."
                                    ),
                                    style={
                                        "margin": "0",
                                        "lineHeight": "1.6",
                                        "color": "#475569",
                                    },
                                ),
                            ],
                            style={
                                **INSIGHT_CARD_STYLE,
                                "borderTop": "5px solid #15803d",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    decoupled_share_text,
                                    style={
                                        "fontSize": "38px",
                                        "fontWeight": "700",
                                        "color": "#d97706",
                                        "marginBottom": "8px",
                                    },
                                ),
                                html.H3(
                                    "Países desacoplados",
                                    style={"margin": "0 0 8px 0"},
                                ),
                                html.P(
                                    (
                                        f"El {decoupled_share_text} de los "
                                        "países analizados presenta "
                                        "desacoplamiento absoluto o relativo."
                                    ),
                                    style={
                                        "margin": "0",
                                        "lineHeight": "1.6",
                                        "color": "#475569",
                                    },
                                ),
                            ],
                            style={
                                **INSIGHT_CARD_STYLE,
                                "borderTop": "5px solid #d97706",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    best_decoupling_country,
                                    style={
                                        "fontSize": "28px",
                                        "fontWeight": "700",
                                        "color": "#2563eb",
                                        "marginBottom": "8px",
                                    },
                                ),
                                html.H3(
                                    "Mayor índice de desacoplamiento",
                                    style={"margin": "0 0 8px 0"},
                                ),
                                html.P(
                                    (
                                        f"{best_decoupling_country} presenta "
                                        "la mayor diferencia favorable, con "
                                        f"{best_score_text}."
                                    ),
                                    style={
                                        "margin": "0",
                                        "lineHeight": "1.6",
                                        "color": "#475569",
                                    },
                                ),
                            ],
                            style={
                                **INSIGHT_CARD_STYLE,
                                "borderTop": "5px solid #2563eb",
                            },
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": (
                            "repeat(auto-fit, minmax(230px, 1fr))"
                        ),
                        "gap": "15px",
                    },
                ),
            ],
            style={"marginBottom": "18px"},
        ),

        html.Div(
            dcc.Graph(id="timeseries-chart"),
            style=GRAPH_CARD_STYLE,
        ),

        html.Div(
            [
                html.H3(
                    "Cómo interpretar el tablero",
                    style={"marginTop": "0"},
                ),
                html.P(
                    (
                        "Existe desacoplamiento absoluto cuando el PIB "
                        "crece mientras las emisiones de CO₂ disminuyen. "
                        "El desacoplamiento relativo ocurre cuando ambas "
                        "variables crecen, pero el PIB crece más rápido. "
                        "El índice se calcula como CAGR del PIB menos "
                        "CAGR del CO₂."
                    ),
                    style={
                        "lineHeight": "1.6",
                        "color": "#475569",
                    },
                ),
                html.P(
                    (
                        "Las series históricas utilizan un índice base 100. "
                        "Un valor de 150 indica un aumento de 50 % respecto "
                        "al primer año disponible."
                    ),
                    style={
                        "marginBottom": "0",
                        "lineHeight": "1.6",
                        "color": "#475569",
                    },
                ),
            ],
            style={
                "backgroundColor": "white",
                "padding": "18px",
                "borderRadius": "12px",
                "boxShadow": "0 1px 5px rgba(0, 0, 0, 0.08)",
                "marginTop": "18px",
            },
        ),

        html.Div(
            [
                html.H3(
                    "Fuente de los datos",
                    style={"marginTop": "0"},
                ),
                html.P(
                    [
                        "Datos públicos de ",
                        html.A(
                            "Our World in Data",
                            href=(
                                "https://ourworldindata.org/"
                                "co2-and-greenhouse-gas-emissions"
                            ),
                            target="_blank",
                        ),
                        ". Periodo analizado: 1990–2022.",
                    ],
                    style={
                        "lineHeight": "1.6",
                        "color": "#475569",
                    },
                ),
            ],
            style={
                "backgroundColor": "white",
                "padding": "18px",
                "borderRadius": "12px",
                "boxShadow": "0 1px 5px rgba(0, 0, 0, 0.08)",
                "marginTop": "18px",
            },
        ),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f1f5f9",
        "minHeight": "100vh",
        "padding": "28px",
    },
)


@app.callback(
    Output("kpi-container", "children"),
    Output("decoupling-scatter", "figure"),
    Output("ranking-chart", "figure"),
    Output("top-co2-chart", "figure"),
    Output("top-gdp-chart", "figure"),
    Output("timeseries-chart", "figure"),
    Input("country-dropdown", "value"),
)
def update_dashboard(selected_country):

    selected_row = dashboard_summary_app[
        dashboard_summary_app["country"]
        == selected_country
    ]

    total_countries = scatter_data["country"].nunique()

    absolute_count = (
        scatter_data["category"]
        .eq("Absolute decoupling")
        .sum()
    )

    relative_count = (
        scatter_data["category"]
        .eq("Relative decoupling")
        .sum()
    )

    decoupled_percentage = (
        (absolute_count + relative_count)
        / total_countries
        * 100
    )

    if selected_row.empty:
        country_gdp = np.nan
        country_co2 = np.nan
        country_score = np.nan
        country_category = "Sin datos"

    else:
        selected = selected_row.iloc[0]
        country_gdp = selected["gdp_cagr"]
        country_co2 = selected["co2_cagr"]
        country_score = selected["decoupling_score"]

        country_category = CATEGORY_TRANSLATION.get(
            selected["category"],
            selected["category"],
        )

    kpis = [
        create_kpi_card(
            "Países analizados",
            f"{total_countries:,}",
        ),
        create_kpi_card(
            "Desacoplamiento absoluto",
            f"{absolute_count:,}",
            "El PIB creció y el CO₂ disminuyó",
        ),
        create_kpi_card(
            "Países desacoplados",
            f"{decoupled_percentage:.1f}%",
            "Casos absolutos y relativos",
        ),
        create_kpi_card(
            f"{selected_country}: PIB",
            format_percentage(country_gdp),
            "Tasa de crecimiento anual compuesta",
        ),
        create_kpi_card(
            f"{selected_country}: CO₂",
            format_percentage(country_co2),
            "Tasa de crecimiento anual compuesta",
        ),
        create_kpi_card(
            "Índice de desacoplamiento",
            format_percentage(country_score),
            country_category,
        ),
    ]

    return (
        kpis,
        create_decoupling_scatter(selected_country),
        create_ranking_chart(),
        create_top_co2_chart(),
        create_top_gdp_chart(),
        create_timeseries_chart(selected_country),
    )

print("Dashboard final construido correctamente.")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )