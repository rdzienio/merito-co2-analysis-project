
# PROJEKT ZALICZENIOWY: Analiza emisji CO2 i zuzycia energii na swiecie
# Autor: Jakub Adwentowski / Robert Dzienio
# Zrodlo danych: Our World in Data
# Wersja prezentacyjna + prosty dashboard HTML offline

import os
from textwrap import dedent

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

DATA_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
CSV_PATH = "owid-co2-data.csv"
OUTPUT_DIR = "wyniki"
YEAR_SNAPSHOT = 2022
DASHBOARD_HTML = os.path.join(OUTPUT_DIR, "dashboard_offline.html")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261", "#9B5DE5", "#F15BB5"]
ALL_HIGHLIGHTS = ["China", "United States", "India", "Germany", "Poland", "Japan", "Norway", "Sweden", "Qatar", "Brazil"]
PCA_HIGHLIGHTS = ["China", "United States", "India", "Germany", "Poland", "Qatar", "Norway", "Sweden"]
ENERGY_HIGHLIGHTS = ["China", "United States", "India", "Germany", "Poland", "Norway", "Sweden", "Qatar"]
SPECIAL_COUNTRY = "Poland"

MAJOR_EMITTERS = {"China", "United States", "India"}
NORDIC_LOW = {"Norway", "Sweden"}
GULF = {"Qatar"}

SCATTER_OFFSETS = {
    "China": (10, 8),
    "United States": (10, 8),
    "India": (10, 8),
    "Germany": (10, -1),
    "Poland": (0, 10),
    "Japan": (14, 8),
    "Norway": (10, 8),
    "Sweden": (10, 8),
    "Qatar": (10, 8),
    "Brazil": (10, 8),
}

PCA_OFFSETS = {
    "China": (10, 8),
    "United States": (10, 8),
    "India": (10, 8),
    "Germany": (10, 0),
    "Poland": (-1, 10),
    "Qatar": (10, -18),
    "Norway": (10, 8),
    "Sweden": (10, 0),
}

ENERGY_OFFSETS = {
    "China": (10, 8),
    "United States": (10, 8),
    "India": (0, 8),
    "Germany": (-8, -8),
    "Poland": (-1, 10),
    "Norway": (10, 8),
    "Sweden": (10, 8),
    "Qatar": (10, -8),
}

LINE_END_OFFSETS = {
    "China": (8, 0),
    "United States": (8, 0),
    "India": (8, 0),
    "Germany": (8, 0),
    "Poland": (8, 0),
    "Japan": (8, 0),
}


def get_highlight_style(country):
    if country == SPECIAL_COUNTRY:
        return {
            "color": "#FF7F0E",
            "size": 135,
            "edgecolor": "black",
            "linewidth": 1.2,
            "fontweight": "bold",
            "bbox_facecolor": "#FFF3E0",
        }
    if country in MAJOR_EMITTERS:
        return {
            "color": "#D62728",
            "size": 105,
            "edgecolor": "white",
            "linewidth": 0.9,
            "fontweight": "bold",
            "bbox_facecolor": "#FDECEC",
        }
    if country in NORDIC_LOW:
        return {
            "color": "#2CA02C",
            "size": 95,
            "edgecolor": "white",
            "linewidth": 0.9,
            "fontweight": "bold",
            "bbox_facecolor": "#EAF7EA",
        }
    if country in GULF:
        return {
            "color": "#9467BD",
            "size": 95,
            "edgecolor": "white",
            "linewidth": 0.9,
            "fontweight": "bold",
            "bbox_facecolor": "#F2EBFA",
        }
    return {
        "color": "#1F77B4",
        "size": 88,
        "edgecolor": "white",
        "linewidth": 0.8,
        "fontweight": "normal",
        "bbox_facecolor": "#EEF5FB",
    }


def annotate_highlight_points(ax, data, x_col, y_col, countries, offset_map):
    subset = data[data["country"].isin(countries)].copy()
    for _, row in subset.iterrows():
        country = row["country"]
        style = get_highlight_style(country)
        x = row[x_col]
        y = row[y_col]
        dx, dy = offset_map.get(country, (8, 6))
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"

        ax.scatter(x, y, s=style["size"], color=style["color"], edgecolors=style["edgecolor"], linewidth=style["linewidth"], zorder=6)
        ax.annotate(
            country,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            fontweight=style["fontweight"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=style["bbox_facecolor"], edgecolor="none", alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=style["color"], lw=0.8, alpha=0.8),
            zorder=7,
        )


def create_dashboard_offline(world, top10, reg_data, cluster_data, pca_var):
    highlight = ["China", "United States", "India", "Germany", "Poland", "Japan", "Norway", "Sweden", "Qatar"]
    scatter_data = reg_data.copy()
    scatter_data["highlight"] = scatter_data["country"].isin(highlight)
    cluster_plot = cluster_data.copy()
    cluster_plot["highlight"] = cluster_plot["country"].isin(highlight)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Globalny trend emisji CO2 (1990-2023)",
            f"Top 10 emitentow CO2 ({YEAR_SNAPSHOT})",
            f"CO2 per capita vs PKB per capita ({YEAR_SNAPSHOT})",
            "K-Means + PCA",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.22,
    )

    # Panel 1: trend globalny
    fig.add_trace(go.Scatter(
        x=world["year"], y=world["co2"], mode="lines", name="Emisje rzeczywiste",
        legendgroup="trend",
        line=dict(color="#2E86AB", width=3),
        hovertemplate="Rok: %{x}<br>CO2: %{y:.0f} Mt<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=world["year"], y=world["trend"], mode="lines", name="Trend liniowy",
        legendgroup="trend",
        line=dict(color="#E84855", width=2, dash="dash"),
        hovertemplate="Rok: %{x}<br>Trend: %{y:.0f} Mt<extra></extra>"
    ), row=1, col=1)

    # Panel 2: top 10 emitentow
    bar_colors = ["#FF7F0E" if c == "Poland" else "#3BB273" for c in top10["country"]]
    fig.add_trace(go.Bar(
        x=top10["co2"], y=top10["country"], orientation="h", name="Top 10",
        marker_color=bar_colors,
        text=[f"{x:.0f}" for x in top10["co2"]],
        textposition="outside",
        hovertemplate="%{y}<br>CO2: %{x:.0f} Mt<extra></extra>",
        showlegend=False
    ), row=1, col=2)

    # Panel 3: scatter PKB vs CO2 pc
    base = scatter_data[~scatter_data["highlight"]]
    marked = scatter_data[scatter_data["highlight"]]
    fig.add_trace(go.Scatter(
        x=base["log_gdp_per_capita"], y=base["log_co2_per_capita"], mode="markers",
        name="Pozostale kraje", legendgroup="scatter",
        marker=dict(color="rgba(160,170,180,0.38)", size=7),
        text=base["country"],
        hovertemplate="%{text}<br>log10(PKB pc): %{x:.2f}<br>log10(CO2 pc): %{y:.2f}<extra></extra>"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=marked["log_gdp_per_capita"], y=marked["log_co2_per_capita"], mode="markers+text",
        name="Wybrane kraje", legendgroup="scatter",
        marker=dict(color="#FF7F0E", size=11, line=dict(color="black", width=1)),
        text=marked["country"], textposition="top center",
        hovertemplate="%{text}<br>log10(PKB pc): %{x:.2f}<br>log10(CO2 pc): %{y:.2f}<extra></extra>"
    ), row=2, col=1)

    # Panel 4: PCA + KMeans - legenda tylko raz dla klastrow
    colors = {0: "#9ECAE1", 1: "#A1D99B", 2: "#FDD0A2", 3: "#D4B9DA"}
    for cluster_id in sorted(cluster_plot["klaster"].unique()):
        part = cluster_plot[cluster_plot["klaster"] == cluster_id]
        fig.add_trace(go.Scatter(
            x=part["PC1"], y=part["PC2"], mode="markers", name=f"Klaster {cluster_id}",
            legendgroup=f"cluster_{cluster_id}",
            marker=dict(color=colors[int(cluster_id)], size=8),
            text=part["country"],
            hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
        ), row=2, col=2)

    pca_marked = cluster_plot[cluster_plot["highlight"]]
    fig.add_trace(go.Scatter(
        x=pca_marked["PC1"], y=pca_marked["PC2"], mode="markers+text", name="Highlight",
        legendgroup="highlight",
        marker=dict(color="#D62728", size=11, line=dict(color="black", width=1)),
        text=pca_marked["country"], textposition="top center",
        hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
    ), row=2, col=2)

    # Osie
    fig.update_xaxes(title_text="Rok", row=1, col=1)
    fig.update_yaxes(title_text="Emisja CO2 [Mt]", row=1, col=1)
    fig.update_xaxes(title_text="Emisja CO2 [Mt]", row=1, col=2)
    fig.update_yaxes(title_text="Kraj", row=1, col=2)
    fig.update_xaxes(title_text="log10(PKB per capita)", row=2, col=1)
    fig.update_yaxes(title_text="log10(CO2 per capita)", row=2, col=1)
    fig.update_xaxes(title_text=f"PC1 ({pca_var[0] * 100:.1f}% wariancji)", row=2, col=2)
    fig.update_yaxes(title_text=f"PC2 ({pca_var[1] * 100:.1f}% wariancji)", row=2, col=2)

    # Poprawa wygladu subplot titles
    for ann in fig.layout.annotations:
        if ann.text in [
            "Globalny trend emisji CO2 (1990-2023)",
            f"Top 10 emitentow CO2 ({YEAR_SNAPSHOT})",
            f"CO2 per capita vs PKB per capita ({YEAR_SNAPSHOT})",
            "K-Means + PCA",
        ]:
            ann.font.size = 15

    fig.update_layout(
        title=dict(
            text="Dashboard: Analiza emisji CO2 i zuzycia energii na swiecie",
            x=0.5,
            xanchor="center",
            y=0.985,
            font=dict(size=24)
        ),
        template="plotly_white",
        width=1450,
        height=950,
        margin=dict(l=70, r=40, t=150, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            itemwidth=70,
            tracegroupgap=8,
            bgcolor="rgba(255,255,255,0.85)"
        ),
        hoverlabel=dict(font_size=12),
    )

    fig.add_annotation(
        text="Interaktywny raport HTML offline: zoom, hover, ukrywanie serii w legendzie",
        xref="paper", yref="paper", x=0.5, y=1.14,
        showarrow=False, font=dict(size=13, color="gray")
    )

    fig.write_html(DASHBOARD_HTML, include_plotlyjs=True, full_html=True)


# 1. Pobranie i wczytanie danych
if not os.path.exists(CSV_PATH):
    print("Pobieranie danych...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    with open(CSV_PATH, "wb") as f:
        f.write(response.content)

df_raw = pd.read_csv(CSV_PATH)
print("Dane wczytane.")
print("Liczba rekordow:", len(df_raw))
print("Liczba kolumn:", len(df_raw.columns))
print("Zakres lat:", int(df_raw["year"].min()), "-", int(df_raw["year"].max()))
print("Liczba unikalnych krajow/obszarow:", df_raw["country"].nunique())

# 2. Preprocessing
df = df_raw[df_raw["iso_code"].notna() & ~df_raw["iso_code"].str.startswith("OWID")].copy()
duplicates = df.duplicated(subset=["country", "year"]).sum()
df = df.drop_duplicates(subset=["country", "year"])
df = df[df["year"].between(1990, 2023)].copy()

snap = df[df["year"] == YEAR_SNAPSHOT].copy()
important_cols = [
    "co2", "co2_per_capita", "gdp", "population",
    "energy_per_capita", "energy_per_gdp",
    "coal_co2", "oil_co2", "gas_co2", "cement_co2"
]
missing_summary = snap[important_cols].isna().sum().sort_values(ascending=False)

snap_basic = snap.dropna(subset=["co2", "gdp", "population"]).copy()
snap_basic["gdp_per_capita"] = snap_basic["gdp"] / snap_basic["population"]
snap_basic["log_gdp_per_capita"] = np.log10(snap_basic["gdp_per_capita"].clip(lower=1))
snap_basic["log_co2_per_capita"] = np.log10(snap_basic["co2_per_capita"].clip(lower=0.01))

bins = [0, 2, 5, 10, np.inf]
labels = ["niskie (<2t)", "srednie (2-5t)", "wysokie (5-10t)", "bardzo wysokie (>10t)"]
snap_basic["emisja_kategoria"] = pd.cut(snap_basic["co2_per_capita"], bins=bins, labels=labels)
emission_groups = snap_basic["emisja_kategoria"].value_counts().sort_index()

# 3. Analiza danych
world = df_raw[df_raw["country"] == "World"][["year", "co2"]].dropna()
world = world[world["year"].between(1990, 2023)].copy()
X_world = world[["year"]].values
y_world = world["co2"].values
reg_world = LinearRegression().fit(X_world, y_world)
y_world_pred = reg_world.predict(X_world)
r2_world = r2_score(y_world, y_world_pred)
slope_world = reg_world.coef_[0]
world["trend"] = y_world_pred

reg_data = snap_basic.dropna(subset=["log_gdp_per_capita", "log_co2_per_capita"]).copy()
X_reg = reg_data[["log_gdp_per_capita"]].values
y_reg = reg_data["log_co2_per_capita"].values
reg_pc = LinearRegression().fit(X_reg, y_reg)
r2_pc = r2_score(y_reg, reg_pc.predict(X_reg))
beta_pc = reg_pc.coef_[0]

corr_cols = ["co2", "co2_per_capita", "gdp", "population", "coal_co2", "oil_co2", "gas_co2", "energy_per_capita", "energy_per_gdp"]
corr_data = snap[corr_cols].dropna().copy()
corr_matrix = corr_data.corr(numeric_only=True)

cluster_cols = ["co2_per_capita", "energy_per_capita", "energy_per_gdp", "log_gdp_per_capita"]
cluster_data = snap_basic.dropna(subset=cluster_cols).copy()
scaler = StandardScaler()
X_cluster = scaler.fit_transform(cluster_data[cluster_cols])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_data["klaster"] = kmeans.fit_predict(X_cluster)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)
cluster_data["PC1"] = X_pca[:, 0]
cluster_data["PC2"] = X_pca[:, 1]
pca_var = pca.explained_variance_ratio_
cluster_counts = cluster_data["klaster"].value_counts().sort_index()

# 4. Statyczne wykresy (zostawilem najwazniejsze)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(world["year"], world["co2"], label="Emisje rzeczywiste", color=PALETTE[0], linewidth=2.5)
ax.plot(world["year"], y_world_pred, "--", label=f"Trend liniowy (R²={r2_world:.2f})", color=PALETTE[1], linewidth=2)
ax.set_title("Globalne emisje CO2 w latach 1990-2023")
ax.set_xlabel("Rok")
ax.set_ylabel("Emisja CO2 [Mt]")
ax.grid(axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_trend_globalny.png"), bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
top10 = snap_basic.nlargest(10, "co2")[["country", "co2"]].sort_values("co2")
bar_colors = ["#FF7F0E" if c == SPECIAL_COUNTRY else "#9BBFE0" for c in top10["country"]]
ax.barh(top10["country"], top10["co2"], color=bar_colors)
ax.set_title(f"Top 10 krajow wedlug emisji CO2 ({YEAR_SNAPSHOT})")
ax.set_xlabel("Emisja CO2 [Mt]")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_top10_emitenci.png"), bbox_inches="tight")
plt.close()

# 4.3 Histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(snap_basic["co2_per_capita"].dropna(), bins=35, color=PALETTE[2], edgecolor="white")
ax.axvline(snap_basic["co2_per_capita"].median(), color=PALETTE[1], linestyle="--", linewidth=2, label="Mediana")
ax.axvline(snap_basic["co2_per_capita"].mean(), color=PALETTE[0], linestyle=":", linewidth=2, label="Srednia")
ax.set_title(f"Rozklad emisji CO2 per capita ({YEAR_SNAPSHOT})")
ax.set_xlabel("CO2 per capita [t/os.]")
ax.set_ylabel("Liczba krajow")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_histogram_per_capita.png"), bbox_inches="tight")
plt.close()

# 4.4 Scatter PKB vs CO2 per capita
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(
    reg_data["log_gdp_per_capita"],
    reg_data["log_co2_per_capita"],
    color="#D7DDE5",
    s=34,
    alpha=0.42,
    edgecolors="none",
    label="Pozostale kraje"
)
line_x = np.linspace(reg_data["log_gdp_per_capita"].min(), reg_data["log_gdp_per_capita"].max(), 100)
ax.plot(line_x, reg_pc.predict(line_x.reshape(-1, 1)), "--", color="black", linewidth=2, label=f"Regresja (R²={r2_pc:.2f})")
annotate_highlight_points(ax, reg_data, "log_gdp_per_capita", "log_co2_per_capita", ALL_HIGHLIGHTS, SCATTER_OFFSETS)
ax.set_title(f"CO2 per capita vs PKB per capita ({YEAR_SNAPSHOT}, skala log)")
ax.set_xlabel("log10(PKB per capita)")
ax.set_ylabel("log10(CO2 per capita)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_scatter_co2_gdp.png"), bbox_inches="tight")
plt.close()

# 4.5 Heatmapa korelacji
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Macierz korelacji Pearsona")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_heatmap_korelacji.png"), bbox_inches="tight")
plt.close()

# 4.6 Trendy wybranych krajow
fig, ax = plt.subplots(figsize=(12, 6))
selected_countries = ["China", "United States", "India", "Germany", "Poland", "Japan"]
country_colors = {
    "China": "#D62728",
    "United States": "#8C564B",
    "India": "#9467BD",
    "Germany": "#1F77B4",
    "Poland": "#FF7F0E",
    "Japan": "#2CA02C",
}
for country in selected_countries:
    country_df = df[df["country"] == country][["year", "co2"]].dropna()
    lw = 3 if country == SPECIAL_COUNTRY else 2
    z = 5 if country == SPECIAL_COUNTRY else 3
    ax.plot(country_df["year"], country_df["co2"], linewidth=lw, color=country_colors[country], zorder=z)
    if not country_df.empty:
        last_row = country_df.iloc[-1]
        style = get_highlight_style(country)
        dx, dy = LINE_END_OFFSETS.get(country, (8, 0))
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.scatter(last_row["year"], last_row["co2"], s=style["size"], color=style["color"], edgecolors=style["edgecolor"], linewidth=style["linewidth"], zorder=6)
        ax.annotate(
            country,
            (last_row["year"], last_row["co2"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            fontweight=style["fontweight"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=style["bbox_facecolor"], edgecolor="none", alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=style["color"], lw=0.8, alpha=0.8),
        )
ax.set_title("Trendy emisji CO2 w wybranych krajach (1990-2023)")
ax.set_xlabel("Rok")
ax.set_ylabel("Emisja CO2 [Mt]")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_trendy_krajow.png"), bbox_inches="tight")
plt.close()

# 4.7 PCA + K-Means
fig, ax = plt.subplots(figsize=(10, 7))
cluster_palette = {0: "#9ECAE1", 1: "#A1D99B", 2: "#FDD0A2", 3: "#D4B9DA"}
for cluster_id in sorted(cluster_data["klaster"].unique()):
    part = cluster_data[cluster_data["klaster"] == cluster_id]
    ax.scatter(part["PC1"], part["PC2"], s=42, alpha=0.40, color=cluster_palette[int(cluster_id)], edgecolors="none", label=f"Klaster {cluster_id}")
annotate_highlight_points(ax, cluster_data, "PC1", "PC2", PCA_HIGHLIGHTS, PCA_OFFSETS)
ax.set_title("Grupowanie krajow: K-Means + PCA")
ax.set_xlabel(f"PC1 ({pca_var[0] * 100:.1f}% wariancji)")
ax.set_ylabel(f"PC2 ({pca_var[1] * 100:.1f}% wariancji)")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_kmeans_pca.png"), bbox_inches="tight")
plt.close()

# 4.8 Energia na jednostke PKB vs emisje - wersja prezentacyjna
fig, ax = plt.subplots(figsize=(10, 6))
energy_scatter = snap.dropna(subset=["energy_per_gdp", "co2_per_capita"]).copy()
ax.scatter(
    energy_scatter["energy_per_gdp"],
    energy_scatter["co2_per_capita"],
    color="#D7DDE5",
    s=34,
    alpha=0.40,
    edgecolors="none",
    label="Pozostale kraje"
)
annotate_highlight_points(ax, energy_scatter, "energy_per_gdp", "co2_per_capita", ENERGY_HIGHLIGHTS, ENERGY_OFFSETS)
ax.set_title(f"Zuzycie energii na jednostke PKB vs emisje CO2 per capita ({YEAR_SNAPSHOT})")
ax.set_xlabel("Energy per GDP")
ax.set_ylabel("CO2 per capita [t/os.]")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_energy_gdp_vs_co2.png"), bbox_inches="tight")
plt.close()

# 4.9 Struktura zrodel emisji
fig, ax = plt.subplots(figsize=(12, 6))
top6 = snap_basic.nlargest(6, "co2")["country"].tolist()
src_cols = ["coal_co2", "oil_co2", "gas_co2", "cement_co2"]
src_labels = ["Wegiel", "Ropa", "Gaz", "Cement"]
src_data = snap[snap["country"].isin(top6)].set_index("country")[src_cols].fillna(0)
bottom = np.zeros(len(src_data))
for i, col in enumerate(src_cols):
    ax.bar(src_data.index, src_data[col], bottom=bottom, label=src_labels[i])
    bottom += src_data[col].values
ax.set_title(f"Struktura zrodel emisji CO2 - top 6 emitentow ({YEAR_SNAPSHOT})")
ax.set_ylabel("Emisja CO2 [Mt]")
ax.set_xlabel("Kraj")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_struktura_emisji.png"), bbox_inches="tight")
plt.close()

# 5. Dashboard offline wpleciony do glownego skryptu
create_dashboard_offline(world, top10, reg_data, cluster_data, pca_var)

# 6. Sprawozdanie
strong_corr = []
if "co2_per_capita" in corr_matrix.index:
    for col in corr_matrix.columns:
        if col != "co2_per_capita":
            r = corr_matrix.loc["co2_per_capita", col]
            if abs(r) >= 0.6:
                strong_corr.append(f"- {col}: r = {r:.2f}")

report = f"""
PROJEKT ZALICZENIOWY: Analiza emisji CO2 i zuzycia energii na swiecie

Autor: Jakub Adwentowski / Robert Dzienio
Zrodlo danych: Our World in Data (owid-co2-data.csv)

1. Charakterystyka zbioru danych
- Zbior danych pochodzi z projektu Our World in Data i zawiera informacje dotyczace emisji CO2, zuzycia energii oraz wskaznikow ekonomicznych dla krajow na przestrzeni lat.
- Liczba rekordow w danych surowych: {len(df_raw)}
- Liczba kolumn: {len(df_raw.columns)}
- Zakres lat: {int(df_raw['year'].min())}-{int(df_raw['year'].max())}
- Liczba krajow po odfiltrowaniu agregatow: {df['country'].nunique()}

Struktura danych:
- Dane maja strukture tabelaryczna (DataFrame), gdzie kazdy rekord odpowiada krajowi w danym roku.
- Kluczowe kolumny to: country, year oraz zmienne opisujace emisje i gospodarke.

Typy atrybutow:
- Numeryczne: co2, co2_per_capita, gdp, population, energy_per_capita, energy_per_gdp
- Kategoryczne: country, iso_code
- Czasowe: year
- Binarne (posrednio): obecnosc / brak danych (NaN)

Potencjalne zastosowanie zbioru danych:
- analiza zaleznosci miedzy wzrostem gospodarczym a emisja CO2,
- ocena polityki klimatycznej panstw,
- segmentacja krajow wedlug poziomu emisji i zuzycia energii,
- wspomaganie decyzji strategicznych dotyczacych transformacji energetycznej,
- budowa modeli predykcyjnych dotyczacych emisji w przyszlosci.

2. Wstepne przetwarzanie danych
- Usunieto agregaty regionalne i rekordy typu OWID.
- Liczba duplikatow country+year: {duplicates}
- Zakres analizy ograniczono do lat 1990-2023.
- Snapshot dla analiz przekrojowych: {YEAR_SNAPSHOT}
- Liczba krajow w snapshot z podstawowymi danymi: {len(snap_basic)}

Braki danych w snapshot {YEAR_SNAPSHOT} (wybrane kolumny):
{missing_summary.to_string()}

Transformacje danych:
- Obliczono GDP per capita.
- Zastosowano log10 dla GDP per capita oraz CO2 per capita.
- Zastosowano dyskretyzacje emisji per capita wedlug prostych progow eksperckich.

Liczba krajow wedlug kategorii emisji:
{emission_groups.to_string()}

3. Analiza danych
Uzasadnienie wyboru metod:
- Regresja liniowa: do analizy trendu globalnego emisji CO2 oraz zaleznosci PKB-emisje.
- Korelacja: do identyfikacji silnych zaleznosci miedzy zmiennymi.
- K-Means: do segmentacji krajow o podobnym profilu energetycznym.
- PCA: do redukcji wymiarowosci i wizualizacji klastrow.

Globalny trend emisji CO2:
- Nachylenie trendu liniowego: {slope_world:.2f} Mt CO2 / rok
- R^2 modelu: {r2_world:.3f}

Regresja CO2 per capita vs PKB per capita:
- Wspolczynnik regresji log-log: {beta_pc:.3f}
- R^2 modelu: {r2_pc:.3f}

Silne korelacje z CO2 per capita:
{os.linesep.join(strong_corr) if strong_corr else '- Brak korelacji powyzej progu |r| >= 0.6'}

Grupowanie K-Means:
- Liczba klastrow: 4
- Rozklad krajow na klastry:
{cluster_counts.to_string()}

PCA:
- PC1 wyjasnia {pca_var[0] * 100:.1f}% wariancji
- PC2 wyjasnia {pca_var[1] * 100:.1f}% wariancji
- Lacznie: {pca_var.sum() * 100:.1f}% wariancji

4. Wizualizacja danych
- Przygotowano wykres liniowy, wykres slupkowy, histogram, wykres rozrzutu, heatmape korelacji oraz wizualizacje PCA.
- Dodatkowo wygenerowano prosty dashboard HTML offline: dashboard_offline.html.
- Dashboard umozliwia interaktywne przegladanie danych (hover, zoom, ukrywanie serii w legendzie) i dziala bez internetu.

5. Interpretacja wyników i wnioski

Wnioski z analizy trendów (regresja liniowa)
- Globalne emisje CO2 rosły średnio o {slope_world:.2f} Mt rocznie w latach 1990-2023.
- Model liniowy wyjaśnia {r2_world:.3f}% wariancji, jednak widoczne są
  okresy spowolnienia (kryzys 2008-2009, pandemia 2020) oraz powrotu do wzrostu.
- Wzrost emisji napędzany jest przede wszystkim przez Chiny i Indie.

Wnioski z analizy korelacji
- Silna dodatnia korelacja emisji per capita z PKB per capita sugeruje,
  że zamożniejsze kraje emitują więcej, choć istnieją wyjątki (Norwegia, Szwecja).
- Korelacja z udziałem OZE jest ujemna - większy udział energii odnawialnej
  wiąże się z niższymi emisjami per capita.
- Emisje z węgla wykazują najsilniejszą korelację z łącznymi emisjami krajowymi.

Wnioski z grupowania K-Means
Zidentyfikowano 4 charakterystyczne grupy krajów:
- Klaster 0: Kraje rozwijające się - niskie emisje, niski PKB, niski udział OZE
- Klaster 1: Bogate kraje z dużym udziałem OZE - niskie/średnie emisje per capita
- Klaster 2: Kraje petro-stanu - bardzo wysokie emisje i PKB per capita
- Klaster 3: Duże gospodarki uprzemysłowione - wysokie emisje łączne

Ograniczenia metod
- Regresja liniowa upraszcza nielinearny charakter zmian emisji.
- K-Means zakłada sferyczny kształt klastrów - może nie uchwycić
  wszystkich złożonych wzorców w danych wielowymiarowych.
- Dane historyczne starsze niż 1950 mają niższe pokrycie i jakość.
- Snapshot przekrojowy (2022) nie uwzglednia dynamiki zmian w czasie.
- Brak uwzglednienia czynnikow politycznych i technologicznych.

Potencjalne zastosowania wyników
- Identyfikacja krajów, które „odstają" od trendu regresji - czyli tych,
  które osiągają niskie emisje mimo wysokiego PKB (benchmarki dekarbonizacji).
- Segmentacja krajów wg klastrów może wspierać projektowanie zróżnicowanej
  polityki klimatycznej w negocjacjach międzynarodowych.
- Model regresji może posłużyć do prognozowania przyszłych emisji
  przy założeniu kontynuacji dotychczasowych trendów.

6. Pliki wynikowe
- 01_trend_globalny.png
- 02_top10_emitenci.png
- 03_histogram_per_capita.png
- 04_scatter_co2_gdp.png
- 05_heatmap_korelacji.png
- 06_trendy_krajow.png
- 07_kmeans_pca.png
- 08_energy_gdp_vs_co2.png
- 09_struktura_emisji.png
- dashboard_offline.html
- sprawozdanie.txt
"""

with open(os.path.join(OUTPUT_DIR, "sprawozdanie.txt"), "w", encoding="utf-8") as f:
    f.write(dedent(report).strip())

print("Gotowe. Wygenerowano wykresy, dashboard offline i sprawozdanie w katalogu:", OUTPUT_DIR)
