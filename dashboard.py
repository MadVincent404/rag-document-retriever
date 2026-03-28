import os
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from main import FEEDBACK_CSV_PATH

st.set_page_config(
    page_title="Dashboard Feedbacks",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────

@st.cache_data(ttl=30)
def load_feedback_data() -> pd.DataFrame:
    if not os.path.exists(FEEDBACK_CSV_PATH):
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "category",
                "question",
                "answer",
                "thumbs",
                "precision",
                "source_path",
            ]
        )

    df = pd.read_csv(FEEDBACK_CSV_PATH, encoding="utf-8")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["is_up"] = df["thumbs"] == "up"
    return df


# ─────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────

st.title("Dashboard Feedbacks")
st.caption(
    f"Source : {FEEDBACK_CSV_PATH} — actualisation toutes les 30 secondes"
)
st.divider()

df = load_feedback_data()

if df.empty:
    st.info(
        "Aucun feedback enregistré. Lance app.py, pose des questions "
        "et soumets des appréciations."
    )
    st.stop()

# ─────────────────────────────────────────
# FILTRES SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.header("Filtres")

    categories = ["Toutes"] + sorted(
        df["category"].dropna().unique().tolist()
    )
    cat_filter = st.selectbox("Catégorie", categories)

    thumb_options = {
        "Tous": None,
        "Positifs (oui)": "up",
        "Négatifs (non)": "down",
    }
    thumb_filter = st.radio("Appréciation", list(thumb_options.keys()))

    if not df["date"].isna().all():
        date_min = df["date"].min()
        date_max = df["date"].max()
        date_range = st.date_input(
            "Période",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        )
    else:
        date_range = None

    st.divider()
    if st.button("Rafraîchir les données"):
        st.cache_data.clear()
        st.rerun()

# Appliquer les filtres
df_filtered = df.copy()

if cat_filter != "Toutes":
    df_filtered = df_filtered[df_filtered["category"] == cat_filter]

selected_thumb = thumb_options[thumb_filter]
if selected_thumb is not None:
    df_filtered = df_filtered[df_filtered["thumbs"] == selected_thumb]

if date_range and len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered["date"] >= date_range[0])
        & (df_filtered["date"] <= date_range[1])
    ]

# ─────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────

total = len(df_filtered)
nb_up = int(df_filtered["is_up"].sum())
nb_down = total - nb_up
taux = round(nb_up / total * 100, 1) if total else 0
with_detail = int(
    df_filtered["precision"].notna().sum()
    and (df_filtered["precision"].str.strip() != "").sum()
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total feedbacks", total)
col2.metric("Positifs", nb_up)
col3.metric("Négatifs", nb_down)
col4.metric("Satisfaction", f"{taux} %")
col5.metric("Avec précision", with_detail)

st.divider()

# ─────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────

col_g1, col_g2 = st.columns(2)

# --- Satisfaction par catégorie (barres groupées) ---
with col_g1:
    st.subheader("Satisfaction par catégorie")
    cat_stats = (
        df_filtered.groupby(["category", "thumbs"])
        .size()
        .reset_index(name="count")
    )
    if not cat_stats.empty:
        chart_cat = (
            alt.Chart(cat_stats)
            .mark_bar()
            .encode(
                x=alt.X(
                    "category:N",
                    title="Catégorie",
                    axis=alt.Axis(labelAngle=-20),
                ),
                y=alt.Y("count:Q", title="Nombre"),
                color=alt.Color(
                    "thumbs:N",
                    scale=alt.Scale(
                        domain=["up", "down"],
                        range=["#3ecf8e", "#e8624a"],
                    ),
                    legend=alt.Legend(title="Appréciation"),
                ),
                xOffset="thumbs:N",
                tooltip=["category", "thumbs", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_cat, use_container_width=True)
    else:
        st.caption("Pas de données.")

# --- Evolution temporelle ---
with col_g2:
    st.subheader("Évolution temporelle")
    if df_filtered["date"].notna().any():
        daily = (
            df_filtered.groupby(["date", "thumbs"])
            .size()
            .reset_index(name="count")
        )
        daily["date"] = pd.to_datetime(daily["date"])
        chart_time = (
            alt.Chart(daily)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("count:Q", title="Nombre"),
                color=alt.Color(
                    "thumbs:N",
                    scale=alt.Scale(
                        domain=["up", "down"],
                        range=["#3ecf8e", "#e8624a"],
                    ),
                    legend=alt.Legend(title="Appréciation"),
                ),
                tooltip=["date:T", "thumbs", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_time, use_container_width=True)
    else:
        st.caption("Pas de données temporelles.")

st.divider()

# --- Taux de satisfaction par catégorie ---
st.subheader("Taux de satisfaction par catégorie (%)")
sat_cat = (
    df_filtered.groupby("category")["is_up"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "positifs", "count": "total"})
    .reset_index()
)
sat_cat["taux"] = (sat_cat["positifs"] / sat_cat["total"] * 100).round(1)

if not sat_cat.empty:
    chart_taux = (
        alt.Chart(sat_cat)
        .mark_bar(color="#5b8dee")
        .encode(
            x=alt.X(
                "taux:Q",
                title="Satisfaction (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "category:N", title="Catégorie", sort="-x"
            ),
            tooltip=["category", "taux", "total"],
        )
        .properties(height=200)
    )
    st.altair_chart(chart_taux, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# TABLEAU DES FEEDBACKS
# ─────────────────────────────────────────

st.subheader("Feedbacks détaillés")

df_display = df_filtered.sort_values(
    "timestamp", ascending=False
).copy()
df_display["timestamp"] = df_display["timestamp"].dt.strftime(
    "%Y-%m-%d %H:%M"
)
df_display["question"] = df_display["question"].str[:120]
df_display["answer"] = df_display["answer"].str[:200]

st.dataframe(
    df_display[
        [
            "timestamp",
            "category",
            "thumbs",
            "question",
            "answer",
            "precision",
            "source_path",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "timestamp": st.column_config.TextColumn("Date"),
        "category": st.column_config.TextColumn("Catégorie"),
        "thumbs": st.column_config.TextColumn("Appréciation"),
        "question": st.column_config.TextColumn("Question"),
        "answer": st.column_config.TextColumn("Réponse"),
        "precision": st.column_config.TextColumn("Précision utilisateur"),
        "source_path": st.column_config.TextColumn("Source"),
    },
)

st.divider()

# ─────────────────────────────────────────
# EXPORT CSV
# ─────────────────────────────────────────

csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Télécharger les données filtrées (CSV)",
    data=csv_bytes,
    file_name=f"feedbacks_{datetime.utcnow().date()}.csv",
    mime="text/csv",
)