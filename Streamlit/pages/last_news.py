from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from Utilities.Parsers import lenta_parser, ria_parser


NEWS_DIR = Path("Data/News")
SOURCES = {
    "Lenta.ru": (NEWS_DIR / "lenta.csv", lenta_parser),
    "RIA": (NEWS_DIR / "ria.csv", ria_parser),
}


def read_news(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["datetime", "title", "url"])

    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        return pd.DataFrame(columns=["datetime", "title", "url"])

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df.sort_values("datetime", ascending=False)


def default_start_date() -> datetime:
    existing_dates = []
    for path, _ in SOURCES.values():
        df = read_news(path)
        if not df.empty:
            existing_dates.append(df["datetime"].max().to_pydatetime())

    if existing_dates:
        return max(existing_dates)
    return datetime.now() - timedelta(days=1)


st.title("Парсинг новостей")
st.caption("Экспериментальный раздел для загрузки новостей Lenta.ru и РИА в локальные CSV.")

with st.sidebar:
    st.header("Параметры")
    start_date = st.date_input("Дата начала", value=default_start_date().date())
    end_date = st.date_input("Дата окончания", value=datetime.now().date())
    selected_sources = st.multiselect("Источники", list(SOURCES.keys()), default=list(SOURCES.keys()))

    if st.button("Обновить новости", disabled=not selected_sources):
        NEWS_DIR.mkdir(parents=True, exist_ok=True)
        start = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.max.time())
        for source_name in selected_sources:
            path, parser = SOURCES[source_name]
            with st.spinner(f"Загружаю {source_name}"):
                parser.parse(start, end, path)
        st.success("Новости обновлены")

for source_name, (path, _) in SOURCES.items():
    st.header(source_name)
    news_df = read_news(path)
    if news_df.empty:
        st.info("Локального файла с новостями пока нет. Нажмите 'Обновить новости' в боковой панели.")
        continue

    st.dataframe(news_df, use_container_width=True, hide_index=True)
