import streamlit as st

st.text("Main menu")    
st.page_link("pages/trade.py", label="Симуляция торговли")
st.page_link("pages/parse_stock.py", label="Парсинг котировок")
st.page_link("pages/last_news.py", label="Парсинг новостей")
