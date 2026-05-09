from Trading.Strategies.BollingerBandsStrategy import BollingerBandsStrategy
from Trading.Strategies.SMAStrategy import SMAStrategy
from Trading.Strategies.TwoSMAStrategy import TwoSMAStrategy
import streamlit as st

def launch_bollinger_bands_strategy():
    with st.sidebar:
        with st.expander("Параметры линий Боллинджера"):
            window = st.number_input("Размер окна", min_value=2, value=20)
            num_of_std = st.number_input("Num of std", min_value=1, value=2) 
    return BollingerBandsStrategy(window, num_of_std)

def launch_MA_strategy():
    with st.sidebar:
        with st.expander("Параметры одной скользящей средней"):
            window = st.number_input("Размер окна", min_value=2, value=20)
    return SMAStrategy(window)

def launch_twoMA_strategy():
    with st.sidebar:
        with st.expander("Параметры скользящих средних"):
            window_1 = st.number_input("Размер первого окна", min_value=2, value=12)
            window_2 = st.number_input("Размер второго окна", min_value=2, value=48)
    return TwoSMAStrategy(window_1, window_2)
