import streamlit as st
from utilities import get_tickers
from Trading.Strategies.RandomStrategy import RandomStrategy
from Trading.Strategies.SMAStrategy import SMAStrategy
from Trading.Strategies.TwoSMAStrategy import TwoSMAStrategy
from Trading.Strategies.BollingerBandsStrategy import BollingerBandsStrategy
from Trading.SimulationSystem import SimulationSystem
from tickers import get_ticker_data
import plotly.graph_objects as go
from Config import config
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from Trading.launch_strategies import (
    launch_bollinger_bands_strategy,
    launch_MA_strategy,
    launch_twoMA_strategy,
)


def pluralize(value, forms):
    """
    Функция для подбора правильного склонения слова.
    forms - это список из трех форм слова: для 1, для 2-4, для 5-0.
    """
    if value % 10 == 1 and value % 100 != 11:
        form = forms[0]
    elif 2 <= value % 10 <= 4 and (value % 100 < 10 or value % 100 >= 20):
        form = forms[1]
    else:
        form = forms[2]
    return f"{value} {form}"

def calculate_date_difference(date1, date2):
    # Преобразование строк в объекты datetime
    date_format = "%d-%m-%Y"
    if isinstance(date1, str):
        date1 = datetime.strptime(date1, date_format)
    if isinstance(date2, str):
        date2 = datetime.strptime(date2, date_format)

    # Вычисление разницы между датами
    diff = relativedelta(date2, date1)

    # Формирование строки результата
    result = []
    if diff.years != 0:
        result.append(pluralize(diff.years, ["год", "года", "лет"]))
    if diff.months != 0:
        result.append(pluralize(diff.months, ["месяц", "месяца", "месяцев"]))
    if diff.days != 0:
        result.append(pluralize(diff.days, ["день", "дня", "дней"]))
    if len(result) > 1:
        return ", ".join(result[:-1]) + " и " + result[-1]
    elif len(result) == 1:
        return result[0]
    else:
        return "0 дней"

def draw_date(label, df, min_date=-1, value=-1) -> datetime:
    if min_date == -1:
        coefficient = 0.5
        dividing_index = int(len(df) * coefficient)
        min_date = df.iloc[dividing_index]["Date"]
        min_date = pd.to_datetime(min_date)
    start_date = pd.to_datetime(st.sidebar.date_input(label, min_value=min_date, max_value=df.iloc[-1]["Date"], value=value if value != -1 else min_date))
    return start_date

def resample_data(df, timeframe):
    df_resampled = df.set_index('Date')
    df_resampled = df_resampled.resample(timeframe).agg({'Open': 'first', 
                                               'High': 'max', 
                                               'Low': 'min', 
                                               'Close': 'last',
                                               'Volume': 'sum'}).dropna().reset_index()
    return df_resampled

def get_sma(param: pd.Series, window=20):
    df = param.copy()
    if len(param) < window: raise ValueError(f"The size of series ({len(param)}) is lower then window ({window})")
    df["Close"] = df["Close"].rolling(window=window).mean()
    return df

def draw_graph(df, ticker, title, key="Graph", window_1=-1, window_2=-1):
    st.title(title)
    timeframe = st.selectbox('Выберите таймфрейм', ['10T', '30T', '1H', '1D'], index=3, key=key)
    graph_type = st.selectbox('Выберите вид графика', ["Линия", "Свечи"], index=1, key=key + '1')
    df_resampled = resample_data(df, timeframe)
    if timeframe in ['10T', '30T', '1H']:
        start_date = df_resampled['Date'].max() - pd.Timedelta(days=7)
        end_date = df_resampled['Date'].max()
        range_x = [start_date, end_date]

        # Определение диапазона для оси Y
        visible_data = df_resampled[(df_resampled['Date'] >= start_date) & (df_resampled['Date'] <= end_date)]
        range_y = [visible_data['Low'].min(), visible_data['High'].max()]
    else:
        range_x = [df_resampled['Date'].min(), df_resampled['Date'].max()]
        range_y = [df_resampled['Low'].min(), df_resampled['High'].max()]
    if graph_type == "Линия":
        fig = go.Figure(data=go.Scatter(x=df_resampled["Date"], 
                                       y=df_resampled["Close"],
                                       mode='lines', 
                                       line=dict(color=config.visualization.color, width=config.visualization.width)))
        title = f'График цены закрытия акций {ticker}'
    # Определение начального и конечного диапазона для мелких таймфреймов
    else:
        fig = go.Figure(data=go.Candlestick(x=df_resampled['Date'],
                                            open=df_resampled['Open'],
                                            high=df_resampled['High'],
                                            low=df_resampled['Low'],
                                            close=df_resampled['Close'],
                                            name='Свечной график'))
        title = f'Свечной график цены закрытия акций {ticker}'

    if window_1 != -1:
        sma_1 = get_sma(df_resampled, window=window_1)
        fig.add_trace(go.Scatter(x=sma_1["Date"], y=sma_1["Close"], name=f'{window_1}-ти фреймовая скользящая средняя'))
    if window_2 != -1:
        sma_2 = get_sma(df_resampled, window=window_2)
        fig.add_trace(go.Scatter(x=sma_2["Date"], y=sma_2["Close"], name=f'{window_2}-ти фреймовая скользящая средняя'))
    # Настройка оформления графика
    fig.update_layout(
        title=title,
        xaxis_title='Дата',
        yaxis_title='Цена',
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date",
            range=range_x
        ),
        yaxis=dict(
            title='Цена',
            titlefont=dict(size=14),
            tickfont=dict(size=12),
            range=range_y
        ),
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    go_config = {
            'scrollZoom': True,  # Отключить масштабирование при прокрутке
            'displayModeBar': True,  # Показать панель инструментов
            'doubleClick': 'reset',  # Настроить двойной клик для сброса зума
            'showTips': False,  # Отключить подсказки при наведении
            'displaylogo': False,  # Отключить логотип Plotly
            'modeBarButtonsToRemove': ['zoom2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],  # Убрать кнопки zoom
            'modeBarButtons': [['pan2d']],  # Установить пан по умолчанию и добавить кнопку для сохранения изображения
            'responsive': True  # Адаптивный график
    }
    st.plotly_chart(fig, config=go_config, key=key+"2")

def draw():
    strategy_list = ["None"]
    strategy_list.extend([strategy_item.name for strategy_item in trade_strategies.keys()])
    tickers_list = ["None"]
    tickers_list.extend(get_tickers())
    filtered_df = None
    st.sidebar.title("Параметры графика")
    ticker = st.sidebar.selectbox("Выберите котировку:", tickers_list, index=1)
    trade_strategy = st.sidebar.selectbox("Стратегия торговли", strategy_list, index=1)
    strategy_class = None
    if trade_strategy != "None":
        for item in trade_strategies.keys():
            if item.name == trade_strategy: 
                strategy_class = item
        st.sidebar.caption(strategy_class.description)
    df = get_ticker_data(ticker)
    if not df is None:
        start_date = draw_date("Дата начала торговли", df)
        end_date = draw_date("Дата окончания торговли", df, start_date, value=df.iloc[-1].Date)
    else:
        start_date = datetime(1950, 1, 1)
        end_date = datetime.now()
    commission = st.sidebar.slider("Коммиссия (в %)", 0.0, 5.0, step=0.01, value=0.0) / 100
    initial_balance = st.sidebar.number_input("Начальный баланс", min_value=0, max_value=1000000, value=1000)
    sma = st.sidebar.toggle("Show SMA")

    if df is None:
        st.subheader("Для отображения информации выберите котировку с меню слева")
    else:
        filtered_df = df[(start_date < df['Date']) & (df['Date'] < end_date)]
        if filtered_df.empty:
            st.warning("В выбранном периоде нет данных")
        else:
            draw_graph(filtered_df, ticker, f'График акций {ticker} за выбранный вами период', key="Date chosen",
                       window_1=12 if sma else -1,
                       window_2=48 if sma else -1)
    strategy = None
    if strategy_class is not None:
        strategy_factory = trade_strategies[strategy_class]
        strategy = strategy_factory() if strategy_factory is not None else strategy_class()
    # Кнопка для начала торговли
    trade_disabled = (
        ticker == 'None'
        or strategy is None
        or initial_balance == 0
        or filtered_df is None
        or filtered_df.empty
    )
    if st.sidebar.button("Start trade", key="Trade button", disabled=trade_disabled):
        simulation_system: SimulationSystem = SimulationSystem(
            filtered_df,
            initial_balance=initial_balance,
            commission=commission
            )
        history = simulation_system.simulate_trading(strategy)
        st.title(f"График вашего баланса")
        sc = go.Figure(data=go.Scatter(x=history["balance_over_time"]["Date"], y=history["balance_over_time"]["Balance"], mode='lines'))
        st.plotly_chart(sc)
        st.text(f'Всего проведено сделок по покупке {len(history["purchases"]["Date"])} штук')
        st.text(f'Всего проведено сделок по продаже {len(history["sales"]["Date"])} штук')
        st.text(f'Всего проведено сделок {len(history["purchases"]["Date"]) + len(history["sales"]["Date"])} штук')
        end_balance = history["balance_over_time"]["Balance"][-1]
        date_diff = calculate_date_difference(history["balance_over_time"]["Date"][0], history["balance_over_time"]["Date"][-1])
        start_phrase = "Общая прибыль за" if end_balance - initial_balance > 0 else "Общий убыток за"
        end_phrase = "составила" if end_balance - initial_balance > 0 else "составил"
        st.text(f'{start_phrase} {date_diff} {end_phrase} {round(abs(end_balance - initial_balance), 2)} рублей или {round(abs(end_balance / initial_balance - 1    ) * 100, 2)}%')

trade_strategies = {
    RandomStrategy: None,
    BollingerBandsStrategy: launch_bollinger_bands_strategy,
    SMAStrategy: launch_MA_strategy,
    TwoSMAStrategy: launch_twoMA_strategy,
    }
draw()
