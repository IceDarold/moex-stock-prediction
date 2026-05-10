from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_regression import DEFAULT_DATA_DIR, load_ticker, resample_ohlcv


DEFAULT_NEWS_DIR = DEFAULT_DATA_DIR / "News"
DEFAULT_TICKERS = ["SBER"]
LIQUID_MARKET_TICKERS = (
    "SBER",
    "GAZP",
    "LKOH",
    "MOEX",
    "YNDX",
    "GMKN",
    "ROSN",
    "NVTK",
    "TATN",
    "VTBR",
    "ALRS",
    "CHMF",
    "NLMK",
    "MGNT",
    "PLZL",
    "SNGS",
    "AFLT",
    "RTKM",
    "MTSS",
    "IRAO",
    "HYDR",
    "RUAL",
    "MAGN",
    "PHOR",
    "PIKK",
)
MODEL_CHOICES = (
    "auto",
    "logistic",
    "hist-gradient-boosting",
    "random-forest",
    "extra-trees",
    "gradient-boosting",
)
AUTO_MODEL_CHOICES = (
    "logistic",
    "hist-gradient-boosting",
    "random-forest",
    "extra-trees",
    "gradient-boosting",
)
NEWS_FEATURE_COLUMNS = (
    "news_count",
    "news_sentiment_sum",
    "news_sentiment_mean",
    "news_positive_count",
    "news_negative_count",
    "news_count_lag_1",
    "news_sentiment_sum_lag_1",
    "news_count_ma_3",
    "news_sentiment_sum_ma_3",
    "news_count_ma_7",
    "news_sentiment_sum_ma_7",
)
POSITIVE_WORDS = (
    "рост",
    "растет",
    "вырос",
    "прибыль",
    "успех",
    "улучш",
    "поддерж",
    "сильн",
    "выше",
    "рекорд",
    "дивиденд",
    "сделка",
    "инвест",
    "одобр",
    "договор",
    "повыс",
)
NEGATIVE_WORDS = (
    "паден",
    "падает",
    "сниз",
    "убыт",
    "кризис",
    "санкц",
    "штраф",
    "ниже",
    "риск",
    "отказ",
    "запрет",
    "слаб",
    "сократ",
    "обвал",
    "дефицит",
    "расслед",
)


@dataclass(frozen=True)
class DirectionSelection:
    model_name: str
    selection_mode: str
    threshold: float
    top_coverage: float
    validation_accuracy: float
    validation_signal_accuracy: float
    validation_signal_coverage: float
    data: pd.DataFrame
    feature_columns: list[str]
    market_tickers_used: int
    news_rows_used: int


@dataclass(frozen=True)
class SignalSelector:
    mode: str
    threshold: float
    top_coverage: float
    accuracy: float
    coverage: float
    count: int


def resolve_market_tickers(data_dir: Path, market_tickers: list[str]) -> list[str]:
    if not market_tickers or market_tickers == ["auto"]:
        return [ticker for ticker in LIQUID_MARKET_TICKERS if (data_dir / f"{ticker}.csv").exists()]

    if market_tickers == ["all"]:
        return sorted(path.stem for path in data_dir.glob("*.csv"))

    return [ticker.upper() for ticker in market_tickers if (data_dir / f"{ticker.upper()}.csv").exists()]


def load_resampled_ticker(data_dir: Path, ticker: str, timeframe: str | None) -> pd.DataFrame:
    return resample_ohlcv(load_ticker(data_dir, ticker), timeframe)


def parse_cli_date(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def resolve_lenta_range(
    data_dir: Path,
    tickers: list[str],
    start: str | None,
    end: str | None,
    days: int,
) -> tuple[datetime, datetime]:
    resolved_end = parse_cli_date(end)
    if resolved_end is None:
        max_dates = []
        for ticker in tickers:
            try:
                df = load_ticker(data_dir, ticker.upper())
            except (FileNotFoundError, ValueError):
                continue
            max_dates.append(df["Date"].max().to_pydatetime())
        if not max_dates:
            resolved_end = datetime.now()
        else:
            resolved_end = max(max_dates)

    resolved_start = parse_cli_date(start)
    if resolved_start is None:
        resolved_start = resolved_end - timedelta(days=max(days - 1, 0))
    return resolved_start, resolved_end


def parse_lenta_news(news_dir: Path, start_date: datetime, end_date: datetime) -> Path:
    from Utilities.Parsers import lenta_parser

    news_dir.mkdir(parents=True, exist_ok=True)
    output_path = news_dir / "lenta.csv"
    lenta_parser.parse(start_date, end_date, output_path)
    return output_path


def build_market_features(
    data_dir: Path,
    market_tickers: list[str],
    timeframe: str | None,
    target_ticker: str,
    market_scope: str,
) -> tuple[pd.DataFrame, int]:
    frames = []
    for ticker in market_tickers:
        if market_scope == "exclude-target" and ticker == target_ticker:
            continue
        try:
            df = load_resampled_ticker(data_dir, ticker, timeframe)
        except (FileNotFoundError, ValueError):
            continue

        close_return = np.log(df["Close"] / df["Close"].shift(1))
        intraday_return = np.log(df["Close"] / df["Open"])
        frames.append(
            pd.DataFrame(
                {
                    "Date": df["Date"],
                    "market_return": close_return,
                    "market_intraday_return": intraday_return,
                    "market_volume": df["Volume"],
                    "market_up": (close_return > 0).astype(float),
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["Date"]), 0

    combined = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    grouped = combined.groupby("Date")
    market = pd.DataFrame(
        {
            "Date": grouped.size().index,
            "market_count": grouped.size().to_numpy(),
            "market_return_mean": grouped["market_return"].mean().to_numpy(),
            "market_return_median": grouped["market_return"].median().to_numpy(),
            "market_intraday_mean": grouped["market_intraday_return"].mean().to_numpy(),
            "market_up_share": grouped["market_up"].mean().to_numpy(),
            "market_volume_log_sum": np.log1p(grouped["market_volume"].sum().to_numpy()),
        }
    ).sort_values("Date")
    market["market_volume_change"] = market["market_volume_log_sum"].diff()

    for column in (
        "market_return_mean",
        "market_return_median",
        "market_intraday_mean",
        "market_up_share",
        "market_volume_log_sum",
        "market_volume_change",
    ):
        for lag in (1, 2, 3, 5):
            market[f"{column}_lag_{lag}"] = market[column].shift(lag)

    for window in (3, 5, 10, 20):
        market[f"market_return_mean_ma_{window}"] = (
            market["market_return_mean"].rolling(window).mean().shift(1)
        )
        market[f"market_up_share_ma_{window}"] = (
            market["market_up_share"].rolling(window).mean().shift(1)
        )
        market[f"market_volume_log_sum_ma_{window}"] = (
            market["market_volume_log_sum"].rolling(window).mean().shift(1)
        )

    return market, len(frames)


def score_news_title(title: object) -> int:
    text = str(title).lower()
    score = sum(1 for word in POSITIVE_WORDS if word in text)
    score -= sum(1 for word in NEGATIVE_WORDS if word in text)
    return score


def build_news_features(news_dir: Path) -> tuple[pd.DataFrame, int]:
    frames = []
    for path in news_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "datetime" not in df.columns or "title" not in df.columns:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        if df.empty:
            continue
        df["Date"] = df["datetime"].dt.normalize()
        df["sentiment"] = df["title"].map(score_news_title)
        frames.append(df[["Date", "sentiment"]])

    if not frames:
        return pd.DataFrame(columns=["Date", *NEWS_FEATURE_COLUMNS]), 0

    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby("Date")
    news = pd.DataFrame(
        {
            "Date": grouped.size().index,
            "news_count": grouped.size().to_numpy(),
            "news_sentiment_sum": grouped["sentiment"].sum().to_numpy(),
            "news_sentiment_mean": grouped["sentiment"].mean().to_numpy(),
            "news_positive_count": grouped["sentiment"].apply(lambda item: (item > 0).sum()).to_numpy(),
            "news_negative_count": grouped["sentiment"].apply(lambda item: (item < 0).sum()).to_numpy(),
        }
    ).sort_values("Date")
    news["news_count_lag_1"] = news["news_count"].shift(1)
    news["news_sentiment_sum_lag_1"] = news["news_sentiment_sum"].shift(1)
    news["news_count_ma_3"] = news["news_count"].rolling(3).mean().shift(1)
    news["news_sentiment_sum_ma_3"] = news["news_sentiment_sum"].rolling(3).mean().shift(1)
    news["news_count_ma_7"] = news["news_count"].rolling(7).mean().shift(1)
    news["news_sentiment_sum_ma_7"] = news["news_sentiment_sum"].rolling(7).mean().shift(1)
    return news, len(combined)


def add_missing_news_columns(data: pd.DataFrame) -> pd.DataFrame:
    for column in NEWS_FEATURE_COLUMNS:
        if column not in data.columns:
            data[column] = 0.0
    return data


def make_direction_features(
    ticker_df: pd.DataFrame,
    market_features: pd.DataFrame,
    news_features: pd.DataFrame,
    lags: int,
    target_mode: str,
    market_context: str,
    use_news: bool,
) -> pd.DataFrame:
    close_return = np.log(ticker_df["Close"] / ticker_df["Close"].shift(1))
    intraday_return = np.log(ticker_df["Close"] / ticker_df["Open"])
    overnight_return = np.log(ticker_df["Open"] / ticker_df["Close"].shift(1))
    candle_range = np.log(ticker_df["High"] / ticker_df["Low"])
    volume_log = np.log1p(ticker_df["Volume"])

    if target_mode == "next-day":
        target_return = np.log(ticker_df["Close"].shift(-1) / ticker_df["Close"])
    else:
        target_return = close_return

    columns: dict[str, pd.Series] = {
        "Date": ticker_df["Date"],
        "target_return": target_return,
        "target_direction": (target_return > 0).astype(int),
        "open_gap": overnight_return,
        "day_of_week": ticker_df["Date"].dt.dayofweek,
        "month": ticker_df["Date"].dt.month,
    }

    if target_mode == "next-day":
        columns["return_current"] = close_return
        columns["intraday_current"] = intraday_return
        columns["range_current"] = candle_range
        columns["volume_log_current"] = volume_log

    for lag in range(1, lags + 1):
        columns[f"return_lag_{lag}"] = close_return.shift(lag)
        columns[f"intraday_lag_{lag}"] = intraday_return.shift(lag)
        columns[f"overnight_lag_{lag}"] = overnight_return.shift(lag)
        columns[f"range_lag_{lag}"] = candle_range.shift(lag)
        columns[f"volume_log_lag_{lag}"] = volume_log.shift(lag)

    rolling_return = close_return if target_mode == "next-day" else close_return.shift(1)
    rolling_volume = volume_log if target_mode == "next-day" else volume_log.shift(1)
    for window in (3, 5, 10, 20, 50):
        columns[f"return_mean_{window}"] = rolling_return.rolling(window).mean()
        columns[f"return_std_{window}"] = rolling_return.rolling(window).std()
        columns[f"volume_log_mean_{window}"] = rolling_volume.rolling(window).mean()
        columns[f"close_vs_ma_{window}"] = np.log(
            ticker_df["Close"] / ticker_df["Close"].rolling(window).mean()
        ).shift(0 if target_mode == "next-day" else 1)

    data = pd.DataFrame(columns).merge(market_features, on="Date", how="left")
    if market_context == "lagged":
        same_day_market_columns = [
            "market_return_mean",
            "market_return_median",
            "market_intraday_mean",
            "market_up_share",
            "market_volume_log_sum",
            "market_volume_change",
        ]
        data = data.drop(columns=[column for column in same_day_market_columns if column in data.columns])

    if use_news:
        data = data.merge(news_features, on="Date", how="left")
    data = add_missing_news_columns(data)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
    for column in NEWS_FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    return data.dropna().reset_index(drop=True)


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if column not in {"Date", "target_return", "target_direction"}
    ]


def create_classifier(model_name: str, feature_columns: list[str]) -> Pipeline:
    if model_name == "logistic":
        transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    else:
        transformer = SimpleImputer(strategy="median")
        if model_name == "hist-gradient-boosting":
            classifier = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.03,
                l2_regularization=0.01,
                max_leaf_nodes=15,
                random_state=42,
            )
        elif model_name == "random-forest":
            classifier = RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif model_name == "extra-trees":
            classifier = ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif model_name == "gradient-boosting":
            classifier = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer([("num", transformer, feature_columns)]),
            ),
            ("classifier", classifier),
        ]
    )


def split_train_validation_test(
    data: pd.DataFrame,
    test_size: float,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_start = int(len(data) * (1 - test_size))
    validation_start = int(test_start * (1 - validation_size))
    if validation_start <= 0 or validation_start >= test_start or test_start >= len(data):
        raise ValueError("Split sizes leave an empty train, validation, or test set")
    return data.iloc[:validation_start], data.iloc[validation_start:test_start], data.iloc[test_start:]


def signal_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    selection_mode: str = "threshold",
    top_coverage: float = 100.0,
) -> tuple[float, float, int]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    if selection_mode == "top-percent":
        count = max(1, int(len(confidence) * top_coverage / 100))
        selected = np.zeros(len(confidence), dtype=bool)
        selected[np.argsort(confidence)[-count:]] = True
    else:
        selected = confidence >= threshold
    if not selected.any():
        return float("nan"), 0.0, 0
    return (
        accuracy_score(y_true[selected], predictions[selected]) * 100,
        selected.mean() * 100,
        int(selected.sum()),
    )


def tune_signal_selector(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_accuracy: float,
    min_coverage: float,
    selection_mode: str,
    max_signal_coverage: float,
) -> SignalSelector:
    full_accuracy = accuracy_score(y_true, probabilities.argmax(axis=1)) * 100
    modes = ("threshold", "top-percent") if selection_mode == "auto" else (selection_mode,)
    candidates = [
        SignalSelector(
            mode="threshold",
            threshold=0.5,
            top_coverage=100.0,
            accuracy=full_accuracy,
            coverage=100.0,
            count=len(y_true),
        )
    ]

    if "threshold" in modes:
        for threshold in np.linspace(0.5, 0.995, 100):
            signal_accuracy, coverage, count = signal_metrics(
                y_true,
                probabilities,
                threshold=float(threshold),
                selection_mode="threshold",
            )
            if count == 0 or coverage < min_coverage or coverage > max_signal_coverage:
                continue
            candidates.append(
                SignalSelector(
                    mode="threshold",
                    threshold=float(threshold),
                    top_coverage=coverage,
                    accuracy=signal_accuracy,
                    coverage=coverage,
                    count=count,
                )
            )

    if "top-percent" in modes:
        upper = min(max_signal_coverage, 100.0)
        for coverage in np.linspace(min_coverage, upper, 100):
            signal_accuracy, actual_coverage, count = signal_metrics(
                y_true,
                probabilities,
                threshold=0.5,
                selection_mode="top-percent",
                top_coverage=float(coverage),
            )
            if count == 0:
                continue
            candidates.append(
                SignalSelector(
                    mode="top-percent",
                    threshold=0.5,
                    top_coverage=float(coverage),
                    accuracy=signal_accuracy,
                    coverage=actual_coverage,
                    count=count,
                )
            )

    def rank(selector: SignalSelector) -> tuple:
        reached = selector.accuracy >= target_accuracy
        if target_accuracy >= 95:
            return reached, selector.accuracy, -selector.coverage, selector.count
        return reached, selector.coverage, selector.accuracy, selector.count

    return max(candidates, key=rank)


def evaluate_candidate(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    test_size: float,
    validation_size: float,
    target_accuracy: float,
    min_coverage: float,
    selection_mode: str,
    max_signal_coverage: float,
) -> tuple[str, SignalSelector, float]:
    train, validation, _ = split_train_validation_test(data, test_size, validation_size)
    model = create_classifier(model_name, feature_columns)
    model.fit(train[feature_columns], train["target_direction"])
    validation_probabilities = model.predict_proba(validation[feature_columns])
    validation_target = validation["target_direction"].to_numpy()
    validation_accuracy = (
        accuracy_score(validation_target, validation_probabilities.argmax(axis=1)) * 100
    )
    selector = tune_signal_selector(
        validation_target,
        validation_probabilities,
        target_accuracy=target_accuracy,
        min_coverage=min_coverage,
        selection_mode=selection_mode,
        max_signal_coverage=max_signal_coverage,
    )
    return model_name, selector, validation_accuracy


def choose_direction_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    test_size: float,
    validation_size: float,
    target_accuracy: float,
    min_coverage: float,
    selection_mode: str,
    max_signal_coverage: float,
    market_tickers_used: int,
    news_rows_used: int,
) -> DirectionSelection:
    model_names = AUTO_MODEL_CHOICES if model_name == "auto" else (model_name,)
    choices = [
        evaluate_candidate(
            data=data,
            feature_columns=feature_columns,
            model_name=candidate,
            test_size=test_size,
            validation_size=validation_size,
            target_accuracy=target_accuracy,
            min_coverage=min_coverage,
            selection_mode=selection_mode,
            max_signal_coverage=max_signal_coverage,
        )
        for candidate in model_names
    ]

    def rank(choice: tuple[str, SignalSelector, float]) -> tuple:
        candidate_model, selector, validation_accuracy = choice
        reached = selector.accuracy >= target_accuracy
        if target_accuracy >= 95:
            model_priority = -AUTO_MODEL_CHOICES.index(candidate_model)
            return reached, selector.accuracy, -selector.coverage, selector.count, model_priority
        return reached, selector.coverage, selector.accuracy, validation_accuracy

    selected = max(choices, key=rank)
    selected_model_name, selected_selector, selected_validation_accuracy = selected
    return DirectionSelection(
        model_name=selected_model_name,
        selection_mode=selected_selector.mode,
        threshold=selected_selector.threshold,
        top_coverage=selected_selector.top_coverage,
        validation_accuracy=selected_validation_accuracy,
        validation_signal_accuracy=selected_selector.accuracy,
        validation_signal_coverage=selected_selector.coverage,
        data=data,
        feature_columns=feature_columns,
        market_tickers_used=market_tickers_used,
        news_rows_used=news_rows_used,
    )


def train_and_evaluate_direction(
    data_dir: Path,
    news_dir: Path,
    ticker: str,
    market_tickers: list[str],
    timeframe: str | None,
    lags: int,
    test_size: float,
    validation_size: float,
    model_name: str,
    target_mode: str,
    market_scope: str,
    market_context: str,
    use_news: bool,
    target_accuracy: float,
    min_coverage: float,
    selection_mode: str,
    max_signal_coverage: float,
) -> dict[str, object]:
    ticker_df = load_resampled_ticker(data_dir, ticker, timeframe)
    market_features, market_tickers_used = build_market_features(
        data_dir=data_dir,
        market_tickers=market_tickers,
        timeframe=timeframe,
        target_ticker=ticker,
        market_scope=market_scope,
    )
    news_features, news_rows_used = build_news_features(news_dir) if use_news else (
        pd.DataFrame(columns=["Date", *NEWS_FEATURE_COLUMNS]),
        0,
    )
    data = make_direction_features(
        ticker_df=ticker_df,
        market_features=market_features,
        news_features=news_features,
        lags=lags,
        target_mode=target_mode,
        market_context=market_context,
        use_news=use_news,
    )
    if len(data) < 200:
        raise ValueError(f"Too few samples after feature generation: {len(data)}")

    feature_columns = get_feature_columns(data)
    selection = choose_direction_model(
        data=data,
        feature_columns=feature_columns,
        model_name=model_name,
        test_size=test_size,
        validation_size=validation_size,
        target_accuracy=target_accuracy,
        min_coverage=min_coverage,
        selection_mode=selection_mode,
        max_signal_coverage=max_signal_coverage,
        market_tickers_used=market_tickers_used,
        news_rows_used=news_rows_used,
    )
    train, validation, test = split_train_validation_test(data, test_size, validation_size)
    train_with_validation = pd.concat([train, validation])
    model = create_classifier(selection.model_name, feature_columns)
    model.fit(train_with_validation[feature_columns], train_with_validation["target_direction"])

    test_target = test["target_direction"].to_numpy()
    test_probabilities = model.predict_proba(test[feature_columns])
    test_predictions = test_probabilities.argmax(axis=1)
    test_accuracy = accuracy_score(test_target, test_predictions) * 100
    signal_accuracy, signal_coverage, signal_count = signal_metrics(
        test_target,
        test_probabilities,
        selection.threshold,
        selection_mode=selection.selection_mode,
        top_coverage=selection.top_coverage,
    )
    majority_baseline = max(test_target.mean(), 1 - test_target.mean()) * 100
    news_days_in_model = int((selection.data["news_count"] > 0).sum()) if "news_count" in selection.data else 0

    return {
        "samples": len(data),
        "train": len(train_with_validation),
        "test": len(test),
        "test_from": test["Date"].iloc[0],
        "test_to": test["Date"].iloc[-1],
        "model": selection.model_name,
        "selection_mode": selection.selection_mode,
        "threshold": selection.threshold,
        "top_coverage": selection.top_coverage,
        "validation_accuracy": selection.validation_accuracy,
        "validation_signal_accuracy": selection.validation_signal_accuracy,
        "validation_signal_coverage": selection.validation_signal_coverage,
        "test_accuracy": test_accuracy,
        "signal_accuracy": signal_accuracy,
        "signal_coverage": signal_coverage,
        "signal_count": signal_count,
        "majority_baseline": majority_baseline,
        "target_reached": signal_accuracy >= target_accuracy if not np.isnan(signal_accuracy) else False,
        "target_mode": target_mode,
        "market_scope": market_scope,
        "market_context": market_context,
        "market_tickers_used": selection.market_tickers_used,
        "news_rows_used": selection.news_rows_used,
        "news_days_in_model": news_days_in_model,
    }


def print_result(ticker: str, result: dict[str, object], target_accuracy: float) -> None:
    status = "OK" if result["target_reached"] else "MISS"
    print(f"\n{ticker}")
    print(
        f"rows={result['samples']} train={result['train']} test={result['test']} "
        f"test_period={result['test_from']} .. {result['test_to']}"
    )
    print(
        f"model={result['model']} selection={result['selection_mode']} "
        f"threshold={result['threshold']:.3f} top_coverage={result['top_coverage']:.2f}% "
        f"target_mode={result['target_mode']} market_scope={result['market_scope']} "
        f"market_context={result['market_context']}"
    )
    print(
        f"market_tickers={result['market_tickers_used']} "
        f"news_rows={result['news_rows_used']} news_days_in_model={result['news_days_in_model']}"
    )
    print(
        f"validation: full_accuracy={result['validation_accuracy']:.2f}% | "
        f"signal_accuracy={result['validation_signal_accuracy']:.2f}% | "
        f"signal_coverage={result['validation_signal_coverage']:.2f}%"
    )
    print(
        f"test: full_accuracy={result['test_accuracy']:.2f}% | "
        f"signal_accuracy={result['signal_accuracy']:.2f}% | "
        f"signal_coverage={result['signal_coverage']:.2f}% | "
        f"signals={result['signal_count']} | majority_baseline={result['majority_baseline']:.2f}%"
    )
    print(f"target_{target_accuracy:.0f}%={status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a direction classifier with market volume and optional news features."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--news-dir", type=Path, default=DEFAULT_NEWS_DIR)
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument("--lags", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="auto")
    parser.add_argument(
        "--target-mode",
        choices=("same-day", "next-day"),
        default="same-day",
        help="same-day uses current market context to classify the current candle; next-day predicts the next candle.",
    )
    parser.add_argument(
        "--market-scope",
        choices=("all", "exclude-target"),
        default="all",
        help="all is the strongest market-regime mode; exclude-target avoids target ticker in market aggregates.",
    )
    parser.add_argument(
        "--market-context",
        choices=("same-day", "lagged"),
        default="same-day",
        help="Use same-day market aggregates or only their lagged/rolling values.",
    )
    parser.add_argument(
        "--market-tickers",
        nargs="*",
        default=["auto"],
        help="auto uses a liquid preset; all uses every CSV in data-dir; otherwise pass ticker symbols.",
    )
    parser.add_argument("--target-accuracy", type=float, default=70.0)
    parser.add_argument("--min-coverage", type=float, default=5.0)
    parser.add_argument(
        "--selection-mode",
        choices=("auto", "threshold", "top-percent"),
        default="auto",
        help="auto uses confidence thresholds for normal targets and top-percent for very high targets.",
    )
    parser.add_argument(
        "--max-signal-coverage",
        type=float,
        default=None,
        help="Upper bound for selected signal coverage. Defaults to 0.5%% for targets >=95%%, otherwise 100%%.",
    )
    parser.add_argument(
        "--parse-lenta-news",
        action="store_true",
        help="Fetch Lenta.ru archive rows into Data/News/lenta.csv before training.",
    )
    parser.add_argument("--news-start", default=None, help="Lenta start date: YYYY-MM-DD.")
    parser.add_argument("--news-end", default=None, help="Lenta end date: YYYY-MM-DD.")
    parser.add_argument(
        "--news-days",
        type=int,
        default=7,
        help="How many days to fetch when --news-start is omitted.",
    )
    parser.add_argument("--no-news", dest="use_news", action="store_false")
    parser.set_defaults(use_news=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selection_mode == "auto":
        selection_mode = "top-percent" if args.target_accuracy >= 95 else "threshold"
    else:
        selection_mode = args.selection_mode
    max_signal_coverage = (
        args.max_signal_coverage
        if args.max_signal_coverage is not None
        else (0.5 if args.target_accuracy >= 95 else 100.0)
    )
    min_coverage = min(args.min_coverage, max_signal_coverage)

    if args.parse_lenta_news:
        news_start, news_end = resolve_lenta_range(
            data_dir=args.data_dir,
            tickers=[ticker.upper() for ticker in args.tickers],
            start=args.news_start,
            end=args.news_end,
            days=args.news_days,
        )
        output_path = parse_lenta_news(args.news_dir, news_start, news_end)
        print(
            f"parsed_lenta={output_path} "
            f"period={news_start.date()}..{news_end.date()}"
        )

    market_tickers = resolve_market_tickers(args.data_dir, args.market_tickers)
    for ticker in args.tickers:
        result = train_and_evaluate_direction(
            data_dir=args.data_dir,
            news_dir=args.news_dir,
            ticker=ticker.upper(),
            market_tickers=market_tickers,
            timeframe=args.timeframe,
            lags=args.lags,
            test_size=args.test_size,
            validation_size=args.validation_size,
            model_name=args.model,
            target_mode=args.target_mode,
            market_scope=args.market_scope,
            market_context=args.market_context,
            use_news=args.use_news,
            target_accuracy=args.target_accuracy,
            min_coverage=min_coverage,
            selection_mode=selection_mode,
            max_signal_coverage=max_signal_coverage,
        )
        print_result(ticker.upper(), result, args.target_accuracy)


if __name__ == "__main__":
    main()
