from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "Data"
TARGET_BASIS_CHOICES = ("prev-close", "open")
MODEL_CHOICES = (
    "auto",
    "linear",
    "ridge",
    "huber",
    "hist-gradient-boosting",
    "random-forest",
)
AUTO_MODEL_CHOICES = ("ridge", "huber", "hist-gradient-boosting")


@dataclass(frozen=True)
class ModelSelection:
    model_name: str
    target_basis: str
    validation_mae: float
    blend_weight: float
    data: pd.DataFrame
    feature_columns: list[str]


def load_ticker(data_dir: Path, ticker: str) -> pd.DataFrame:
    path = data_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data file for ticker {ticker}: {path}")

    df = pd.read_csv(path)
    required_columns = {"Open", "High", "Low", "Close", "Volume", "Date"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{ticker} is missing columns: {sorted(missing_columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=list(required_columns))
    return df.sort_values("Date").reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str | None) -> pd.DataFrame:
    if not timeframe:
        return df

    return (
        df.set_index("Date")
        .resample(timeframe)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )


def make_features(
    df: pd.DataFrame,
    lags: int,
    target_basis: str = "prev-close",
) -> pd.DataFrame:
    close_return = np.log(df["Close"] / df["Close"].shift(1))
    intraday_return = np.log(df["Close"] / df["Open"])
    overnight_return = np.log(df["Open"] / df["Close"].shift(1))
    candle_range = np.log(df["High"] / df["Low"])
    volume_log = np.log1p(df["Volume"])

    columns: dict[str, pd.Series] = {
        "Date": df["Date"],
        "Open": df["Open"],
        "Close_prev_1": df["Close"].shift(1),
        "gap_log": overnight_return,
        "day_of_week": df["Date"].dt.dayofweek,
        "month": df["Date"].dt.month,
    }

    for lag in range(1, lags + 1):
        columns[f"return_lag_{lag}"] = close_return.shift(lag)
        columns[f"intraday_return_lag_{lag}"] = intraday_return.shift(lag)
        columns[f"overnight_return_lag_{lag}"] = overnight_return.shift(lag)
        columns[f"range_lag_{lag}"] = candle_range.shift(lag)
        columns[f"volume_log_lag_{lag}"] = volume_log.shift(lag)
        columns[f"volume_change_lag_{lag}"] = np.log(
            df["Volume"].shift(lag) / df["Volume"].shift(lag + 1)
        )

    for window in (3, 5, 10, 20):
        columns[f"return_mean_{window}"] = close_return.shift(1).rolling(window).mean()
        columns[f"return_std_{window}"] = close_return.shift(1).rolling(window).std()
        columns[f"intraday_mean_{window}"] = (
            intraday_return.shift(1).rolling(window).mean()
        )
        columns[f"range_mean_{window}"] = candle_range.shift(1).rolling(window).mean()
        columns[f"volume_mean_{window}"] = volume_log.shift(1).rolling(window).mean()

    if target_basis == "open":
        columns["target_log_return"] = intraday_return
    elif target_basis == "prev-close":
        columns["target_log_return"] = close_return
    else:
        raise ValueError(f"Unknown target basis: {target_basis}")

    columns["target_close"] = df["Close"]
    return (
        pd.DataFrame(columns)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )


def create_regressor(model_name: str):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=10.0)
    if model_name == "huber":
        return HuberRegressor(max_iter=5000, epsilon=1.35)
    if model_name == "hist-gradient-boosting":
        return HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            l2_regularization=1.0,
            max_leaf_nodes=7,
            random_state=42,
        )
    if model_name == "random-forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if column not in {"Date", "target_log_return", "target_close"}
    ]


def create_pipeline(feature_columns: list[str], model_name: str) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("num", SimpleImputer(strategy="median"), feature_columns)]
                ),
            ),
            ("regressor", create_regressor(model_name)),
        ]
    )


def split_tail(data: pd.DataFrame, tail_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < tail_size < 1:
        raise ValueError(f"Split size must be between 0 and 1, got {tail_size}")

    split_index = int(len(data) * (1 - tail_size))
    if split_index <= 0 or split_index >= len(data):
        raise ValueError(f"Split size {tail_size} leaves an empty train or test set")

    return data.iloc[:split_index], data.iloc[split_index:]


def predict_close(
    data: pd.DataFrame,
    predicted_log_return: np.ndarray,
    target_basis: str,
) -> np.ndarray:
    if target_basis == "open":
        anchor = data["Open"].to_numpy()
    else:
        anchor = data["Close_prev_1"].to_numpy()
    return anchor * np.exp(predicted_log_return)


def optimize_blend_weight(
    actual_close: np.ndarray,
    model_close: np.ndarray,
    baseline_close: np.ndarray,
    enabled: bool,
) -> tuple[float, float]:
    if not enabled:
        return 1.0, mean_absolute_error(actual_close, model_close)

    weights = np.linspace(0, 1, 21)
    maes = [
        mean_absolute_error(actual_close, weight * model_close + (1 - weight) * baseline_close)
        for weight in weights
    ]
    best_index = int(np.argmin(maes))
    return float(weights[best_index]), float(maes[best_index])


def evaluate_candidate_on_validation(
    data: pd.DataFrame,
    model_name: str,
    target_basis: str,
    test_size: float,
    validation_size: float,
    blend_baseline: bool,
) -> ModelSelection:
    if len(data) < 100:
        raise ValueError(f"Too few samples after feature generation: {len(data)}")

    train, _ = split_tail(data, test_size)
    fit, validation = split_tail(train, validation_size)
    feature_columns = get_feature_columns(data)
    model = create_pipeline(feature_columns, model_name)
    model.fit(fit[feature_columns], fit["target_log_return"])

    model_close = predict_close(
        validation,
        model.predict(validation[feature_columns]),
        target_basis,
    )
    baseline_close = validation["Close_prev_1"].to_numpy()
    blend_weight, validation_mae = optimize_blend_weight(
        validation["target_close"].to_numpy(),
        model_close,
        baseline_close,
        blend_baseline,
    )

    return ModelSelection(
        model_name=model_name,
        target_basis=target_basis,
        validation_mae=validation_mae,
        blend_weight=blend_weight,
        data=data,
        feature_columns=feature_columns,
    )


def select_model(
    df: pd.DataFrame,
    lags: int,
    test_size: float,
    validation_size: float,
    model_name: str,
    target_basis: str,
    blend_baseline: bool,
) -> ModelSelection:
    if model_name == "auto":
        models = AUTO_MODEL_CHOICES
        target_bases = TARGET_BASIS_CHOICES if target_basis == "auto" else (target_basis,)
    else:
        models = (model_name,)
        target_bases = ("prev-close",) if target_basis == "auto" else (target_basis,)

    selections: list[ModelSelection] = []
    prepared_data = {
        basis: make_features(df, lags, basis)
        for basis in target_bases
    }
    for basis, data in prepared_data.items():
        for candidate_model_name in models:
            selections.append(
                evaluate_candidate_on_validation(
                    data=data,
                    model_name=candidate_model_name,
                    target_basis=basis,
                    test_size=test_size,
                    validation_size=validation_size,
                    blend_baseline=blend_baseline,
                )
            )

    return min(selections, key=lambda selection: selection.validation_mae)


def train_and_evaluate(
    df: pd.DataFrame,
    lags: int,
    test_size: float,
    validation_size: float,
    model_name: str,
    target_basis: str,
    blend_baseline: bool,
) -> dict[str, object]:
    selection = select_model(
        df=df,
        lags=lags,
        test_size=test_size,
        validation_size=validation_size,
        model_name=model_name,
        target_basis=target_basis,
        blend_baseline=blend_baseline,
    )
    data = selection.data
    if len(data) < 100:
        raise ValueError(f"Too few samples after feature generation: {len(data)}")

    train, test = split_tail(data, test_size)
    model = create_pipeline(selection.feature_columns, selection.model_name)
    model.fit(train[selection.feature_columns], train["target_log_return"])

    model_close = predict_close(
        test,
        model.predict(test[selection.feature_columns]),
        selection.target_basis,
    )
    baseline_close = test["Close_prev_1"].to_numpy()
    baseline_open = test["Open"].to_numpy()
    predicted_close = (
        selection.blend_weight * model_close
        + (1 - selection.blend_weight) * baseline_close
    )
    actual_close = test["target_close"].to_numpy()

    actual_direction = np.sign(actual_close - baseline_close)
    predicted_direction = np.sign(predicted_close - baseline_close)
    non_flat = actual_direction != 0
    direction_accuracy = (
        (predicted_direction[non_flat] == actual_direction[non_flat]).mean() * 100
        if non_flat.any()
        else float("nan")
    )

    mae = mean_absolute_error(actual_close, predicted_close)
    baseline_mae = mean_absolute_error(actual_close, baseline_close)
    baseline_open_mae = mean_absolute_error(actual_close, baseline_open)
    best_baseline_mae = min(baseline_mae, baseline_open_mae)
    best_baseline_name = (
        "prev close" if baseline_mae <= baseline_open_mae else "open"
    )

    return {
        "samples": len(data),
        "train": len(train),
        "test": len(test),
        "test_from": test["Date"].iloc[0],
        "test_to": test["Date"].iloc[-1],
        "model": selection.model_name,
        "target_basis": selection.target_basis,
        "validation_mae": selection.validation_mae,
        "blend_weight": selection.blend_weight,
        "mae": mae,
        "rmse": mean_squared_error(actual_close, predicted_close) ** 0.5,
        "mape": np.mean(np.abs((actual_close - predicted_close) / actual_close)) * 100,
        "r2": r2_score(actual_close, predicted_close),
        "baseline_mae": baseline_mae,
        "baseline_rmse": mean_squared_error(actual_close, baseline_close) ** 0.5,
        "baseline_open_mae": baseline_open_mae,
        "baseline_open_rmse": mean_squared_error(actual_close, baseline_open) ** 0.5,
        "best_baseline_name": best_baseline_name,
        "best_baseline_mae": best_baseline_mae,
        "mae_improvement_pct": (baseline_mae - mae) / baseline_mae * 100,
        "best_baseline_improvement_pct": (
            (best_baseline_mae - mae) / best_baseline_mae * 100
        ),
        "direction_accuracy": direction_accuracy,
    }


def print_result(ticker: str, rows: int, result: dict[str, object]) -> None:
    print(f"\n{ticker}")
    print(
        f"rows={rows} samples={result['samples']} "
        f"train={result['train']} test={result['test']}"
    )
    print(
        f"model={result['model']} target_basis={result['target_basis']} | "
        f"validation_MAE={result['validation_mae']:.4f} RUB | "
        f"baseline_blend_weight={result['blend_weight']:.2f}"
    )
    print(f"test_period={result['test_from']} .. {result['test_to']}")
    print(
        f"MAE={result['mae']:.4f} RUB | "
        f"RMSE={result['rmse']:.4f} RUB | "
        f"MAPE={result['mape']:.4f}% | "
        f"R2={result['r2']:.5f}"
    )
    print(
        f"baseline_prev_close: MAE={result['baseline_mae']:.4f} RUB | "
        f"RMSE={result['baseline_rmse']:.4f} RUB"
    )
    print(
        f"baseline_open: MAE={result['baseline_open_mae']:.4f} RUB | "
        f"RMSE={result['baseline_open_rmse']:.4f} RUB"
    )
    print(
        f"MAE improvement vs prev close={result['mae_improvement_pct']:.2f}% | "
        f"vs best baseline ({result['best_baseline_name']})="
        f"{result['best_baseline_improvement_pct']:.2f}% | "
        f"direction_accuracy={result['direction_accuracy']:.2f}%"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate an OHLCV price model.")
    parser.add_argument("--tickers", nargs="+", default=["SBER"], help="Ticker symbols to train on.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with ticker CSV files.")
    parser.add_argument("--lags", type=int, default=20, help="Number of lagged candles to use.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tail share used as test data.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Tail share of train data used for model selection.")
    parser.add_argument("--timeframe", default=None, help="Optional pandas resample timeframe, e.g. 1D.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="auto", help="Model to train. auto validates several candidates.")
    parser.add_argument(
        "--target-basis",
        choices=("auto", *TARGET_BASIS_CHOICES),
        default="auto",
        help="Predict return from previous close or from current open. auto tries both for --model auto.",
    )
    parser.add_argument(
        "--no-blend",
        dest="blend_baseline",
        action="store_false",
        help="Disable validation-tuned blending with the previous-close baseline.",
    )
    parser.set_defaults(blend_baseline=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ticker in args.tickers:
        df = load_ticker(args.data_dir, ticker)
        df = resample_ohlcv(df, args.timeframe)
        result = train_and_evaluate(
            df=df,
            lags=args.lags,
            test_size=args.test_size,
            validation_size=args.validation_size,
            model_name=args.model,
            target_basis=args.target_basis,
            blend_baseline=args.blend_baseline,
        )
        print_result(ticker, len(df), result)


if __name__ == "__main__":
    main()
