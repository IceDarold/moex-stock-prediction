from Trading.TradeStrategy import TradeStrategy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import timedelta, datetime

min_df_size = 100

class NotPreparedError(Exception):
    def __init__(self):
        super().__init__("You can not run regression without prepareLinrearRegression() function. Call it first of all and only then call other functions")
        
class NotTrainedError(Exception):
    def __init__(self):
        super().__init__("You can not get prediction without training model. Call the train_model() functionand only then get predictions")

class TooSmallDataframeError(Exception):
     def __init__(self, df_len):
        super().__init__(f"The dataframe is too small for linear regression ({df_len} rows). It has to has at least {min_df_size} rows")
      
class RegressionSettings:
    def __init__(self, prediction_data, predicted_param="Close", start_date: datetime | str =0.1, random_state=42, features_number=10, is_log=True, test_sample_size=0.1):
        """
        Args:
            prediction_data (DataFrame): The data, that uses for prediction of predicted_param with predicted_param itself for checking. 
            Needs to contain columns Open, Close, High, Low, Volume, Date. Without any if these columns, the ValueError will be arised
        """
        if predicted_param not in ["Close", "Open", "High", "Low"]: raise ValueError(f"Param {predicted_param} can not be predicted")
        self.features_number = features_number
        self.is_log = is_log
        self.random_state = random_state
        self.start_date = start_date
        self.random_state = random_state
        self.test_sample_size = test_sample_size
        if not isinstance(prediction_data, pd.DataFrame):
            raise ValueError("The type of prediction data is {}, but must be DataFrame".format(type(prediction_data)))
        self.prediction_data = prediction_data

class LinearRegressionStrategy(TradeStrategy):
    name = "Линейная регрессия"
    description = "Экспериментальная модель прогноза доходности на основе исторических OHLCV-признаков."
    def __init__(self):
        self.regression_settings = None
        self._trained_data = None
        self._prepared = False
        self._trained = False
        super().__init__()
    
    def prepareLinearRegression(self, regressionSettings: RegressionSettings):
        self.regression_settings = regressionSettings
        self.validate_dataframe(regressionSettings.prediction_data)
        if isinstance(regressionSettings.start_date, float):
            X_train, X_test, y_train, y_test = train_test_split(regressionSettings.prediction_data, regressionSettings.prediction_data["Close"], 
                                                                test_size=regressionSettings.test_sample_size, random_state=regressionSettings.random_state,
                                                                shuffle=False)
        elif isinstance(regressionSettings.start_date, datetime):
            train_df = regressionSettings.prediction_data[regressionSettings.prediction_data["Date"] < regressionSettings.start_date]
            X_train = train_df.drop("Close", axis=1)
            y_train = train_df["Close"]
            test_df = regressionSettings.prediction_data[regressionSettings.prediction_data["Date"] >= regressionSettings.start_date]
            X_test = test_df.drop("Close", axis=1)
            y_test = test_df["Close"]
        else:
            raise ValueError("The type of regressionSettings.start_date is {}, that is not acceptable. The type is need to be float or datetime.datetime".format(type(regressionSettings.start_date)))
        self._prepared = True
        self._model, self._model_accuracy, self._X_test, self._y_predicted, self._y_test = self.train_model(pd.concat([X_train, y_train], axis=1))
        self._trained_data = pd.concat([X_train, y_train])
        self.prices = pd.concat([X_test["Date"], y_test])
        self._trained = True

    def validate_dataframe(self, df, except_list=[]):
        """
        return:
            Result of validation and the error if it is
        """
        if len(df) < min_df_size:
            raise TooSmallDataframeError(len(df))
        required_columns = [item for item in ["Open", "Close", "High", "Low", "Volume", "Date"] if item not in except_list]
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"There is no param {column}")


    def _generate_features_days_ago(self, df):
        new_df = df.copy()
        for i in range(1, self.regression_settings.features_number + 1):
            if self.regression_settings.is_log:
                new_df[f"Y{i}Low"] = np.log(df["Low"].shift(i) / df["Open"].shift(i))
                new_df[f"Y{i}High"] = np.log(df["High"].shift(i) / df["Open"].shift(i))
                new_df[f"Y{i}Close"] = np.log(df["Close"].shift(i) / df["Open"].shift(i))
            else:
                new_df[f"Y{i}Low"] = df["Low"].shift(i)
                new_df[f"Y{i}High"] = df["High"].shift(i)
                new_df[f"Y{i}Close"] = df["Close"].shift(i)
        return new_df
    
    def _convert_df(self, df) -> pd.DataFrame:
        new_df = df.copy()
        new_df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        new_df["YVolume"] = df["Volume"].shift(1)
        new_df["YClose"] = df["Close"].shift(1)
        new_df = self._generate_features_days_ago(new_df)

        if self.regression_settings.is_log:
            new_df["YLow"] = np.log(df["Low"].shift(1) / df["Open"])
            new_df["YHigh"] = np.log(df["High"].shift(1) / df["Open"])
            new_df["log_lable"] =  np.log(df["Close"] / df["YClose"])
        else:
            new_df["YLow"] = df["Low"].shift(1)
            new_df["YHigh"] = df["High"].shift(1)
            new_df["log_lable"] =  df["Close"]
        new_df = new_df.drop(["Close", "Low", "High", "Volume", "YClose"], axis=1)
        return new_df
            
    def _fill(self, df, feature):
        return df[feature].fillna(df[feature].mean())

    def train_model(self, X_train, test_size=0.1):
        if not self._prepared: raise NotPreparedError()
        train_df = self._convert_df(X_train)
        X_tr, X_te, y_train, y_test = train_test_split(train_df, train_df["log_lable"], test_size=test_size, random_state=42,
                                                            shuffle=False)
        X_train = X_tr.drop(["log_lable", "Date"], axis=1)
        numeric_features = X_train.columns
        numeric_transformer = Pipeline(steps=[
            ("imputes", SimpleImputer(strategy="mean"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ]
        )
        model = Pipeline(steps=[
           ("preprocessor", preprocessor),
           ("regressor", LinearRegression())
        ])
        model.fit(X_train, y_train)
        X_test = X_te.drop(["log_lable"], axis=1)
        y_test_predict = model.predict(X_test)
        accuracy = mean_absolute_error(y_test_predict, y_test)
        return model, accuracy, X_te, y_test_predict, y_test
    
    def get_prediction(self, model, data, prepared=False):
        if not self._prepared: raise NotPreparedError()
        if not self._trained: raise NotTrainedError()
        self.validate_dataframe(data, ["Close"])
        if not prepared:
            data = data[["Open", "High", "Low", "Volume", "Date"]]
            self._convert_df(data)
        return pd.concat([model.predict(data), data["Date"]])

    def buy(self, simulation_system, date):
        predicted_df = self.get_prediction(self._model, simulation_system.actual_prices[simulation_system.actual_prices["Date"] == date])#Make sure about equality
        predicted_price_today = predicted_df.iloc(1)["Close"]
        actual_price_yestersay = simulation_system.get_price_by_date(date - timedelta(days=1))
        # predicted_price_today = actual_price_yesterday * np.exp(get_prediction(model, X_test.iloc[i]))

        # Решение о покупке или продаже
        if predicted_price_today > actual_price_yestersay and simulation_system.balance > 0:
            return True, simulation_system.balance
        return False,

    def sell(self, simulation_system, date):
        predicted_df = self.get_prediction(self._model, simulation_system.actual_prices[simulation_system.actual_prices["Date"] == date])#Make sure about equality
        predicted_price_today = predicted_df.iloc(1)["Close"]
        actual_price_yestersay = simulation_system.get_price_by_date(date - timedelta(days=1))
        # predicted_price_today = actual_price_yesterday * np.exp(get_prediction(model, X_test.iloc[i]))

        # Решение о покупке или продаже
        if predicted_price_today < actual_price_yestersay and simulation_system.shares > 0:
            return True, simulation_system.shares
        return False,
