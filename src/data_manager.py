import time
import uuid
import plotly.express as px
import polars as pl
import pandas as pd
from typing import Union, Optional, List
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly
import statsmodels.api as sm

from methodtools import lru_cache

COLUMNS = ['temperature', 'season', 'city', 'timestamp']
FILES_DIR = 'files'


class DataManager:

    def __init__(self, file_path: str = None, parallel: bool = True) -> None:
        if file_path is None:
            file_path = 'temperature_data.csv'
        self.file_path = file_path
        self.parallel = parallel
        # self.files_dir = os.path.join(os.path.dirname(__file__), 'files_folder')
        self.files_dir = os.path.join("./", 'files_folder')
        self.check_files_dir()
        self.data = self.check_data(self.load_data(file_path))
        self.table = self.process_data()


    def check_files_dir(self):
        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)

    def load_file(self, file_data):
        file_name = str(uuid.uuid4())
        file_path = f"{os.path.join(self.files_dir, file_name)}.csv"
        with open(file_path, "wb") as f:
            f.write(file_data)

        return file_path

    def upload_streamlit_file(self, file: bytes):
        file_path = self.load_file(file)
        return self.load_data(file_path)

    def load_data(self, file_path: str = None) -> Optional[Union[pl.DataFrame, pd.DataFrame]]:
        print("file_path", file_path)
        if file_path is None:
            file_path = self.file_path
        try:
            if self.parallel:
                return pl.read_csv(file_path)
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error during data loading: {e}")
            return None

    def check_data(self, data) -> None:
        try:
            for col in COLUMNS:
                _ = data[col]
            return data
        except Exception as e:
            print(f"Error during data check: {e}")
            return None

    @lru_cache()
    def get_rolling_mean(self, city: str, days: int = 30) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            if self.parallel:
                return self.data.filter(pl.col("city") == city) \
                    .select(pl.col("temperature") \
                            .rolling_mean(window_size=days).alias("rolling_mean"))["rolling_mean"]
            else:
                return self.data[self.data["city"] == city]["temperature"] \
                    .rolling(window=days).mean().reset_index(drop=True)
        except Exception as e:
            print(f"Error calculating rolling mean: {e}")
            return None

    @lru_cache()
    def get_rolling_std(self, city: str, days: int = 30) -> Optional[Union[pl.Series, pd.Series]]:
        try:
            if self.parallel:
                return self.get_data_by_city(city) \
                    .select(pl.col("temperature") \
                            .rolling_std(window_size=days).alias("rolling_std"))["rolling_std"]
            return self.get_data_by_city(city) \
                .apply(lambda g: g[g['city'] == city])["temperature"] \
                .rolling(days).std()
        except Exception as e:
            print(f"Error calculating rolling std: {e}")
            return None

    @lru_cache()
    def get_data_by_city(self, city: str):
        return self.table.filter(pl.col("city") == city) if self.parallel\
            else self.table[self.table["city"] == city]

    @lru_cache()
    def get_temperature_by_city(self, city: str):
        return self.get_data_by_city(city)["temperature"]


    @lru_cache()
    def detect_anomalies(self, city: str) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        if city is None or city not in self.get_cities():
            raise ValueError("City value is incorrect")

        try:
            city_data = self.get_data_by_city(city)

            return city_data["is_anomaly"]

        except Exception as e:
            raise ValueError(f"Error during detecting anomalies: {str(e)}")

    @lru_cache()
    def get_trend(self, city: str) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            city_data = self.get_data_by_city(city)
            trend = self.get_rolling_mean(city, days=30)

            if self.parallel:
                city_data = city_data.with_columns(trend.alias("trend"))
            else:
                city_data['trend'] = trend

            return city_data
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return None

    def get_describe_statisic(self, city: str):
        return self.get_data_by_city(city)["temperature"].describe()


    def get_data(self) -> Union[pl.DataFrame, pd.DataFrame]:
        return self.data

    def get_cities(self) -> Union[pl.Series, pd.Series]:
        return sorted(self.data["city"].unique())


    @lru_cache()
    def is_temperature_normal(self, city: str, timestamp: str = None, season: str = None,
                              current_temp: float = None) -> bool:
        if city is None or city not in self.get_cities():
            raise ValueError("City value is incorrect")

        if season is not None and timestamp is not None:
            raise ValueError("Season and timestamp are both not None")

        data = self.get_data_by_city(city)
        lower_bound, upper_bound = None, None
        if self.parallel:
            if season is not None:
                data = data.filter(pl.col("season") == season)
                lower_bound, upper_bound = data[0]["lower_bound"], data[0]["upper_bound"]

            if timestamp is not None:
                data = data.filter(pl.col("timestamp") == timestamp)
                lower_bound, upper_bound = data[0]["lower_bound"], data[0]["upper_bound"]
        else:
            if season is not None:
                data = data[data["season"] == season]

                lower_bound, upper_bound = data.iloc[0]["lower_bound"], data.iloc[0]["upper_bound"]

            if timestamp is not None:
                data = data[data["timestamp"] == timestamp]

                lower_bound, upper_bound = data.iloc[0]["lower_bound"], data.iloc[0]["upper_bound"]

        return lower_bound.item() <= current_temp <= upper_bound.item()

    def get_decomposition(self, city: str, period=365):
        if self.parallel:
            data = self.get_data_by_city(city).to_pandas().set_index("timestamp")
        else:
            data = self.get_data_by_city(city)
            data.set_index('timestamp', inplace=True)

        decomposition = seasonal_decompose(data['temperature'], model='additive', period=period)

        return decomposition

    def train_arima(self, city: str, order: tuple = (5, 1, 0)):
        data = self.get_data_by_city(city)
        temperatures = data['temperature'].to_numpy()

        arima_model = sm.tsa.arima.ARIMA(temperatures, order=order)
        arima_model_fit = arima_model.fit()

        return arima_model_fit

    def predict_arima(self, city: str, steps: int = 30, order: tuple = (5, 1, 0)):
        arima_model_fit = self.train_arima(city, order=order)
        forecast = arima_model_fit.forecast(steps=steps)
        forecast_index = pd.date_range(start=self.get_data_by_city(city)['timestamp'].max(),
                                       periods=steps + 1,
                                       freq='D')[1:]
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Predicted Temperature'])

        return forecast_df

    def process_data(self) -> Union[pd.DataFrame, pl.DataFrame]:
        if self.data is None:
            raise ValueError("Данные не загружены")

        try:
            if self.parallel:
                smoothed_data = self.data.with_columns(
                    pl.col("temperature").rolling_mean(window_size=30).alias("smoothed_temperature")
                )

                seasonal_stats = smoothed_data.group_by(["city", "season"]).agg([
                    pl.col("temperature").mean().alias("season_mean"),
                    pl.col("temperature").std().alias("season_std")
                ])

                processed_data = smoothed_data.join(seasonal_stats, on=["city", "season"])

                processed_data = processed_data.with_columns([
                    (pl.col("season_mean") - 2 * pl.col("season_std")).alias("lower_bound"),
                    (pl.col("season_mean") + 2 * pl.col("season_std")).alias("upper_bound"),
                ])

                processed_data = processed_data.with_columns([
                    (pl.col("temperature") < pl.col("lower_bound")).or_(
                        pl.col("temperature") > pl.col("upper_bound")
                    ).alias("is_anomaly")
                ])

                return processed_data

            else:
                self.data["smoothed_temperature"] = (
                    self.data.groupby("city")["temperature"]
                    .rolling(window=30, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

                seasonal_stats = self.data.groupby(["city", "season"])["temperature"].agg(
                    season_mean="mean",
                    season_std="std"
                ).reset_index()

                processed_data = pd.merge(
                    self.data,
                    seasonal_stats,
                    on=["city", "season"],
                    how="left"
                )

                processed_data["lower_bound"] = (
                        processed_data["season_mean"] - 2 * processed_data["season_std"]
                )
                processed_data["upper_bound"] = (
                        processed_data["season_mean"] + 2 * processed_data["season_std"]
                )
                processed_data["is_anomaly"] = (
                        (processed_data["temperature"] < processed_data["lower_bound"]) |
                        (processed_data["temperature"] > processed_data["upper_bound"])
                )

                return processed_data

        except Exception as e:
            print(f"Ошибка обработки данных: {e}")
            return None


def test(file_path: str, parallel: bool = True):
    start_time = time.time()
    data_manager = DataManager(file_path, parallel)
    data_manager.detect_anomalies("Moscow")
    print("Время выполнения (parallel={}): {}".format(parallel, time.time() - start_time))


if __name__ == "__main__":
    test('temperature_data.csv', parallel=True)
    test('temperature_data.csv', parallel=False)
