import time
import uuid

import polars as pl
import pandas as pd
from typing import Union, Optional, List
import os
import plotly

from methodtools import lru_cache

COLUMNS = ['temperature', 'season', 'city', 'timestamp']
FILES_DIR = 'files'


class DataManager:

    def __init__(self, file_path: str = None, parallel: bool = True) -> None:
        if file_path is None:
            file_path = 'temperature_data.csv'
        self.file_path = file_path
        self.parallel = parallel
        self.files_dir = os.path.join(os.path.dirname(__file__), 'files_folder')

        self.check_files_dir()
        self.data = self.check_data(self.load_data(file_path))

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
        print("file_path",file_path)
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


    def get_rolling_mean(self) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            data = self.data['temperature']
            if self.parallel:
                return data.rolling_mean(30)
            return data.rolling(30).mean()
        except Exception as e:
            print(f"Error calculating rolling mean: {e}")
            return None

    def get_rolling_std(self) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            data = self.data['temperature']

            if self.parallel:
                return data.rolling_std(30)

            return data.rolling(30).std()
        except Exception as e:
            print(f"Error calculating rolling mean: {e}")
            return None

    def detect_anomalies(self) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            self.rolling_mean = self.get_rolling_mean()
            self.rolling_std = self.get_rolling_std()

            lower_bound = self.rolling_mean - 2 * self.rolling_std
            upper_bound = self.rolling_mean + 2 * self.rolling_std

            self.anomalies = (self.data['temperature'] < lower_bound) | (self.data['temperature'] > upper_bound)

            return self.anomalies
        except Exception as e:
            print(f"Error calculating rolling mean: {e}")
            return None

    def get_trend(self) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            data = self.data['temperature']

            if self.parallel:
                return data.rolling_mean(365)

            return data.rolling(365).mean()
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return None

    @lru_cache()
    def get_historical_range(self, season: str) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            if self.parallel:
                season_data = self.data.filter(self.data['season'] == season)
            else:
                season_data = self.data[self.data['season'] == season]

            q1 = season_data['temperature'].quantile(0.25)
            q3 = season_data['temperature'].quantile(0.75)

            return pl.Series([q1, q3]) if self.parallel else pd.Series([q1, q3])
        except Exception as e:
            print(f"Error calculating historical range: {e}")
            return None

    def get_data(self) -> Union[pl.DataFrame, pd.DataFrame]:
        return self.data

    def get_cities(self) -> Union[pl.Series, pd.Series]:
        return sorted(self.data["city"].unique())

    @lru_cache()
    def get_season_by_city(self, city: str):
        if self.parallel:
            filtered_data = self.data.filter(self.data["city"] == city)
            return filtered_data.select("season").to_series()[-1]
        else:
            return self.data[self.data['city'] == city]['season'].iloc[-1]

    @lru_cache()
    def is_temperature_anomalous(self, city: str, current_temp: float) -> str:
        season = self.get_season_by_city(city)
        season_range = self.get_historical_range(season)

        if season_range[0] <= current_temp <= season_range[1]:
            return "normal"
        return "anomalous"

    def plot_anomalies(self):
        if self.data is None:
            raise ValueError("Data is None")
        try:
            import plotly.express as px

            # Создаем DataFrame с информацией об аномалиях
            data = self.data.to_pandas() if self.parallel else self.data
            data['is_anomaly'] = self.anomalies

            # Построение интерактивного графика с аномалиями
            fig = px.scatter(
                data,
                x="timestamp",
                y="temperature",
                color="is_anomaly",
                color_discrete_map={True: "red", False: "blue"},
                labels={"is_anomaly": "Anomaly"},
                title="Temperature Anomalies",
            )
            fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                legend_title="Anomalies",
                template="plotly_white",
            )
            return fig
        except Exception as e:
            print(f"Error during plotting anomalies: {e}")
            return None

    def add_data_column(self, name, values):
        if self.parallel:
            self.data = self.data.with_columns(pl.Series(name=name, values=values))
        else:
            self.data[name] = values


def test(file_path: str, parallel: bool = True):
    print(file_path)
    data_manager = DataManager(file_path, parallel)
    start_time = time.time()
    anomalies = data_manager.detect_anomalies()
    trend = data_manager.get_trend()
    season_range = data_manager.get_historical_range('summer')
    print("Время выполнения (parallel={}): {}".format(parallel, time.time() - start_time))


if __name__ == "__main__":
    data_manager = DataManager()
    data_manager.load_data()

    trend = data_manager.get_trend()
    if trend is not None:
        stdata_manager.add_data_column("trend", trend)
        trend_data = data_manager.get_data()
        x = 1
    #test('temperature_data.csv', parallel=True)
    #test('temperature_data.csv', parallel=False)
