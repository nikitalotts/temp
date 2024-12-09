import asyncio

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.data_manager import DataManager
from src.open_weather_client import OpenWeatherClient
import plotly.express as px
st.set_page_config(layout="wide")


OPEN_WEATHER_API = "d15dafe9f8bce85be20df85899ff8d12"


async def main():
    uploaded_file = await upload_file_block()

    if uploaded_file is not None:
        data_manager = await upload_file(uploaded_file)

        city = await city_block(data_manager)
        valid_key, open_weather_client = await api_key_block()

        if not valid_key:
            st.error("API KEY IS NOT VALID")
        else:
            if st.button("Analyze and Monitor Temperature"):
                await render_anomalies(data_manager)
                await render_trend(data_manager)
                await render_current_weather(city, data_manager, open_weather_client)


async def render_anomalies(data_manager: DataManager):
    anomalies = data_manager.detect_anomalies()
    if anomalies is not None:
        st.subheader("Temperature Anomalies")
        st.plotly_chart(data_manager.plot_anomalies())


async def render_trend(data_manager: DataManager):
    trend = data_manager.get_trend()
    if trend is not None:
        st.subheader("Temperature Trend")
        data_manager.add_data_column("trend", trend)
        trend_data = data_manager.get_data()
        fig = px.line(
            trend_data,
            x="timestamp",
            y=["trend"],
            labels={"value": "Temperature (°C)", "variable": "Type"},
            title="Temperature Trend",
            hover_data=["timestamp"]
        )
        st.plotly_chart(fig, use_container_width=True)


async def upload_file(uploaded_file: UploadedFile) -> DataManager:
    data_manager = DataManager()
    data_manager.upload_streamlit_file(uploaded_file.getvalue())
    return data_manager


async def city_block(data_manager: DataManager):
    st.subheader("Select a City")
    cities = data_manager.get_cities()
    city = st.selectbox("City", cities)
    return city


async def api_key_block():
    st.subheader("Enter OpenWeatherMap API Key")
    api_key = st.text_input("API Key", placeholder="Enter key", value=OPEN_WEATHER_API)
    open_weather_client = OpenWeatherClient(api_key)
    valid_key = open_weather_client.check_key()

    return valid_key, open_weather_client


async def upload_file_block():
    st.title("Weather Analysis App")
    st.subheader("Upload Historical Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    return uploaded_file


async def render_current_weather(
        city: str,
        data_manager: DataManager,
        open_weather_client: OpenWeatherClient):
    current_weather = await open_weather_client.get_weather_async(city)
    if "error" not in current_weather:
        current_temp = current_weather["main"]["temp"]
        st.write(f"Current temperature in {city}: {current_temp:.2f}°C")
        print(current_temp)
        anomaly_status = data_manager.is_temperature_anomalous(city, current_temp)
        st.write(f"The current temperature is {anomaly_status} for the season.")
    else:
        st.error(f"Failed to fetch current weather: {current_weather['error']}")


if __name__ == "__main__":
    asyncio.run(main())