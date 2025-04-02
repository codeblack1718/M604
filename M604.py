import numpy as np

import pandas as pd

df = pd.read_csv('/content/german_temperature_data_1996_2021_from_selected_weather_stations.csv')

df.head()

column_names = ["date"] + [f"station_{col}" for col in df.columns[1:]]
df.columns = column_names

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df.fillna(method="ffill", inplace=True)
df.head()

df.shape

print("\n📌 Column Names:")
print(df.columns.tolist())

df.info()

print(df.isnull().sum())

df.fillna(method='ffill', inplace=True)

df["station_298"].fillna(df["station_298"].mean(), inplace=True)

df.isnull().sum()

df.duplicated().sum()

df.nunique()

df.sample(20)

df.describe()

!pip install pandas numpy matplotlib seaborn plotly folium geopandas statsmodels

for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].abs()
(df.head())

df.sample(20)

pip install pandas plotly gradio

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

def load_data():
    print("Loading data...")
    df = pd.read_csv("german_temperature_data_1996_2021_from_selected_weather_stations.csv")
    df.rename(columns={"MESS_DATUM": "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])

    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    return df
def get_station_columns(df):

    station_columns = [col for col in df.columns if col.startswith('station_')]


    if not station_columns:
        non_station_cols = ['date', 'year', 'month', 'MESS_DATUM']
        station_columns = [col for col in df.columns if col not in non_station_cols
                          and pd.api.types.is_numeric_dtype(df[col])]

    if not station_columns:
        station_columns = [
            "station_Berlin_Tempelhof",
            "station_Hamburg_Fuhlsbuettel",
            "station_Munich_Airport",
            "station_Cologne_Bonn_Airport",
            "station_Frankfurt_Airport",
            "station_Stuttgart_Airport",
            "station_Dresden",
            "station_Hannover",
            "station_Nuremberg",
            "station_Leipzig"
        ]
        print("No station columns detected. Using sample station names.")

    return station_columns

def create_time_series(df, station, start_date, end_date):
    """Create a time series plot for the selected station and date range"""
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    if filtered_df.empty:
        return px.line(title="No data available for the selected date range")

    fig = px.line(filtered_df, x='date', y=station,
                  title=f"Temperature Trend for {station}",
                  labels={"date": "Date", "value": "Temperature (°C)"})

    fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°C)")
    return fig

def create_heatmap(df, station, start_year, end_year):
    """Create a year-month heatmap for the selected station"""
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    if filtered_df.empty:
        return px.imshow(title="No data available for the selected year range")

    pivot_data = filtered_df.pivot_table(index='year', columns='month', values=station, aggfunc='mean')

    fig = px.imshow(pivot_data,
                   labels=dict(x="Month", y="Year", color="Temperature (°C)"),
                   x=[f"Month {i}" for i in range(1, 13)],
                   y=pivot_data.index,
                   title=f"Monthly Temperature Heatmap for {station}",
                   color_continuous_scale="RdBu_r")

    fig.update_layout(coloraxis_colorbar=dict(title="Temp (°C)"))
    return fig

def create_seasonal_box_plot(df, station, start_year, end_year):
    """Create a seasonal box plot for the selected station"""
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    if filtered_df.empty:
        return px.box(title="No data available for the selected year range")

    season_map = {
        1: "Winter", 2: "Winter", 3: "Spring",
        4: "Spring", 5: "Spring", 6: "Summer",
        7: "Summer", 8: "Summer", 9: "Fall",
        10: "Fall", 11: "Fall", 12: "Winter"
    }

    filtered_df['season'] = filtered_df['month'].map(season_map)

    fig = px.box(filtered_df, x='season', y=station,
                 title=f"Seasonal Temperature Distribution for {station}",
                 category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]},
                 color='season',
                 color_discrete_map={
                     "Winter": "blue", "Spring": "green",
                     "Summer": "red", "Fall": "orange"
                 })

    fig.update_layout(xaxis_title="Season", yaxis_title="Temperature (°C)")
    return fig

def create_yearly_trend(df, station, start_year, end_year):
    """Create a yearly trend analysis for the selected station"""
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    if filtered_df.empty:
        return px.line(title="No data available for the selected year range")

    yearly_avg = filtered_df.groupby('year')[station].mean().reset_index()

    fig = px.scatter(yearly_avg, x='year', y=station,
                     title=f"Yearly Temperature Trend for {station}")

    fig.add_trace(go.Scatter(
        x=yearly_avg['year'],
        y=yearly_avg[station].rolling(window=3).mean(),
        mode='lines',
        name='3-Year Moving Average',
        line=dict(color='red')
    ))

    fig.update_layout(xaxis_title="Year", yaxis_title="Average Temperature (°C)")
    return fig

def create_monthly_comparison(df, stations, year):
    """Create a comparison of multiple stations for a specific year by month"""
    filtered_df = df[df['year'] == year]

    if filtered_df.empty:
        return px.line(title=f"No data available for year {year}")

    monthly_data = filtered_df.groupby('month')[stations].mean().reset_index()

    fig = go.Figure()

    for station in stations:
        fig.add_trace(go.Scatter(
            x=monthly_data['month'],
            y=monthly_data[station],
            mode='lines+markers',
            name=station
        ))

    fig.update_layout(
        title=f"Monthly Temperature Comparison for Year {year}",
        xaxis=dict(
            title="Month",
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        yaxis=dict(title="Temperature (°C)"),
        legend_title="Stations"
    )

    return fig

def detect_anomalies(df, station, threshold=3):
    """Detect temperature anomalies using Z-score method"""
    z_scores = (df[station] - df[station].mean()) / df[station].std()

    anomalies = df[abs(z_scores) > threshold].copy()

    if anomalies.empty:
        return px.scatter(title=f"No anomalies found for threshold {threshold}")

    fig = px.scatter(df, x='date', y=station, opacity=0.5,
                    title=f"Temperature Anomalies for {station} (Z-score > {threshold})")

    fig.add_trace(go.Scatter(
        x=anomalies['date'],
        y=anomalies[station],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Anomalies'
    ))

    fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°C)")
    return fig

def create_correlation_heatmap(df, stations):
    """Create a correlation heatmap between selected stations"""
    corr_matrix = df[stations].corr()

    fig = px.imshow(corr_matrix,
                   title="Station Temperature Correlation Heatmap",
                   color_continuous_scale="RdBu_r",
                   labels=dict(color="Correlation"))

    fig.update_layout(
        xaxis_title="Stations",
        yaxis_title="Stations"
    )

    return fig

def create_histogram(df, station, bins=30):
    """Create a histogram of temperature distribution for the selected station"""
    fig = px.histogram(df, x=station, nbins=bins,
                      title=f"Temperature Distribution for {station}",
                      labels={station: "Temperature (°C)"},
                      opacity=0.7,
                      color_discrete_sequence=["skyblue"])

    mean_temp = df[station].mean()
    fig.add_vline(x=mean_temp, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_temp:.2f}°C",
                 annotation_position="top right")

    fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Frequency")
    return fig

def display_visualization(viz_type, station, start_date, end_date, year,
                         compare_stations, anomaly_threshold, correlation_stations,
                         histogram_bins):
    """Main function to create and display the selected visualization"""
    try:
        df = load_data()

        if not isinstance(df, pd.DataFrame):
            return "Error: Data not available or invalid format"

        if viz_type == "Time Series":
            return create_time_series(df, station, start_date, end_date)

        elif viz_type == "Monthly Heatmap":
            start_year = pd.to_datetime(start_date).year
            end_year = pd.to_datetime(end_date).year
            return create_heatmap(df, station, start_year, end_year)

        elif viz_type == "Seasonal Box Plot":
            start_year = pd.to_datetime(start_date).year
            end_year = pd.to_datetime(end_date).year
            return create_seasonal_box_plot(df, station, start_year, end_year)

        elif viz_type == "Yearly Trend":
            start_year = pd.to_datetime(start_date).year
            end_year = pd.to_datetime(end_date).year
            return create_yearly_trend(df, station, start_year, end_year)

        elif viz_type == "Station Comparison":
            stations_list = [s.strip() for s in compare_stations.split(",")]
            return create_monthly_comparison(df, stations_list, year)

        elif viz_type == "Anomaly Detection":
            return detect_anomalies(df, station, anomaly_threshold)

        elif viz_type == "Correlation Heatmap":
            stations_list = [s.strip() for s in correlation_stations.split(",")]
            return create_correlation_heatmap(df, stations_list)

        elif viz_type == "Histogram":
            return create_histogram(df, station, histogram_bins)

        else:
            return "Please select a visualization type"

    except Exception as e:
        return f"Error: {str(e)}"

def create_interface():
    try:
        df = load_data()
        station_columns = get_station_columns(df)

        viz_types = [
            "Time Series", "Monthly Heatmap", "Seasonal Box Plot", "Yearly Trend",
            "Station Comparison", "Anomaly Detection", "Correlation Heatmap", "Histogram"
        ]

        with gr.Blocks(title="German Temperature Data Explorer") as app:
            gr.Markdown("# 🌡️ German Temperature Data Explorer")
            gr.Markdown("Explore temperature data from German weather stations (1996-2021)")

            with gr.Row():
                with gr.Column(scale=1):
                    viz_type = gr.Dropdown(viz_types, label="Visualization Type", value="Time Series")

                    station = gr.Dropdown(station_columns, label="Select Station",
                                         value=station_columns[0] if station_columns else None)

                    start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="1996-01-01")
                    end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2021-12-31")

                    with gr.Accordion("Additional Options", open=False):
                        year = gr.Number(label="Year (for Station Comparison)", value=2020, minimum=1996, maximum=2021, step=1)

                        default_compare = ", ".join([station_columns[i] for i in range(min(3, len(station_columns)))])
                        compare_stations = gr.Textbox(label="Stations to Compare (comma-separated)",
                                                    value=default_compare)

                        anomaly_threshold = gr.Slider(label="Anomaly Z-score Threshold", minimum=1.0, maximum=5.0, value=3.0, step=0.1)

                        default_correlation = ", ".join([station_columns[i] for i in range(min(5, len(station_columns)))])
                        correlation_stations = gr.Textbox(label="Stations for Correlation (comma-separated)",
                                                        value=default_correlation)

                        histogram_bins = gr.Slider(label="Histogram Bins", minimum=10, maximum=100, value=30, step=5)

                    generate_btn = gr.Button("Generate Visualization")

                with gr.Column(scale=2):
                    output = gr.Plot(label="Visualization Output")

            with gr.Accordion("Help & Information", open=False):
                gr.Markdown("""
                ## How to Use This Dashboard

                1. **Select a Visualization Type** from the dropdown menu
                2. **Choose a Weather Station** to analyze
                3. **Set the Date Range** for your analysis
                4. Use **Additional Options** for specific visualization settings
                5. Click **Generate Visualization** to create the plot

                ## Visualization Types

                - **Time Series**: Shows temperature changes over time
                - **Monthly Heatmap**: Displays temperatures by month and year
                - **Seasonal Box Plot**: Shows temperature distribution by season
                - **Yearly Trend**: Analyzes yearly temperature trends with moving average
                - **Station Comparison**: Compares multiple stations by month for a specific year
                - **Anomaly Detection**: Identifies unusual temperature readings
                - **Correlation Heatmap**: Shows correlation between station temperatures
                - **Histogram**: Displays the distribution of temperature values
                """)

            with gr.Accordion("Available Stations", open=False):
                station_list_markdown = ", ".join(station_columns)
                gr.Markdown(f"### Available Stations\n{station_list_markdown}")

            generate_btn.click(
                fn=display_visualization,
                inputs=[viz_type, station, start_date, end_date, year,
                       compare_stations, anomaly_threshold, correlation_stations,
                       histogram_bins],
                outputs=output
            )

            viz_type.change(
                fn=lambda x: gr.update(visible=(x == "Station Comparison")),
                inputs=[viz_type],
                outputs=[year]
            )

        return app

    except Exception as e:
        print(f"Error creating interface: {str(e)}")
        with gr.Blocks(title="German Temperature Data Explorer") as app:
            gr.Markdown("# 🌡️ German Temperature Data Explorer")
            gr.Markdown(f"**Error loading data: {str(e)}**")
            gr.Markdown("Please check your data file and ensure it contains the required columns.")
        return app


if __name__ == "__main__":
    create_interface().launch()

