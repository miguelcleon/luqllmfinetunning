import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np

# Define the endpoint and parameters - fixed PostgREST format
url = "http://data.prepestuaries.org:3001/timeseriesresultvalues"
params = {
    "resultid": "in.(79026,79017)",  # Using "in.()" operator for multiple values
    "select": "valuedatetime,datavalue,resultid"
}

# Fetch the data
response = requests.get(url, params=params)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)

# Print info for debugging
print(f"DataFrame shape: {df.shape}")
print("DataFrame head:")
print(df.head())

# Convert valuedatetime to datetime
df['valuedatetime'] = pd.to_datetime(df['valuedatetime'])

# Convert datavalue to float and handle any conversion errors
df['datavalue'] = pd.to_numeric(df['datavalue'], errors='coerce')

# Print data info after conversion
print("Data types after conversion:")
print(df.dtypes)
print("NaN values count:", df['datavalue'].isna().sum())

# Remove rows with NaN in datavalue
df = df.dropna(subset=['datavalue'])
print(f"DataFrame shape after removing NaNs: {df.shape}")

# Save to CSV for further processing
df.to_csv('timeseries_data.csv', index=False)
print("Saved raw data to timeseries_data.csv")

# Map resultids to meaningful names
resultid_map = {
    79017: 'Electrical_conductivity',
    79026: 'Chlorophyll_a'
}

# Convert resultid to integer to ensure proper mapping
df['resultid'] = df['resultid'].astype(int)

# Create a new column with mapped names
df['parameter'] = df['resultid'].map(resultid_map)
print("Parameter mapping complete. Unique parameters:", df['parameter'].unique())

# Pivot using the parameter names
df_pivot = df.pivot(index='valuedatetime', columns='parameter', values='datavalue')
print("Pivot table shape:", df_pivot.shape)
print("Pivot table columns:", df_pivot.columns)
print("Pivot table data types:")
print(df_pivot.dtypes)
print("NaN values in pivot table:")
print(df_pivot.isna().sum())

# Resample to handle missing values with explicit float conversion
try:
    # Convert to float first if needed
    for col in df_pivot.columns:
        df_pivot[col] = df_pivot[col].astype(float)

    # Then resample
    df_pivot = df_pivot.resample('D').mean()
    print("Resampling successful")

    # Now interpolate
    df_pivot = df_pivot.interpolate(method='linear')
    print("Interpolation successful")
except Exception as e:
    print(f"Error during resampling/interpolation: {e}")
    # Try to resample each column individually as a fallback
    try:
        result = pd.DataFrame(index=pd.date_range(start=df_pivot.index.min(), end=df_pivot.index.max(), freq='D'))
        for col in df_pivot.columns:
            series = df_pivot[col].astype(float)
            resampled = series.resample('D').mean()
            interpolated = resampled.interpolate(method='linear')
            result[col] = interpolated
        df_pivot = result
        print("Fallback resampling/interpolation successful")
    except Exception as e2:
        print(f"Fallback approach also failed: {e2}")
        # Just continue with what we have

# Reset index for Prophet
df_reset = df_pivot.reset_index()
df_reset.rename(columns={'valuedatetime': 'ds'}, inplace=True)
print("Reset index complete. DataFrame ready for Prophet")


# Function to detect anomalies
def detect_anomalies(df, column):
    # Check if column exists in the dataframe
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        return None

    # Check if column has data
    if df[column].isna().all():
        print(f"No data for {column}")
        return None

    df_prophet = df[['ds', column]].rename(columns={column: 'y'}).dropna()

    # Check if there's enough data
    if len(df_prophet) < 2:
        print(f"Not enough data for {column} after dropping NAs")
        return None

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=0)  # No forecasting, just analyzing existing data
    forecast = model.predict(future)

    df_prophet = df_prophet.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    df_prophet['anomaly'] = (df_prophet['y'] < df_prophet['yhat_lower']) | (df_prophet['y'] > df_prophet['yhat_upper'])

    return df_prophet


try:
    # Detect anomalies for each column that exists
    anomalies_results = {}

    for column in ['Chlorophyll_a', 'Electrical_conductivity']:
        if column in df_reset.columns:
            print(f"Detecting anomalies for {column}...")
            anomalies = detect_anomalies(df_reset, column)
            if anomalies is not None:
                anomalies_results[column] = anomalies

                # Save results to CSV
                anomalies.to_csv(f'{column.lower()}_anomalies.csv', index=False)
                print(f"Saved {column} anomalies to {column.lower()}_anomalies.csv")

                # Create and save plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_reset['ds'], df_reset[column], label=column)
                ax.scatter(anomalies[anomalies['anomaly']]['ds'],
                           anomalies[anomalies['anomaly']]['y'],
                           color='red', label='Anomalies')
                ax.set_title(f'{column} Anomalies')
                ax.legend()
                plt.savefig(f'{column.lower()}_anomalies.png')
                print(f"Saved {column} plot to {column.lower()}_anomalies.png")
        else:
            print(f"Column '{column}' not found, skipping anomaly detection")

    print("Analysis complete.")

except Exception as e:
    print(f"Error during anomaly detection or plotting: {e}")