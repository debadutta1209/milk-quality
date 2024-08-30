from flask import Flask, render_template, jsonify
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ThingSpeak credentials and channel details
channel_id = 2640422
read_api_key = 'THQEFJ8K1JUZM4TQ'
base_url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.json'


def fetch_data(num_points=100):
    url = f'{base_url}?api_key={read_api_key}&results={num_points}'
    response = requests.get(url)
    data = response.json()
    feeds = data['feeds']
    df = pd.DataFrame(feeds)
    return df


@app.route('/')
def index():
    # Load data and perform analysis
    df = fetch_data()

    # Ensure all required fields are present and convert to numeric
    fields = ['field1', 'field2', 'field3', 'field4', 'field5', 'field6']
    df = df[fields].apply(pd.to_numeric, errors='coerce')
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df.fillna(0, inplace=True)

    # Normalize data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=fields)

    # Define features and labels
    X = df_scaled
    y = (df['field1'] < 5) & (df['field2'] < 500)

    # Train a decision tree classifier
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # Predict safety for all data points
    df['predicted_safe'] = model.predict(df_scaled)

    # Calculate statistics
    safe_count = int(df['predicted_safe'].sum())
    unsafe_count = len(df) - safe_count
    avg_temp = df['field1'].mean()
    avg_gas = df['field2'].mean()

    # Generate plot
    img = io.BytesIO()
    plt.figure(figsize=(14, 10))
    # Generate plot
img = io.BytesIO()
plt.figure(figsize=(14, 10))

# (Your existing plotting code here...)

plt.tight_layout()
plt.savefig(img, format='png')
img.seek(0)
plot_url = base64.b64encode(img.getvalue()).decode()
plot_url = 'data:image/png;base64,' + plot_url  # Add this line to ensure the base64 string is correctly formatted.


    plt.subplot(3, 1, 1)
    plt.plot(df['field1'], label='Temperature (°C)', color='blue', marker='o')
    plt.axhline(y=5, color='red', linestyle='--', label='Threshold Temperature (5°C)')
    plt.title('Temperature over Time')
    plt.xlabel('Sample')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df['field2'], label='Gas Concentration (MQ-4)', color='green', marker='o')
    plt.axhline(y=500, color='red', linestyle='--', label='Threshold Gas Concentration (500)')
    plt.title('Gas Concentration over Time')
    plt.xlabel('Sample')
    plt.ylabel('Gas Concentration (MQ-4)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['predicted_safe'], label='Predicted Safety', color='orange', marker='o', linestyle='-')
    plt.axhline(y=1, color='blue', linestyle='--', label='Safe')
    plt.axhline(y=0, color='red', linestyle='--', label='Unsafe')
    plt.title('Milk Safety Analysis')
    plt.xlabel('Sample')
    plt.ylabel('Safety Status')
    plt.yticks([0, 1], ['Unsafe', 'Safe'])
    plt.legend()

    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html',
                           safe_count=safe_count,
                           unsafe_count=unsafe_count,
                           avg_temp=avg_temp,
                           avg_gas=avg_gas,
                           plot_url=plot_url)


if __name__ == '__main__':
    app.run(debug=True)
