import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Define ThingSpeak credentials and channel details
channel_id = 2640422
read_api_key = 'THQEFJ8K1JUZM4TQ'
write_api_key = 'NJ597V14SXZVY3VK'
base_url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.json'
update_url = f'https://api.thingspeak.com/update.json'

# Fetch data from ThingSpeak
def fetch_data(num_points=100):
    url = f'{base_url}?api_key={read_api_key}&results={num_points}'
    response = requests.get(url)
    data = response.json()
    feeds = data['feeds']
    df = pd.DataFrame(feeds)
    return df

# Send data to ThingSpeak
def send_data_to_thingspeak(field_values):
    payload = {
        'api_key': write_api_key,
        'field1': field_values[0],
        'field2': field_values[1],
        'field3': field_values[2],
        'field4': field_values[3],
        'field5': field_values[4],
        'field6': field_values[5],
        'field7': field_values[6],  # Assuming field7 is used for some result or status
    }
    response = requests.post(update_url, data=payload)
    if response.status_code == 200:
        print("Data successfully sent to ThingSpeak")
    else:
        print(f"Failed to send data. Status code: {response.status_code}")

# Load data
df = fetch_data()

# Ensure all required fields are present and convert to numeric
fields = ['field1', 'field2', 'field3', 'field4', 'field5', 'field6']
df = df[fields].apply(pd.to_numeric, errors='coerce')

# Clean data: Replace infinities and NaNs with a specific value or remove rows with them
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.fillna(0, inplace=True)

# Normalize data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=fields)

# Define features and labels
X = df_scaled
# Simple rule for labeling: Safe if temperature < 5°C and gasValue < 500
y = (df['field1'] < 5) & (df['field2'] < 500)

# Train a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict safety for all data points
df['predicted_safe'] = model.predict(df_scaled)

# Calculate and prepare data to send
safe_count = df['predicted_safe'].sum()
unsafe_count = len(df) - safe_count

# Example values to send (update according to your needs)
field_values = [
    df['field1'].mean(),  # Example: Average temperature
    df['field2'].mean(),  # Example: Average gas concentration
    safe_count,           # Count of safe instances
    unsafe_count,         # Count of unsafe instances
    None,                 # Placeholder for unused fields
    None,                 # Placeholder for unused fields
    None                  # Placeholder for unused fields
]

# Send data to ThingSpeak
send_data_to_thingspeak(field_values)

# Visualization
plt.figure(figsize=(14, 10))

# Plot Temperature
plt.subplot(3, 1, 1)
plt.plot(df['field1'], label='Temperature (°C)', color='blue', marker='o')
plt.axhline(y=5, color='red', linestyle='--', label='Threshold Temperature (5°C)')
plt.title('Temperature over Time')
plt.xlabel('Sample')
plt.ylabel('Temperature (°C)')
plt.legend()

# Plot Gas Concentration
plt.subplot(3, 1, 2)
plt.plot(df['field2'], label='Gas Concentration (MQ-4)', color='green', marker='o')
plt.axhline(y=500, color='red', linestyle='--', label='Threshold Gas Concentration (500)')
plt.title('Gas Concentration over Time')
plt.xlabel('Sample')
plt.ylabel('Gas Concentration (MQ-4)')
plt.legend()

# Plot Milk Safety
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
plt.show()
