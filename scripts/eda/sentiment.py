import matplotlib.pyplot as plt
import pandas as pd

# Example Data: Sentiment Data with timestamps
data = {'timestamp': ['2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04', '2025-02-05'],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']}
df = pd.DataFrame(data)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Convert sentiment to numerical values for plotting
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
df['sentiment_value'] = df['sentiment'].map(sentiment_map)

# Plotting the Sentiment Over Time
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['sentiment_value'], marker='o', color='b', linestyle='-', label='Sentiment')
plt.xlabel('Date')
plt.ylabel('Sentiment Value')
plt.title('Sentiment Over Time')
plt.xticks(rotation=45)
plt.grid(True)

# Saving the plot to the specified folder
plt.savefig('../../outputs/plots/sentiment_over_time.png')
plt.close()
