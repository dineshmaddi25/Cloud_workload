import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Load the data
file_path = 'your_data_file.xlsx'  # Replace with your actual file path
data = pd.read_excel(file_path)

# Clean and process the data
data_cleaned = data.dropna()
data_cleaned['Timestamp'] = pd.to_datetime(data_cleaned['Timestamp [ms]'], unit='ms')
data_cleaned['\tCPU usage [%]'] = pd.to_numeric(data_cleaned['\tCPU usage [%]'], errors='coerce')
data_cleaned['\tMemory usage [KB]'] = pd.to_numeric(data_cleaned['\tMemory usage [KB]'], errors='coerce')

# Visualizing the data
plt.figure(figsize=(10, 6))
plt.plot(data_cleaned['Timestamp'], data_cleaned['\tCPU usage [%]'], color='b', label='CPU Usage (%)')
plt.xlabel('Timestamp')
plt.ylabel('CPU Usage (%)')
plt.title('CPU Usage Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 2: Feature Engineering - Rolling Averages for CPU and Memory Usage
data_cleaned['CPU_rolling_avg'] = data_cleaned['\tCPU usage [%]'].rolling(window=10).mean()
data_cleaned['Memory_rolling_avg'] = data_cleaned['\tMemory usage [KB]'].rolling(window=10).mean()

# Step 3: Train-Test Split for CPU usage prediction
X = data_cleaned[['\tMemory usage [KB]', '\tDisk read throughput [KB/s]', '\tDisk write throughput [KB/s]',
                  '\tNetwork received throughput [KB/s]', '\tNetwork transmitted throughput [KB/s]']]
y = data_cleaned['\tCPU usage [%]']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Linear Regression Model (as a baseline)
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Check model performance for Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print(f"Linear Regression - Mean Absolute Error: {mae_lr}")
print(f"Linear Regression - R-squared: {r2_lr}")
print(f"Linear Regression - Mean Squared Error: {mse_lr}")

# Step 5: Build Transformer Model for Future Prediction
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(TransformerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, num_encoder_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.transformer(x, x)
        x = self.fc2(x)
        return x

# Prepare the data for training the Transformer model
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for Transformer training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer for Transformer model
model_transformer = TransformerModel(input_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_transformer.parameters(), lr=0.001)

# Train the Transformer model
for epoch in range(50):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        output = model_transformer(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# Predict using the Transformer model
model_transformer.eval()
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_pred_transformer = model_transformer(X_test_tensor).detach().numpy()

# Check model performance for Transformer
mae_transformer = mean_absolute_error(y_test, y_pred_transformer)
r2_transformer = r2_score(y_test, y_pred_transformer)
mse_transformer = mean_squared_error(y_test, y_pred_transformer)

print(f"Transformer Model - Mean Absolute Error: {mae_transformer}")
print(f"Transformer Model - R-squared: {r2_transformer}")
print(f"Transformer Model - Mean Squared Error: {mse_transformer}")

# Step 6: Plot the Actual vs Predicted CPU Usage (Linear Regression vs Transformer)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Regression')
plt.scatter(y_test, y_pred_transformer, color='red', label='Transformer')
plt.xlabel('Actual CPU Usage (%)')
plt.ylabel('Predicted CPU Usage (%)')
plt.title('Actual vs Predicted CPU Usage')
plt.legend()
plt.tight_layout()
plt.show()

# Step 7: Create a report with future predictions
def generate_future_predictions(num_periods=5):
    last_data = data_cleaned.iloc[-1]
    future_predictions = []
    for i in range(num_periods):
        # Using the last values to predict the next ones
        future_data = last_data[['\tMemory usage [KB]', '\tDisk read throughput [KB/s]',
                                 '\tDisk write throughput [KB/s]', '\tNetwork received throughput [KB/s]',
                                 '\tNetwork transmitted throughput [KB/s]']].values.reshape(1, -1)
        
        future_data_tensor = torch.tensor(future_data, dtype=torch.float32)
        predicted_cpu_usage = model_transformer(future_data_tensor).detach().numpy()[0][0]
        
        future_timestamp = last_data['Timestamp'] + pd.Timedelta(minutes=5 * (i + 1))  # Assuming 5-minute intervals
        future_predictions.append([future_timestamp, predicted_cpu_usage])
    
    future_df = pd.DataFrame(future_predictions, columns=['Timestamp', 'Predicted CPU Usage [%]'])
    return future_df

# Example: Predict the next 5 periods
num_periods = 5
future_predictions = generate_future_predictions(num_periods)
print(future_predictions)

# Plot future predictions along with the historical data
plt.figure(figsize=(10, 6))
plt.plot(data_cleaned['Timestamp'], data_cleaned['\tCPU usage [%]'], color='blue', label='Historical CPU Usage')
plt.plot(future_predictions['Timestamp'], future_predictions['Predicted CPU Usage [%]'], color='red', linestyle='--', label='Predicted CPU Usage')
plt.xlabel('Timestamp')
plt.ylabel('CPU Usage (%)')
plt.title('Historical and Predicted CPU Usage')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the predictions to a report (Excel)
output_report_path = 'future_predictions_report.xlsx'
future_predictions.to_excel(output_report_path, index=False)
print(f"Future Predictions saved to {output_report_path}")
