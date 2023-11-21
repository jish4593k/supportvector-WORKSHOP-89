import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Importing the dataset
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling. Required for SVR
y = y.reshape(-1, 1)  # Reshape y to a 2D array
sc_X = torch.std_mean(torch.tensor(X, dtype=torch.float32), dim=0)
sc_y = torch.std_mean(torch.tensor(y, dtype=torch.float32), dim=0)

X = (torch.tensor(X, dtype=torch.float32) - sc_X[1]) / sc_X[0]
y = (torch.tensor(y, dtype=torch.float32) - sc_y[1]) / sc_y[0]

# Training SVR model
class SVRModel(nn.Module):
    def __init__(self):
        super(SVRModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SVRModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X)

    # Compute the loss
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Convert back to numpy arrays for visualization
X_np = sc_X[0] * X.numpy() + sc_X[1]
y_np = sc_y[0] * y.numpy() + sc_y[1]
y_pred_np = sc_y[0] * model(X).detach().numpy() + sc_y[1]

# Visualizing the SVR Results using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_np.flatten(), y=y_np.flatten(), color='red', label='Actual Data')
sns.lineplot(x=X_np.flatten(), y=y_pred_np.flatten(), color='blue', label='SVR Prediction')
plt.title('Support Vector Regression with Seaborn')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualizing the SVR Results using Plotly
fig = px.scatter(x=X_np.flatten(), y=y_np.flatten(), title='Support Vector Regression with Plotly',
                 labels={'x': 'Position Level', 'y': 'Salary'}, trendline='ols', trendline_color_override='blue')
fig.show()
