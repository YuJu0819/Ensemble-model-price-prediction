import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleCNN, SimpleLSTM, TCN
# Model parameters
input_dim = 5  # Number of features
hidden_dim = 50  # Number of features in hidden state
num_layers = 1  # Number of stacked lstm layers
output_dim = 2  # Binary classification

# Dummy data preparation
# 100 samples, 10 time steps per sample, 5 features per time step
features = torch.randn(100, 10, 5)
targets = torch.randint(0, 2, (100,))  # Binary targets for each sample

# Create Dataset
dataset = TensorDataset(features, targets)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

model = SimpleCNN(input_dim, hidden_dim, num_layers, output_dim)
model_lstm = SimpleLSTM()
model_TCN = TCN()
loss_function = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, optimizer, loss_function, epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(data)  # Forward pass
            loss = loss_function(outputs, target)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            total_loss += loss.item()  # Sum up batch loss

            if batch_idx % 10 == 0:
                print(
                    f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}')

        print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader)}')


# Execute training
train_model(model, train_loader, optimizer, loss_function, epochs=10)
