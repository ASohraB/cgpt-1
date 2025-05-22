#import torch
import torch.optim as optim
#import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.gpt import GPT
from data.dataset import Dataset


def complex_mse_loss(input, target):
    return ((input - target).abs() ** 2).mean()


# CE loss for complexNN
def train_model(model, data_loader, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #criterion = cLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # Convert complex outputs to real (choose one):
            #outputs_real = outputs.abs()  # or outputs.real
            # cLoss expects shape [batch, num_classes] and targets [batch]
            #loss = criterion(outputs_real.view(-1, outputs_real.size(-1)), targets.view(-1))
            
            loss = complex_mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    # Initialize dataset and model
    dataset = Dataset()
    model = GPT()

    # Start training
    train_model(model, dataset)