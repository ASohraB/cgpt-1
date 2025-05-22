import torch
#import torch.nn as nn
from data.dataset import Dataset
from train.train import train_model
from torch.utils.data import DataLoader
from model.gpt import GPT
from utils.helpers import oncInp


def main():
    # Hyperparameters
    batch_size = 32
    input_size = 2
    embed_size = 128   # New: embedding size for attention
    output_size = 2
    nheads = 8        # Number of attention heads
    mlp_hidden_size = 512  # Hidden size for MLP
    num_layers = 12
    num_epochs = 200
    learning_rate = 0.0001


    #as1=cInp(4)
    #print("as: ", as1)

    # Load data and labels
    data, labels = Dataset.load_data("data/sample.pkl")
    # Initialize dataset and dataloader
    dataset = Dataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = GPT(input_size, embed_size, output_size, nheads, mlp_hidden_size, num_layers)

    # Train the model
    train_model(model, train_loader, num_epochs, learning_rate)
    # --- Test on a single input after training ---

    model.eval()
    with torch.no_grad():

        # Use the first sample from the data as a test input
        if isinstance(data[0], torch.Tensor):
            test_input = data[0].clone().detach().unsqueeze(0)
        else:
            test_input = torch.tensor(data[0], dtype=torch.cdouble).unsqueeze(0)
        #test_input = torch.tensor([[0.1416+0.2867j, 0.1308-0.9384j]], dtype=torch.cdouble)

        #uncoment for noisy input
        #test_input = oncInp(data[0].clone().detach().unsqueeze(0))
        
        output = model(test_input)
        print("--------------------------")
        print("Test input:", test_input)
        print("Model output:", output)
        print("Target:", labels[0])
    

if __name__ == '__main__':
    main()