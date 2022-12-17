import numpy as np  
import torch

import opnet
from volume_inversion_data import load_data

def train_test(model: opnet.OperatorNet, loss_f:torch.nn.MSELoss, lr, name, epochs):
    train_loader = torch.utils.data.DataLoader(load_data("../Data_from_Aili/volume_inversion_train_data.npz"), batch_size = 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses=[]
    
    for epoch in range(epochs): 
        #print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(train_loader):
            # Compute prediction error
            pred = model(X)
            loss = loss_f(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch%50==0:

            test_loader = torch.utils.data.DataLoader(load_data("../Data_from_Aili/volume_inversion_test_data.npz"), batch_size=64)
            num_batches = len(test_loader)
            test_loss = 0
            with torch.no_grad():
                for X, y in test_loader:
                    pred = model(X)
                    test_loss += loss_f(pred, y).item()
            test_loss /= num_batches
            losses.append(test_loss)
            #print(f"Avg loss: {test_loss:>8f}")
    return losses