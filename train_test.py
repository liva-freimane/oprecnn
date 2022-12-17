import numpy as np  
import torch

import opnet
from simple_inversion_data import load_data


#save_path = "./simple_inversion.pth"
def train_test(model: opnet.OperatorNet, loss_f:torch.nn.MSELoss, lr, name, epochs):
    train_loader = torch.utils.data.DataLoader(load_data("../Data_from_Aili/simple_inversion_train_data.npz"), batch_size = 64)
    #train_loader = torch.utils.data.DataLoader(load_data("../Data_from_Aili/simple_inversion_train_dataNEGA.npz"), batch_size = 64)
    #train_loader = torch.utils.data.DataLoader(load_data("../Data_from_Aili/simple_inversion_train_dataPOSNEG.npz"), batch_size = 64)
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
        
        if epoch%10==1:

            test_loader = torch.utils.data.DataLoader(load_data("../Data_from_Aili/simple_inversion_test_data.npz"), batch_size=64)
            #test_loader = torch.utils.data.DataLoader(load_data("../Data_from_Aili/simple_inversion_test_dataNEGA.npz"), batch_size=64)
            #test_loader = torch.utils.data.DataLoader(load_data("../Data_from_Aili/simple_inversion_test_dataPOSNEG.npz"), batch_size=64)
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

