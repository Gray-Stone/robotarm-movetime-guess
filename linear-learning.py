#! /usr/bin/env python3
import yaml
import dataclasses
import dataclass_wizard
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import plotly.express as px

class LinearOnlyNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.Linear(input_dim * 4, input_dim*2),
            nn.Linear(input_dim*2, int(input_dim * 1.5)),
            nn.Linear(int(input_dim * 1.5), int(input_dim * 0.8)),
            nn.Linear(int(input_dim * 0.8), int(input_dim / 2)),
            nn.Linear(int(input_dim / 2), int(input_dim / 3)),
            nn.Linear(int(input_dim / 3), 1),
        )

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred

class WideLinearNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers = []
        init_output_dim = input_dim * 10
        num_layer = 10
        
        # layers.append(nn.Linear(input_dim, init_output_dim))
        last_output_dim = input_dim

        dim_step_size = init_output_dim / num_layer

        next_output_dim = init_output_dim
        while next_output_dim > 5:

            layers.append(nn.Linear(int(last_output_dim), int(next_output_dim)))
            print(f"{last_output_dim}->{next_output_dim}")
        
            last_output_dim = next_output_dim
            next_output_dim = last_output_dim - dim_step_size


        layers.append(nn.Linear(int(last_output_dim), 1))
        self.layer_stack = nn.Sequential(*layers)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred




def train_once(dataloader:data.DataLoader , model , loss_fn , optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    total_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        # x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss+=loss.item()

        # Back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(x)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss /num_batches
    return avg_loss

def test_once(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss


def main():
    data_file_name = "condensed_datas.json"
    df = pd.read_json(data_file_name , dtype=True)

    # This is the input
    input_df = df.filter(like="target_j")
    output_df = df.filter(like="motion_duration")
    input_dim = len(input_df.columns)
    print(f"{input_dim} Input columns: {input_df.columns}")
    print(f"Output columns: {output_df.columns}")

    # Forming datas into the pytorch format
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    print (torch.cuda.get_device_name(device=device))

    x_tensor = torch.tensor(input_df.values , requires_grad=True,dtype=torch.float,device=device)
    y_tensor = torch.tensor(output_df.values , requires_grad=True,dtype=torch.float,device=device)
    print(f"xshape:{x_tensor.shape}")
    print(f"yshape:{y_tensor.shape}")

    test_data_percent = 0.2
    train_dataset, test_dataset = data.random_split(data.TensorDataset(x_tensor, y_tensor),
                                                    [1 - test_data_percent, test_data_percent])

    train_loader = data.DataLoader(train_dataset,batch_size=64)
    test_loader = data.DataLoader(test_dataset,batch_size=64)




    model_dict = {
        "LinearOnlyNetwork": LinearOnlyNetwork(input_dim).to(device),
        "WideLinearNetwork": WideLinearNetwork(input_dim).to(device),
    }


    loss_data_dict = {}

    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()
    for name, model in model_dict.items():
        print(f"Working on model {name}")
        optimizer = optim.SGD(model.parameters() , lr=0.001,momentum=0.9)
        loss_data = []
        for epoch in range(50):

            train_avg_loss = train_once(train_loader , model , loss_fn , optimizer)
            test_loss = test_once(test_loader, model, loss_fn)
            loss_data.append((train_avg_loss, test_loss))
            # totalLoss = 0
            # for i in range(len(x_tensor)):
            #     # Single Forward Pass
            #     ypred = model(x_tensor[i])

            #     # Measure how well the model predicted vs the actual value
            #     loss = loss_fn(ypred, y_tensor[i])

            #     # Track how well the model predicted (called loss)
            #     totalLoss+=loss.item()

            #     # Update the neural network
            #     loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()

            # # Print out our loss after each training iteration
            # print ("Total Loss: ", totalLoss)
        loss_data_dict[name] = loss_data

    for name,loss_data in loss_data_dict.items():
        fig = px.line(loss_data,title=name)
        fig.show()
main()
