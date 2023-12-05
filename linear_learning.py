#! /usr/bin/env python3
import time
import yaml
import dataclasses
import dataclass_wizard
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import plotly.express as px

class SimpleLinearNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim*2, int(input_dim * 1.5)),
            nn.Linear(int(input_dim * 1.5), int(input_dim * 0.7)),
            nn.Linear(int(input_dim * 0.7), 1),
        )
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred

class WideLinearNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers= gen_shrinking_linear_layers(input_dim,15 , layer_number=6)
        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred

class LongLinearNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers= gen_shrinking_linear_layers(input_dim,50 , layer_number=50)
        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred

class NonLinearNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim*2, int(input_dim * 1.5)),
            nn.Tanh(),
            nn.Linear(int(input_dim * 1.5), int(input_dim * 0.7)),
            nn.Tanh(),
            nn.Linear(int(input_dim * 0.7), 1),
        )
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred  

class WideMixedNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers= gen_shrinking_linear_layers(input_dim,15 , layer_number=6)

        self.layer_stack = nn.Sequential(*intersperse(layers,nn.Tanh()))
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred

class MultiMixedNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up,last_dim = gen_even_step_layers(input_dim , 100 , layer_number=2)
        layers_down,last_dim = gen_even_step_layers(last_dim , 5 , layer_number=6)

        layers = intersperse(layers_up+layers_down,nn.Tanh())
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(last_dim), 1))

        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred

class SuperWideMixedNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers= gen_shrinking_linear_layers(input_dim,70, layer_number=10)

        self.layer_stack = nn.Sequential(*intersperse(layers,nn.Tanh()))
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred

class DiamondLinerNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up,last_dim = gen_even_step_layers(input_dim , 120 , layer_number=5)
        layers_down,last_dim = gen_even_step_layers(last_dim , 3 , layer_number=5)
        layers_down.append(nn.Linear(int(last_dim),1))
        self.layer_stack = nn.Sequential(* layers_up+layers_down)
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred


class DiamondMixesNetwork(nn.Module):
    def __init__(self,input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up,last_dim = gen_even_step_layers(input_dim , 120 , layer_number=5)
        layers_down,last_dim = gen_even_step_layers(last_dim , 3 , layer_number=5)
        layers_down.append(nn.Linear(int(last_dim),1))

        layers = intersperse(layers_up+layers_down,nn.Tanh())
        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self,x):
        ypred = self.layer_stack(x)
        return ypred


def intersperse(base_list, mix_item):
    result = [mix_item] * (len(base_list) * 2 - 1)
    result[0::2] = base_list
    return result

def gen_expanding_linear_layers(starting_dim,end_dim , layer_number):
    
    dim_step_size = (end_dim - starting_dim) /  layer_number
    
    last_out_dim = starting_dim
    next_out_dim = starting_dim + dim_step_size

    layers = []
    while next_out_dim < end_dim:
        layers.append(nn.Linear(int(last_out_dim), int(next_out_dim)))
        last_out_dim = next_out_dim
        next_out_dim = last_out_dim + dim_step_size

    return (layers ,last_out_dim)


def gen_even_step_layers(starting_dim , end_dim , layer_number):
    
    dim_step_size = (end_dim - starting_dim) /  layer_number
    
    last_out_dim = starting_dim
    next_out_dim = starting_dim + dim_step_size
    end_cond = None
    if end_dim > starting_dim:
        # This is an expanding case
        end_cond = lambda dim : dim < end_dim
    else:
        end_cond = lambda dim : dim > end_dim

    layers = []
    while end_cond(next_out_dim):
        layers.append(nn.Linear(int(last_out_dim), int(next_out_dim)))
        last_out_dim = next_out_dim
        next_out_dim = last_out_dim + dim_step_size

    return (layers ,last_out_dim)


def gen_shrinking_linear_layers(starting_dim, scale_up_factor, layer_number):
    init_output_dim = starting_dim * scale_up_factor

    init_layer = nn.Linear(int(starting_dim), int(init_output_dim))
    layers,last_dim = gen_even_step_layers(init_output_dim,1,layer_number)
    layers.append(nn.Linear(int(last_dim), 1))
    
    return [init_layer]+layers


def train_once(dataloader:data.DataLoader , model , loss_fn , optimizer):
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

    train_loss_model_map = {}
    test_loss_model_map = {}

    # Creating different models
    model_dict = {
        # "SimpleLinearNetwork": SimpleLinearNetwork(input_dim).to(device),
        # "WideLinearNetwork": WideLinearNetwork(input_dim).to(device),
        # "LongLinearNetwork": LongLinearNetwork(input_dim).to(device),
        # "NonLinearNetwork": NonLinearNetwork(input_dim).to(device),
        "WideMixedNetwork": WideMixedNetwork(input_dim).to(device),
        # "SuperWideMixedNetwork": SuperWideMixedNetwork(input_dim).to(device),
        # "DiamondLinerNetwork": DiamondLinerNetwork(input_dim).to(device),
        # "DiamondMixesNetwork": DiamondMixesNetwork(input_dim).to(device),
        "MultiMixedNetwork": MultiMixedNetwork(input_dim).to(device),
    }

    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()
    for name, model in model_dict.items():
        start_time = time.time()
        print(f"Working on model {name}")
        optimizer = optim.SGD(model.parameters() , lr=0.001,momentum=0.9)

        train_loss_list = []
        test_loss_list = []

        for epoch in range(250):

            train_avg_loss = train_once(train_loader , model , loss_fn , optimizer)
            test_loss = test_once(test_loader, model, loss_fn)
            train_loss_list.append(train_avg_loss)
            test_loss_list.append(test_loss)
        
        print(f"{name} took {time.time() - start_time} sec to learn")
        train_loss_model_map[name] = train_loss_list
        test_loss_model_map[name] = test_loss_list

    # for name,loss_data in train_loss_model_map.items():
    #     fig = px.line(loss_data,title=name)
    #     fig.show()

    fig = px.line(train_loss_model_map , title = "train")
    fig.show()
    fig = px.line(test_loss_model_map , title = "test")
    fig.show()
    # Now we save all models to file
    for name,model in model_dict.items():
        file_name = f"{name}.model.pth"
        torch.save(model, file_name)
        print(f"Saved PyTorch Model to {file_name}")

if __name__ == "__main__":
    main()
