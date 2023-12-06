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


# class SimpleLinearNetwork(nn.Module):

#     def __init__(self, input_dim, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.layer_stack = nn.Sequential(
#             nn.Linear(input_dim, input_dim * 2),
#             nn.Linear(input_dim * 2, int(input_dim * 1.5)),
#             nn.Linear(int(input_dim * 1.5), int(input_dim * 0.7)),
#             nn.Linear(int(input_dim * 0.7), 1),
#         )
#         print(self.layer_stack)

#     def forward(self, x):
#         ypred = self.layer_stack(x)
#         return ypred


# class WideLinearNetwork(nn.Module):

#     def __init__(self, input_dim, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         layers = gen_shrinking_linear_layers(input_dim, 15, layer_number=6)
#         self.layer_stack = nn.Sequential(*layers)
#         print(self.layer_stack)

#     def forward(self, x):
#         ypred = self.layer_stack(x)
#         return ypred


# class LongLinearNetwork(nn.Module):

#     def __init__(self, input_dim, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         layers = gen_shrinking_linear_layers(input_dim, 50, layer_number=50)
#         self.layer_stack = nn.Sequential(*layers)
#         print(self.layer_stack)

#     def forward(self, x):
#         ypred = self.layer_stack(x)
#         return ypred


class NonLinearNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, int(input_dim * 1.5)),
            nn.Tanh(),
            nn.Linear(int(input_dim * 1.5), int(input_dim * 0.7)),
            nn.Tanh(),
            nn.Linear(int(input_dim * 0.7), 1),
        )
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred


class WideMixedNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers = gen_shrinking_linear_layers(input_dim, 15, layer_number=6)

        self.layer_stack = nn.Sequential(*intersperse(layers, nn.Tanh()))
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred


class TanhReluNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up, last_dim = gen_even_step_layers(input_dim, 100, layer_number=2)
        layers_down, last_dim = gen_even_step_layers(last_dim, 5, layer_number=10)

        layers = intersperse(layers_up + layers_down, nn.Tanh())
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(last_dim), 1))
        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred

class MixedDisNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up, last_dim = gen_even_step_layers(input_dim, 120, layer_number=2)
        layers_down, last_dim = gen_even_step_layers(last_dim, 5, layer_number=10)

        layers = intersperse(layers_up + layers_down, nn.Tanh())
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(last_dim), 1))
        
        layers[5] = nn.LeakyReLU()
        
        self.layer_stack = nn.Sequential(*layers)
        
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred


class SimplerTanhNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up, last_dim = gen_even_step_layers(input_dim, 100, layer_number=2)
        layers_down, last_dim = gen_even_step_layers(last_dim, 1, layer_number=6)

        layers = intersperse(layers_up + layers_down, nn.Tanh())
        layers.append(nn.Tanh())
        layers.append(nn.Linear(int(last_dim), 1))

        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred


class TanhShrinkNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up, last_dim = gen_even_step_layers(input_dim, 100, layer_number=2)
        layers_down, last_dim = gen_even_step_layers(last_dim, 1, layer_number=6)

        layers = intersperse(layers_up + layers_down, nn.Tanh())
        layers.append(nn.Tanh())
        layers.append(nn.Linear(int(last_dim), 1))

        layers[3] = nn.Tanhshrink()

        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred


class LongMixedNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up, last_dim = gen_even_step_layers(input_dim, 300, layer_number=5)
        layers_down, last_dim = gen_even_step_layers(last_dim, 3, layer_number=40)

        layers = intersperse(layers_up + layers_down, nn.Tanh())
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(last_dim), 1))
        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred


class DiamondLinerNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up, last_dim = gen_even_step_layers(input_dim, 120, layer_number=5)
        layers_down, last_dim = gen_even_step_layers(last_dim, 3, layer_number=5)
        layers_down.append(nn.Linear(int(last_dim), 1))
        self.layer_stack = nn.Sequential(*layers_up + layers_down)
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred


class DiamondMixesNetwork(nn.Module):

    def __init__(self, input_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        layers_up, last_dim = gen_even_step_layers(input_dim, 120, layer_number=5)
        layers_down, last_dim = gen_even_step_layers(last_dim, 3, layer_number=5)
        layers_down.append(nn.Linear(int(last_dim), 1))

        layers = intersperse(layers_up + layers_down, nn.Tanh())
        self.layer_stack = nn.Sequential(*layers)
        print(self.layer_stack)

    def forward(self, x):
        ypred = self.layer_stack(x)
        return ypred


def intersperse(base_list, mix_item):
    result = [mix_item] * (len(base_list) * 2 - 1)
    result[0::2] = base_list
    return result


def gen_expanding_linear_layers(starting_dim, end_dim, layer_number):

    dim_step_size = (end_dim - starting_dim) / layer_number

    last_out_dim = starting_dim
    next_out_dim = starting_dim + dim_step_size

    layers = []
    while next_out_dim < end_dim:
        layers.append(nn.Linear(int(last_out_dim), int(next_out_dim)))
        last_out_dim = next_out_dim
        next_out_dim = last_out_dim + dim_step_size

    return (layers, last_out_dim)


def gen_even_step_layers(starting_dim, end_dim, layer_number):

    dim_step_size = (end_dim - starting_dim) / layer_number

    last_out_dim = starting_dim
    next_out_dim = starting_dim + dim_step_size
    end_cond = None
    if end_dim > starting_dim:
        # This is an expanding case
        end_cond = lambda dim: dim < end_dim
    else:
        end_cond = lambda dim: dim > end_dim

    layers = []
    while end_cond(next_out_dim):
        layers.append(nn.Linear(int(last_out_dim), int(next_out_dim)))
        last_out_dim = next_out_dim
        next_out_dim = last_out_dim + dim_step_size

    return (layers, last_out_dim)


def gen_shrinking_linear_layers(starting_dim, scale_up_factor, layer_number):
    init_output_dim = starting_dim * scale_up_factor

    init_layer = nn.Linear(int(starting_dim), int(init_output_dim))
    layers, last_dim = gen_even_step_layers(init_output_dim, 1, layer_number)
    layers.append(nn.Linear(int(last_dim), 1))

    return [init_layer] + layers


def train_once(dataloader: data.DataLoader, model, loss_fn, optimizer):
    num_batches = len(dataloader)

    model.train()
    total_loss = 0
    for batch, (x, y) in enumerate(dataloader):
        # x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches
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


class ModelAndData():

    def __init__(self, model, train_data, test_data) -> None:
        self.model = model
        self.train_data = train_data
        self.test_data = test_data


def main():
    data_file_name = "condensed_datas.json"
    df = pd.read_json(data_file_name, dtype=True)

    # This is the input
    input_df = df.filter(like="target_j")

    input_expanded_df = input_df.copy()

    start_joint_df = input_df.filter(like="start")
    end_joint_df = input_df.filter(like="end")

    i = 0
    for (start_name, start_value), (end_name, end_value) in zip(start_joint_df.items(),
                                                                end_joint_df.items()):
        # print(f"{start_name} - {end_name} , i {i}")
        input_expanded_df[f"diff_{i}"] = start_value - end_value
        i += 1

    output_df = df.filter(like="motion_duration")
    
    input_dim = len(input_df.columns)
    input_extended_dim = len(input_expanded_df.columns)
    print(f"{input_dim} Input columns: {input_df.columns}")
    print(f"{input_extended_dim} Input columns: {input_expanded_df.columns}")
    print(f"Output columns: {output_df.columns}")

    # Forming datas into the pytorch format
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    print(torch.cuda.get_device_name(device=device))

    basic_x_tensor = torch.tensor(input_df.values,
                                  requires_grad=True,
                                  dtype=torch.float,
                                  device=device)
    extend_x_tensor = torch.tensor(input_expanded_df.values,
                                   requires_grad=True,
                                   dtype=torch.float,
                                   device=device)
    y_tensor = torch.tensor(output_df.values, requires_grad=True, dtype=torch.float, device=device)
    print(f"basic xshape:{basic_x_tensor.shape}")
    print(f"extend xshape:{extend_x_tensor.shape}")
    print(f"yshape:{y_tensor.shape}")

    test_data_percent = 0.2
    basic_train_dataset, basic_test_dataset = data.random_split(
        data.TensorDataset(basic_x_tensor, y_tensor), [1 - test_data_percent, test_data_percent])
    extend_train_dataset, extend_test_dataset = data.random_split(
        data.TensorDataset(extend_x_tensor, y_tensor), [1 - test_data_percent, test_data_percent])

    basic_train_loader = data.DataLoader(basic_train_dataset, batch_size=64)
    basic_test_loader = data.DataLoader(basic_test_dataset, batch_size=64)
    extend_train_loader = data.DataLoader(extend_train_dataset, batch_size=64)
    extend_test_loader = data.DataLoader(extend_test_dataset, batch_size=64)
    train_loss_model_map = {}
    test_loss_model_map = {}

    # Creating different models
    model_dict = {
        "LongMixedNetwork_basic": 
                    ModelAndData(
                LongMixedNetwork(basic_x_tensor.shape[1]).to(device), basic_train_loader,
                basic_test_loader),
        "TanhReluNetwork_basic": 
                    ModelAndData(
                TanhReluNetwork(basic_x_tensor.shape[1]).to(device), basic_train_loader,
                basic_test_loader),
        "DiamondMixesNetwork-basic": 
                    ModelAndData(
                DiamondMixesNetwork(basic_x_tensor.shape[1]).to(device), basic_train_loader,
                basic_test_loader),
        "MixedDisNetwork-basic": 
                    ModelAndData(
                MixedDisNetwork(basic_x_tensor.shape[1]).to(device), basic_train_loader,
                basic_test_loader),
        "TanhShrinkNetwork-basic": 
                    ModelAndData(
                TanhShrinkNetwork(basic_x_tensor.shape[1]).to(device), basic_train_loader,
                basic_test_loader),
    }

    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()
    for name, model_and_data in model_dict.items():
        start_time = time.time()
        print(f"Working on model {name}")

        current_model, train_loader, test_loader = (model_and_data.model, model_and_data.train_data,
                                                    model_and_data.test_data)
        optimizer = optim.SGD(current_model.parameters(), lr=0.001, momentum=0.9)

        train_loss_list = []
        test_loss_list = []

        for epoch in range(300):

            train_avg_loss = train_once(train_loader, current_model, loss_fn, optimizer)
            test_loss = test_once(test_loader, current_model, loss_fn)
            train_loss_list.append(train_avg_loss)
            test_loss_list.append(test_loss)

        print(f"{name} took {time.time() - start_time} sec to learn")
        train_loss_model_map[name] = train_loss_list
        test_loss_model_map[name] = test_loss_list

    fig = px.line(train_loss_model_map, title="Training data cost over iteration")
    fig.show()
    fig = px.line(test_loss_model_map, title="Testing data cost over iteration")
    fig.show()
    # Now we save all models to file
    for name, model_and_data in model_dict.items():
        file_name = f"{name}.model.pth"
        torch.save(model_and_data.model, file_name)
        print(f"Saved PyTorch Model to {file_name}")


if __name__ == "__main__":
    main()
