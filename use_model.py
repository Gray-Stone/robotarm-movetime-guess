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

from linear_learning import LongLinearNetwork , SimpleLinearNetwork , WideLinearNetwork ,NonLinearNetwork


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

    # model:SimpleLinearNetwork = torch.load('SimpleLinearNetwork.model.pth')
    model:NonLinearNetwork = torch.load('NonLinearNetwork.model.pth')

    model.eval()

    size_limit = 20
    with torch.no_grad():
        size =0 
        for x,y in zip(x_tensor,y_tensor):
            pred_y = model(x)
            print(f"predicted [{pred_y.item()}] for actual [{y.item()}]")
            size +=1
            if size > size_limit:
                break
            



if __name__ == "__main__":
    main()