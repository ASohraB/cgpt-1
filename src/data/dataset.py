import os
import pickle
import torch

import os
import pickle
import torch
import csv
import json
from utils.helpers import cInp


class Dataset:
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        #if self.transform:
        #    sample = self.transform(sample)
        # Convert to tensors
        #sample = torch.tensor(sample, dtype=torch.cdouble)  # Use torch.cfloat for complexNN
        #label = torch.tensor(label, dtype=torch.cdouble)    # Now label is also complex       
        #if not isinstance(sample, torch.Tensor):
        #    sample = torch.tensor(sample, dtype=torch.cdouble)
        #if not isinstance(label, torch.Tensor):
        #    label = torch.tensor(label, dtype=torch.cdouble)
        #or, if you know they're always tensors:
        #sample = sample.clone().detach()
        #label = label.clone().detach()
        return sample, label


    @staticmethod
    def load_data(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if os.path.exists(file_path):
            if ext == ".pkl":
                with open(file_path, "rb") as f:
                    loaded = pickle.load(f)
                data = loaded.get("data", [])
                labels = loaded.get("labels", [])
                return data, labels
            elif ext == ".csv":
                data, labels = [], []
                with open(file_path, "r", newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        # Assumes last column is label, rest is data
                        *sample, label = row
                        data.append([float(x) for x in sample])
                        labels.append(int(label))
                return data, labels
            elif ext == ".json":
                with open(file_path, "r") as f:
                    loaded = json.load(f)
                data = loaded.get("data", [])
                labels = loaded.get("labels", [])
                return data, labels
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            input_size = 2   # Match main.py
            output_size = 2  # Match main.py ?
            num_samples = 32
            data = torch.stack([x[0] for x in cInp(num_samples)])
            #as autoencoder
            labels=data # if needed, here temporary
            #autoregressive , comment for autoencoder
            labels = torch.stack([x[1] for x in cInp(num_samples)])
            sample_dict = {
                "data": data,
                "labels": labels
            }

           
            return sample_dict["data"], sample_dict["labels"]
        
    @staticmethod
    def preprocess_data(raw_data):
        # Implement data preprocessing logic here
        pass    
