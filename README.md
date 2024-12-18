# Vertical Federated Learning with Client Dropout

This repository is a modified version of the original implementation of Vertical Federated Learning (VFL) by Kang Wei et al. We intend to use this code for our course project that investigates drop out clients in VFL.

## 📄 **Credit**

The original implementation was authored by:

- Kang Wei  
- Jun Li  
- Chuan Ma  
- Ming Ding  
- Sha Wei  
- Fan Wu  
- Guihai Chen  
- Thilina Ranbaduge  

The corresponding paper can be found here: [arXiv:2202.04309](https://arxiv.org/abs/2202.04309)  
The original GitHub repository is available here: [Vertical_FL](https://github.com/AdamWei-boop/Vertical_FL)

## 🛠️ **Changes**

This repository has been modified to suit our course project, which investigates the impact of client dropout in Vertical Federated Learning.  
Key changes include:

1. **Removed Features**:
   - Differential Privacy (DP) implementation.
   - Contribution calculation logic.

2. **Added Features**:
   - Logic for handling **active clients**.
   - A toggleable variable named `active_clients` to specify which clients are currently participating in the training process.

## 🚀 **Getting Started**

### Prerequisites

Ensure you have `pipenv` installed to manage the Python environment.
```
pipenv install -r requirements.txt
python run_vfl_basic_model.py
```
The above command runs the VFL model for four clients without client dropout. In order to change the number of clients, change the default_organization_num variable in run_vfl_basic_model.py. For random client drop out patterns, change the dropout_probability variable, and for changing the number of clients, default_organization_num variable in the run_vfl_base_model_with_random_client_dropouts.py file and run the below command.
```
python run_vfl_base_model_with_random_client_dropouts.py
```
In order to change which client to dropout during start of the training toggle the active_clients variable in run_vfl_basic_model.py
To drop out clients in the middle of the training uncomment lines 204 ans 205 in torch_vertical_FL_train_base_model.py file.

You dont have to run the code using the args listed in the code. You can run the code directly using the above command. 

