import argparse
import time
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import DataLoader # type: ignore
from utils import load_dat, batch_split
from torch_model import torch_organization_model, torch_top_model
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

default_organization_num = 5
one_hot = False
class Vertical_FL_Train:
	''' Vertical Federated Learning Training Class'''
	def __init__(self, active_clients):
		if active_clients is None:
			self.active_clients = [True] * (default_organization_num - 1)
		else:
			self.active_clients = active_clients
		
	def run(self, args, learning_rates, batch_size):

		''' Main function for the program'''
		data_type = args.data_type                  
		model_type = args.model_type                
		epochs = args.epochs 
		
		organization_num = args.organization_num    
		attribute_split_array = \
			np.zeros(organization_num).astype(int)  
		
		# dataset preprocessing
		if data_type == 'original':

			if args.dname == 'MNIST' or args.dname == 'FMNIST':
				# hidden_layer_size = trial.suggest_int('hidden_layer_size', 1, 100)
				file_path = "./datasets/{0}.csv".format(args.dname)
				X = pd.read_csv(file_path)
				y = X['0']
				# X = X[:100]
				X = X.drop(['0'], axis=1)
				
				N, dim = X.shape
				columns = list(X.columns)
				
				attribute_split_array = \
					np.ones(len(attribute_split_array)).astype(int) * \
					int(dim/organization_num)
				if np.sum(attribute_split_array) > dim:
					print('unknown error in attribute splitting!')
				elif np.sum(attribute_split_array) < dim:
					missing_attribute_num = (dim) - np.sum(attribute_split_array)
					attribute_split_array[-1] = attribute_split_array[-1] + missing_attribute_num
				else:
					print('Successful attribute split for multiple organizations')
	
		else:
			file_path = "./dataset/{0}.dat".format(args.dname)
			X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)  
	
		# initialize the arrays to be return to the main function
		train_loss_array = []
		val_loss_array = []
		val_auc_array = []
		train_auc_array = []
		
		if model_type == 'vertical':
			# define the attributes in each group for each organization
			attribute_groups = []
			attribute_start_idx = 0
			for organization_idx in range(organization_num):
				attribute_end_idx = attribute_start_idx + attribute_split_array[organization_idx]
				attribute_groups.append(columns[attribute_start_idx : attribute_end_idx])
				attribute_start_idx = attribute_end_idx
				# print('The attributes held by Organization {0}: {1}'.format(organization_idx, attribute_groups[organization_idx]))                        
			
			#attributes per organization
			for organization_idx in range(organization_num):
				print('The number of attributes held by Organization {0}: {1}'.format(organization_idx, len(attribute_groups[organization_idx])))
				
			# get the vertically split data with one-hot encoding for multiple organizations
			
			vertical_splitted_data = {}
			encoded_vertical_splitted_data = {}

			chy_one_hot_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
			for organization_idx in range(organization_num):
				
				vertical_splitted_data[organization_idx] = \
					X[attribute_groups[organization_idx]].values#.astype('float32')
				
				
				encoded_vertical_splitted_data[organization_idx] = \
					chy_one_hot_enc.fit_transform(vertical_splitted_data[organization_idx])
		
		
			if one_hot:
				chy_one_hot_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
				for organization_idx in range(organization_num):
					
					vertical_splitted_data[organization_idx] = \
						X[attribute_groups[organization_idx]].values#.astype('float32')
					
					
					encoded_vertical_splitted_data[organization_idx] = \
						chy_one_hot_enc.fit_transform(vertical_splitted_data[organization_idx])
			else:
				for organization_idx in range(organization_num):
					
					vertical_splitted_data[organization_idx] = \
						X[attribute_groups[organization_idx]].values#.astype('float32')
					
					encoded_vertical_splitted_data = vertical_splitted_data
			
					# encoded_vertical_splitted_data = self.feature_selection(vertical_splitted_data[organization_idx], 'pca')
			
			
			print('X shape:',X.shape)
			# set up the random seed for dataset split
			random_seed = 1001
			
			# split the encoded data samples into training and test datasets
			X_train_vertical_FL = {}
			X_val_vertical_FL = {}
			X_test_vertical_FL = {}
			# selected_features = None
			
			for organization_idx in range(organization_num):
					
				# clients dont have access to the labels, only server does.
				if organization_idx == 0:
					X_train_vertical_FL[organization_idx], X_test_vertical_FL[organization_idx], y_train, y_test = \
						train_test_split(encoded_vertical_splitted_data[organization_idx], y, test_size=0.2, random_state=random_seed)
					
					X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], y_train, y_val = \
						train_test_split(X_train_vertical_FL[organization_idx], y_train, test_size=0.25, random_state=random_seed)
					
				else:
					
					X_train_vertical_FL[organization_idx], X_test_vertical_FL[organization_idx], y_train1, _ = \
						train_test_split(encoded_vertical_splitted_data[organization_idx], y, test_size=0.2, random_state=random_seed)

					X_train_vertical_FL[organization_idx], X_val_vertical_FL[organization_idx], _, _ = \
						train_test_split(X_train_vertical_FL[organization_idx], y_train1, test_size=0.25, random_state=random_seed)	
			
			train_loader_list, test_loader_list, val_loader_list = [], [], []
			for organization_idx in range(organization_num):
			
				X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
				X_val_vertical_FL[organization_idx] = torch.from_numpy(X_val_vertical_FL[organization_idx]).float()
				X_test_vertical_FL[organization_idx] = torch.from_numpy(X_test_vertical_FL[organization_idx]).float()

				train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=batch_size))
				val_loader_list.append(DataLoader(X_val_vertical_FL[organization_idx], batch_size=len(X_val_vertical_FL[organization_idx]), shuffle=False))
				test_loader_list.append(DataLoader(X_test_vertical_FL[organization_idx], batch_size=len(X_test_vertical_FL[organization_idx]), shuffle=False))
				
			y_train = torch.from_numpy(y_train.to_numpy()).long()
			y_val = torch.from_numpy(y_val.to_numpy()).long()
			y_test = torch.from_numpy(y_test.to_numpy()).long()
			train_loader_list.append(DataLoader(y_train, batch_size=batch_size))
			val_loader_list.append(DataLoader(y_val, batch_size=batch_size))
			test_loader_list.append(DataLoader(y_test, batch_size=batch_size))
			
			# NN architecture
			#num_organization_hidden_layers = params['num_organization_hidden_layers']
			num_organization_hidden_units = 128   #params['num_organization_hidden_units']
			organization_hidden_units_array = [np.array([num_organization_hidden_units])]*organization_num   #* num_organization_hidden_layers
			organization_output_dim = np.array([64 for i in range(organization_num)])
			num_top_hidden_units = 64  #params['num_top_hidden_units']
			top_hidden_units = np.array([num_top_hidden_units])
			top_output_dim = 10
			
			# build the client models
			organization_models = {}
			for organization_idx in range(organization_num):
				organization_models[organization_idx] = \
					torch_organization_model(X_train_vertical_FL[organization_idx].shape[-1],\
									organization_hidden_units_array[organization_idx],
									organization_output_dim[organization_idx])
			# build the top model over the client models
			top_model = torch_top_model(sum(organization_output_dim), top_hidden_units, top_output_dim)
			# define the neural network optimizer
			optimizer = torch.optim.Adam(top_model.parameters(), lr=learning_rates[0], weight_decay=1e-5)  #params['learning_rate_top_model'] 2.21820943080931e-05, 0.00013203242287235933
			
			optimizer_organization_list = []
			for organization_idx in range(organization_num):
				optimizer_organization_list.append(torch.optim.Adam(organization_models[organization_idx].parameters(), lr=learning_rates[1], weight_decay=1e-5))    #params['learning_rate_organization_model']0.0004807528058809301,0.000100295059051174
				
			
			print('\nStart vertical FL......\n')   
			criterion = nn.CrossEntropyLoss()
			top_model.train()
			
			for i in range(epochs):
				# if i >= 30:   # drop out after ith epoch
				# 	self.active_clients = [False, False, False, False, False, True,False, True, True,False]    # specify which clients to be dropped out
				print('Epoch: ', i)
				batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), batch_size, args.batch_type)
					
				for bidx in range(len(batch_idxs_list)):
					batch_idxs = batch_idxs_list[bidx]
					train_auc_array_temp=[]
					optimizer.zero_grad()
					for organization_idx in range(organization_num):
						optimizer_organization_list[organization_idx].zero_grad()
					organization_outputs = {}
						
					
										
					for organization_idx in range(organization_num):
							organization_outputs[organization_idx] = \
								organization_models[organization_idx](X_train_vertical_FL[organization_idx][batch_idxs])
						
					organization_outputs_cat = organization_outputs[0].float()
					for organization_idx in range(1, organization_num):
						if self.active_clients[organization_idx - 1]:
							organization_outputs_cat = torch.cat((organization_outputs_cat, organization_outputs[organization_idx]), 1).float()

						else:
							if bidx == 0:
								print("client ", organization_idx-1, " input zeroed out")
							zeroed_inputs = torch.zeros_like(organization_outputs[organization_idx])
							organization_outputs_cat = torch.cat((organization_outputs_cat, zeroed_inputs), 1).float()
											
						
					organization_outputs_cat = organization_outputs_cat.float()  # Ensure it's a Float tensor
			
					outputs = top_model(organization_outputs_cat)

					log_probs = outputs
					train_loss = criterion(log_probs, y_train[batch_idxs])
					predictions = torch.argmax(log_probs, dim=1)
					correct = (predictions == y_train[batch_idxs]).sum().item()
					total = y_train[batch_idxs].size(0)
					train_auc = correct / total
					train_auc_array_temp.append(train_auc)

					train_loss.backward()  # backpropagate the loss
					optimizer.step() # adjust parameters based on the calculated gradients 
					
					for organization_idx in range(organization_num):
						if self.active_clients[organization_idx - 1]:
							optimizer_organization_list[organization_idx].step()
						
				print('For the {0}-th epoch, train loss: {1}, train auc: {2}'.format(i+1, train_loss.detach().numpy(), np.mean(train_auc_array_temp)))
				train_auc_array.append(np.mean(train_auc_array_temp))
				train_loss=train_loss.detach().numpy()
				train_loss_array.append(train_loss.item())
					
				if (i+1)%1 == 0:
					batch_idxs_list = batch_split(len(X_val_vertical_FL[0]), batch_size, args.batch_type)
				
					for batch_idxs in batch_idxs_list:
						val_auc_array_temp = []
						val_loss_array_temp = []
						organization_outputs_for_val = {}
						
						feature_mask_tensor_list = []
						for organization_idx in range(organization_num):
							organization_outputs_for_val[organization_idx] = organization_models[organization_idx](X_val_vertical_FL[organization_idx][batch_idxs])
							feature_mask_tensor_list.append(torch.full(organization_outputs_for_val[organization_idx].shape, organization_idx))
						organization_outputs_for_val_cat = organization_outputs_for_val[0].float()
					
						#DP
						if len(organization_outputs_for_val) >= 2:
							for organization_idx in range(1, organization_num):
								organization_outputs_for_val_cat = torch.cat((organization_outputs_for_val_cat, organization_outputs_for_val[organization_idx]), 1).float()
			
						organization_outputs_for_val_cat = organization_outputs_for_val_cat.float()

						log_probs = top_model(organization_outputs_for_val_cat)
						val_loss = criterion(log_probs, y_val[batch_idxs].type(torch.LongTensor))
						predictions = torch.argmax(log_probs, dim=1)
						correct = (predictions == y_val[batch_idxs]).sum().item()
						total = y_val[batch_idxs].size(0)
						val_auc = correct / total
						val_auc_array_temp.append(val_auc)
						val_loss_array_temp.append(val_loss.detach().numpy())
				
					print('For the {0}-th epoch, val loss: {1}, val auc: {2}'.format(i+1, np.mean(val_loss_array_temp), np.mean(val_auc_array_temp)))
					val_auc_array.append(np.mean(val_auc_array_temp))
					val_loss_array.append(np.mean(val_loss_array_temp))

			# testing
			organization_outputs_for_test = {}
			for organization_idx in range(organization_num):
				organization_outputs_for_test[organization_idx] = organization_models[organization_idx](X_test_vertical_FL[organization_idx])
			organization_outputs_for_test_cat = organization_outputs_for_test[0].float()
			for organization_idx in range(1, organization_num):
						organization_outputs_for_test_cat = torch.cat((organization_outputs_for_test_cat, organization_outputs_for_test[organization_idx]), 1).float()
			
					
			outputs = top_model(organization_outputs_for_test_cat)
			log_probs = outputs
			predictions = torch.argmax(log_probs, dim=1)
			
			correct = (predictions == y_test).sum().item()  
			total = y_val.size(0) 
			test_acc = correct / total 
			print(f'test_auc = {test_acc}')
			return train_loss_array, val_loss_array, train_auc_array, val_auc_array,test_acc

			
