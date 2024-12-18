from torch_vertical_FL_train_base_model import Vertical_FL_Train
import optuna
import argparse
import time
import sys
import random
import numpy as np
import torch

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Hyperparameters
hyper_parameters = {
    'learning_rate_top_model': 0.001, 
    'learning_rate_organization_model': 0.0005, 
    'batch_size': 512
}
dataset = 'MNIST'
default_organization_num = '10'
num_default_epochs = 60
num_iterations = 1
dropout_probability = 0.7 # Probability of dropping a client

one_hot = False
fs = False

# Toggle this to make clients inactive
active_clients = [True]*int(default_organization_num)

batch_size = hyper_parameters['batch_size']
learning_rates = [hyper_parameters['learning_rate_top_model'], hyper_parameters['learning_rate_organization_model']]
default_organization_num = int(default_organization_num) + 1
r

def run_base_model(args):
	
	vfl_model = Vertical_FL_Train(active_clients=active_clients)
	test_acc_sum = 0
	# Initialize arrays to accumulate results
	train_loss_array_sum = np.zeros(args.epochs)
	val_loss_array_sum = np.zeros(args.epochs)
	train_auc_array_sum = np.zeros(args.epochs)
	val_auc_array_sum = np.zeros(args.epochs)

	test_acc_array = []
	
	for i in range(num_iterations):
		print("Iteration: ", i)
		train_loss_array, val_loss_array, train_auc_array, val_auc_array, test_acc = vfl_model.run(args, learning_rates, batch_size)
		train_loss_array_sum += np.array(train_loss_array)
		val_loss_array_sum += np.array(val_loss_array)
		train_auc_array_sum += np.array(train_auc_array)
		val_auc_array_sum += np.array(val_auc_array)
		test_acc_array.append(test_acc)
		test_acc_sum += test_acc

	# Calculate averages
	train_loss_avg = (train_loss_array_sum / num_iterations).tolist()
	val_loss_avg = (val_loss_array_sum / num_iterations).tolist()
	train_auc_avg = (train_auc_array_sum / num_iterations).tolist()
	val_auc_avg = (val_auc_array_sum / num_iterations).tolist()
	test_acc_avg = test_acc_sum / num_iterations

	
	print("train_auc_array = ", train_auc_array)
	print("train_loss_array = ", train_loss_array)
	print("val_auc_array = ", val_auc_array)
	print("val_loss_array = ", val_loss_array)
	print("test_acc = ", test_acc_avg)
	

	return {
		'train_loss_avg': train_loss_avg,
		'val_loss_avg': val_loss_avg,
		'train_auc_avg': train_auc_avg,
		'val_auc_avg': val_auc_avg,
		'test_acc_avg': test_acc_avg,
		'test_acc_array': test_acc_array
	}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vertical FL')
    parser.add_argument('--dname', default=dataset, help='dataset name: AVAZU, ADULT')
    parser.add_argument('--epochs', type=int, default=num_default_epochs, help='number of training epochs')  
    parser.add_argument('--batch_type', type=str, default='mini-batch')
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--data_type', default='original', help='define the data options: original or one-hot encoded')
    parser.add_argument('--model_type', default='vertical', help='define the learning methods: vertical or centralized')    
    parser.add_argument('--organization_num', type=int, default=default_organization_num, help='number of organizations, if we use vertical FL')
    parser.add_argument('--contribution_schem',  type=str, default='ig', help='define the contribution evaluation method')
    parser.add_argument('--attack', default='original', help='define the data attack or not')
    parser.add_argument('--deactivate_client', type=int, default=None, help='Index of client to deactivate (0-based)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save output files')
    args = parser.parse_args()

    run_base_model(args)
