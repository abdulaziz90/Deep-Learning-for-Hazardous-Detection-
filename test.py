import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
# from torchsummary import summary
from torchinfo import summary
import re
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import functional as F
import subprocess
import sys
import glob


'''
python test.py -c configs/unet.yaml --temporal_resolution 20 --save_name 'unet_T=20'
python test.py -c configs/unet.yaml --temporal_resolution 10 --save_name 'unet_T=10'

python test.py -c configs/ddl.yaml --temporal_resolution 20 --save_name 'ddl_T=20'
python test.py -c configs/ddl.yaml --temporal_resolution 10 --save_name 'ddl_T=10'

python test.py -c configs/vae.yaml --temporal_resolution 20 --save_name 'vae_T=20'
python test.py -c configs/vae.yaml --temporal_resolution 10 --save_name 'vae_T=10'

'''

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/unet.yaml')

# Add new arguments
# model_params
parser.add_argument('--latent_dim2', type=int, help='Size of secondary latent space')
parser.add_argument('--cloud_weight', type=float, help='Cloud weight')
parser.add_argument('--kl_weight1', type=float, help='KL weight - primary latent space')
parser.add_argument('--param_weight', type=float, help='Params of primary latent space MSE')
parser.add_argument('--kl_weight2', type=float, help='KL weight - secondary latent space')
parser.add_argument('--arch_option', type=str, help='Architecture option: spatial_attention or NONE')
parser.add_argument('--lstm_option', type=str, help='LSTM option: temporal_attention or NONE')

# data_params
parser.add_argument('--data_path', type=str, help='Where the data is stored')
parser.add_argument('--temporal_resolution', type=int, help='temporal_resolution (10 or 20)')
parser.add_argument('--train_batch_size', type=int, help='train_batch_size')
parser.add_argument('--val_batch_size', type=int, help='val_batch_size')

# exp_params
parser.add_argument('--LR', type=float, help='learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay')
parser.add_argument('--scheduler_gamma', type=float, help='scheduler_gamma')
parser.add_argument('--manual_seed', type=int, help='manual_seed')

# trainer_params
parser.add_argument('--gpus', type=int, nargs='+', help='GPUs to use')
parser.add_argument('--max_epochs', type=int, help='max_epochs')

# logging_params
parser.add_argument('--save_dir', type=str, help='save_dir')
parser.add_argument('--save_name', type=str, help='save_name')

# test_params
parser.add_argument('--test_params', type=bool, help='test_params') # For vae and ddl
parser.add_argument('--test_cloud', type=bool, help='test_cloud') # For unet and vae
parser.add_argument('--cloud_id', type=int, help='cloud_id') # For unet and vae
parser.add_argument('--generate_cloud', type=bool, help='generate_cloud') # For vae
parser.add_argument('--save_clouds', type=bool, help='save_clouds') # For unet

args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Override model_params
if args.cloud_weight is not None:
    config['model_params']['cloud_weight'] = args.cloud_weight
if args.kl_weight1 is not None:
    config['model_params']['kl_weight1'] = args.kl_weight1
if args.param_weight is not None:
    config['model_params']['param_weight'] = args.param_weight
if args.kl_weight2 is not None:
    config['model_params']['kl_weight2'] = args.kl_weight2
if args.arch_option is not None:
    config['model_params']['arch_option'] = args.arch_option
if args.lstm_option is not None:
    config['model_params']['lstm_option'] = args.lstm_option

# Override data_params
if args.data_path is not None:
    config['data_params']['data_path'] = args.data_path
if args.temporal_resolution is not None:  # Note: typo in your code for "resolution"
    config['data_params']['temporal_resolution'] = args.temporal_resolution
if args.train_batch_size is not None:
    config['data_params']['train_batch_size'] = args.train_batch_size
if args.val_batch_size is not None:
    config['data_params']['val_batch_size'] = args.val_batch_size

# Override exp_params
if args.LR is not None:
    config['exp_params']['LR'] = args.LR
if args.weight_decay is not None:
    config['exp_params']['weight_decay'] = args.weight_decay
if args.scheduler_gamma is not None:
    config['exp_params']['scheduler_gamma'] = args.scheduler_gamma
if args.manual_seed is not None:
    config['exp_params']['manual_seed'] = args.manual_seed

# Override trainer_params
if args.gpus is not None:
    config['trainer_params']['gpus'] = args.gpus
if args.max_epochs is not None:
    config['trainer_params']['max_epochs'] = args.max_epochs

# Override logging_params
if args.save_dir is not None:
    config['logging_params']['save_dir'] = args.save_dir
if args.save_name is not None:
    config['logging_params']['save_name'] = args.save_name

# Override test_params
if args.test_params is not None:
    config['test_params']['test_params'] = args.test_params
if args.test_cloud is not None:
    config['test_params']['test_cloud'] = args.test_cloud
if args.cloud_id is not None:
    config['test_params']['cloud_id'] = args.cloud_id
if args.generate_cloud is not None:
    config['test_params']['generate_cloud'] = args.generate_cloud
if args.save_clouds is not None:
    config['test_params']['save_clouds'] = args.save_clouds

# Initialize TensorBoardLogger with an empty string for 'version'
tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['save_name'],
    version=""  # Set version to an empty string
)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
if config['model_params']['name'] == 'Unet':
    data.setup_stage1()
else: 
    data.setup_stage2()

model = vae_models[config['model_params']['name']](**config['model_params'], data_path=config["data_params"]["data_path"], temporal_resolution=config["data_params"]["temporal_resolution"])

device = next(model.parameters()).device
model = model.to(device)
# its shape is (batch_size, channels, depth, height, width)
print(device)

experiment = VAEXperiment(model, config['exp_params'])

# Create the custom model checkpoint callback with the specified save threshold
checkpoint_callback = ModelCheckpoint(
    dirpath=tb_logger.log_dir,
    filename='checkpoint_{epoch}',
)

runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        checkpoint_callback,
                    ],
                    strategy=DDPPlugin(find_unused_parameters=False),
                    **config['trainer_params'])

if config['model_params']['name'] == 'Unet':
    # its shape is (batch_size, channels, depth, height, width)
    summary(model, input_size=(16, 3, config["data_params"]["temporal_resolution"], 128, 128))
else: 
    # its shape is (batch_size, channels, depth, height, width)
    summary(model, input_size=(16, 1, config["data_params"]["temporal_resolution"], 128, 128))

print(f"======= Testing {config['model_params']['name']} =======")

# Assuming your checkpoints are saved in this directory
checkpoints_dir = tb_logger.log_dir
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# Get a list of all checkpoint files in the directory
checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]

# Check if there are any checkpoint files
if not checkpoint_files:
    print("No checkpoint files found. Downloading from URL...")
    
    if config['model_params']['name'] == 'VAE':
        # Define the URL and the destination path
        if config['data_params']['temporal_resolution'] == 20:
            checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_vae_T=20.ckpt")
            url = 'https://www.dropbox.com/scl/fi/i1e4ac9rxitb1lrlz5od6/checkpoint_vae_T-20.ckpt?rlkey=pip9jcrarlf17bijkkkjezcjt&dl=1'
            subprocess.run(['wget', '-O', checkpoint_path, url], check=True)
            print(f"Download complete.")
        elif config['data_params']['temporal_resolution'] == 10:
            checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_vae_T=10.ckpt")
            url = 'https://www.dropbox.com/scl/fi/8sdnsqffxmz92x9fc46lj/checkpoint_vae_T-10.ckpt?rlkey=qeu7tgladut9gxint0et4uind&dl=1'
            subprocess.run(['wget', '-O', checkpoint_path, url], check=True)
            print(f"Download complete.")
        else: 
            raise ValueError(f"Invalid temporal resolution: {config['data_params']['temporal_resolution']}. Expected 10 or 20.")
    elif config['model_params']['name'] == 'DDL':
        # Define the URL and the destination path
        if config['data_params']['temporal_resolution'] == 20:
            checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_ddl_T=20.ckpt")
            url = 'https://www.dropbox.com/scl/fi/lszld9d5avts4w04qn5cu/checkpoint_ddl_T-20.ckpt?rlkey=c29u2at0lud1kom7fehfnol9f&dl=1'
            subprocess.run(['wget', '-O', checkpoint_path, url], check=True)
            print(f"Download complete.")
        elif config['data_params']['temporal_resolution'] == 10:
            checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_ddl_T=10.ckpt")
            url = 'https://www.dropbox.com/scl/fi/6koxgsgfj3f6j297z7xpq/checkpoint_ddl_T-10.ckpt?rlkey=owc65muj78yd5bxr4mwjyv8fx&dl=1'
            subprocess.run(['wget', '-O', checkpoint_path, url], check=True)
            print(f"Download complete.")
        else: 
            raise ValueError(f"Invalid temporal resolution: {config['data_params']['temporal_resolution']}. Expected 10 or 20.")
    elif config['model_params']['name'] == 'Unet':
        # Define the URL and the destination path
        if config['data_params']['temporal_resolution'] == 20:
            checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_unet_T=20.ckpt")
            url = 'https://www.dropbox.com/scl/fi/w18h09705isxqym60d51k/checkpoint_unet_T-20.ckpt?rlkey=uml6l3gk7i4mxtthdi9plihme&dl=1'
            subprocess.run(['wget', '-O', checkpoint_path, url], check=True)
            print(f"Download complete.")
        elif config['data_params']['temporal_resolution'] == 10:
            checkpoint_path = os.path.join(checkpoints_dir, "checkpoint_unet_T=10.ckpt")
            url = 'https://www.dropbox.com/scl/fi/h84ffibzbh5qvmshh8qbt/checkpoint_unet_T-10.ckpt?rlkey=zrgitxeqaoy6m7rewfpsi524m&dl=1'
            subprocess.run(['wget', '-O', checkpoint_path, url], check=True)
            print(f"Download complete.")
        else: 
            raise ValueError(f"Invalid temporal resolution: {config['data_params']['temporal_resolution']}. Expected 10 or 20.")
    else: 
        raise ValueError(f"Invalid model name: {config['model_params']['name']}. Expected VAE or DDL.")
                
    
    # Now checkpoint_path contains the downloaded checkpoint
    checkpoint = torch.load(checkpoint_path)
else:
    # Sort the files so that the latest one (e.g., based on timestamp in the name) comes last
    checkpoint_files.sort()
        
    # Pick the latest checkpoint
    latest_checkpoint = checkpoint_files[-1]
        
    # Load the weights from the latest checkpoint
    checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
        
    checkpoint = torch.load(checkpoint_path)

print(checkpoint_path)
# Load the state dictionary into the model/experiment
experiment.load_state_dict(checkpoint['state_dict'])

# Run the model on the test dataset
test_loss = runner.test(experiment, datamodule=data)

test_dataset = data.test_dataloader().dataset
x_test = torch.stack([data[0] for data in test_dataset])
y_test = torch.stack([data[1] for data in test_dataset])

x_test_ = np.squeeze(x_test.numpy())
y_test_ = y_test.numpy()


if (config.get('model_params', {}).get('name') == 'Unet') and (config.get('test_params', {}).get('save_clouds')):
    train_results = runner.predict(experiment, dataloaders=data.train_dataloader())
    output_list = []
    for i in range(len(train_results)):
        results = train_results[i]
        output_list.append(results[0])
    output_train = torch.cat(output_list, dim=0)
    output_train[output_train <=0] = 0

    test_results = runner.predict(experiment, dataloaders=data.test_dataloader())
    output_list = []
    for i in range(len(test_results)):
        results = test_results[i]
        output_list.append(results[0])
    output_test = torch.cat(output_list, dim=0)
    output_test[output_test <=0] = 0

    with h5py.File(os.path.join(config["data_params"]["data_path"],'outputs_stage1_T='+str(config["data_params"]["temporal_resolution"])+'.h5'), 'w') as hf:
        # Using gzip compression and automatic chunking
        hf.create_dataset("output_train", data=output_train.cpu().numpy(), compression='gzip')
        
        # Using gzip compression and automatic chunking
        hf.create_dataset("output_test", data=output_test.cpu().numpy(), compression='gzip')


if (config.get('model_params', {}).get('name') == 'Unet') and (config.get('test_params', {}).get('test_cloud')):
    j = config['test_params']['cloud_id']
    j = 10
    input = x_test[j:j+1]
    input_ = np.squeeze(x_test_[j:j+1]).transpose((1,2,3,0))
    results = experiment.model.forward(input)
    output = results[0]
    output_ = output.squeeze().detach().numpy()

    target = y_test[j:j+1].squeeze()
    target_ = target.numpy()
    loss = F.mse_loss(target , output)
    print(loss)

    print('np.max(target_: ',np.max(target_[-1,...]))
    print('np.max(output_: ',np.max(output_[-1,...]))

    array = np.arange(0, 1)

    for i in array:
        for t in range(len(target_)):
            print(t)
            fig, axes = plt.subplots(1, 3, figsize=(8, 4))

            # Function to remove ticks from an axis
            def remove_ticks(ax):
                ax.set_xticks([])
                ax.set_yticks([])

            # Remove ticks for both axes
            for ax in axes:
                remove_ticks(ax)

            vmin_ = np.min(target_[t, ...])
            vmax_ = np.max(target_[t, ...])

            # Display the true image
            true_img = axes[0].imshow(input_[t, ...])
            axes[0].set_title(f'Input - Timestep {t}')
            fig.colorbar(true_img, ax=axes[0], fraction=0.046, pad=0.04, extend='max')

            # Display the predicted mu mu image
            pred_img = axes[1].imshow(output_[t, ...].reshape((128,128)), cmap='inferno', vmin=vmin_, vmax=vmax_)
            axes[1].set_title(f'Output - loss: {loss:.2e}')
            fig.colorbar(pred_img, ax=axes[1], fraction=0.046, pad=0.04, extend='max')

            # Display the true image
            true_img = axes[2].imshow(target_[t, ...].reshape((128,128)), cmap='inferno', vmin=vmin_, vmax=vmax_)
            axes[2].set_title(f'Target - Timestep {t}')
            fig.colorbar(true_img, ax=axes[2], fraction=0.046, pad=0.04, extend='max')

            plt.savefig(os.path.join(tb_logger.log_dir,f'Predicted_cloud_frame_{t}.png'), format='png', bbox_inches='tight')
            plt.show()
            plt.pause(1)


def compute_median_quartiles(data):
    """
    Computes the median, lower quartile (Q1), and upper quartile (Q3) for a given dataset.

    Parameters:
        data (list or np.array): Input data values.

    Returns:
        tuple: (median, Q1, Q3), all rounded to 2 decimal places.
    """
    if len(data) == 0 or not np.any(np.isfinite(data)):  # Ensure non-empty, finite data
        return 0.0, 0.0, 0.0  # Return 0.0 instead of None for safe handling

    median = np.round(np.median(data), 2)
    q1 = np.round(np.percentile(data, 5), 2)  # 5th percentile
    q3 = np.round(np.percentile(data, 95), 2) # 95th percentile

    return float(median), float(q1), float(q3)  # Ensure float type


if (config.get('model_params', {}).get('name') == 'VAE') and (config.get('test_params', {}).get('test_params')):
    compute_quartiles = True
    test_results = runner.predict(experiment, datamodule=data)

    output_list, mean1_list, var1_list, z1_list = [], [], [], []
    mean2_list, var2_list, z2_list = [], [], []
    mean3_list, var3_list, z3_list = [], [], []
    mean_new_list, var_new_list, z_new_list = [], [], []

    for results in test_results:
        output_list.append(results[0])
        mean1_list.append(results[1])
        var1_list.append(results[2])
        z1_list.append(results[3])
        mean2_list.append(results[4])
        var2_list.append(results[5])
        z2_list.append(results[6])
        mean3_list.append(results[7])
        var3_list.append(results[8])
        z3_list.append(results[9])

        if config['model_params']['arch_option'] != 'NO-Z2':
            mean_new_list.append(results[10])
            var_new_list.append(results[11])
            z_new_list.append(results[12])

    # Convert to NumPy arrays
    output = torch.squeeze(torch.cat(output_list, dim=0)).numpy()
    mean1, var1, z1 = map(lambda x: torch.squeeze(torch.cat(x, dim=0)).numpy(), [mean1_list, var1_list, z1_list])
    mean2, var2, z2 = map(lambda x: torch.squeeze(torch.cat(x, dim=0)).numpy(), [mean2_list, var2_list, z2_list])
    mean3, var3, z3 = map(lambda x: torch.squeeze(torch.cat(x, dim=0)).numpy(), [mean3_list, var3_list, z3_list])

    if config['model_params']['arch_option'] != 'NO-Z2':
        mean_new, var_new, z_new = map(lambda x: torch.squeeze(torch.cat(x, dim=0)).numpy(), [mean_new_list, var_new_list, z_new_list])

    # Set output file path
    output_file_path = os.path.join(tb_logger.log_dir, 'parameter_estimation_error_new.txt')

    with open(output_file_path, 'w') as f:
        original_stdout = sys.stdout  # Save original stdout
        sys.stdout = f  # Redirect stdout to file
        print("MAE and Uncertainty results over the testing dataset.")

        try:
            for metric, target_idx, mean_idx, var_idx in zip(
                ["X", "Y", "t", "u", "v", "cloud"],
                [0, 1, 2, 3, 4, None],
                [(mean1, 0), (mean1, 1), (mean2, None), (mean3, 0), (mean3, 1), (output, None)],
                [(var1, 0), (var1, 1), (var2, None), (var3, 0), (var3, 1), (None, None)]
            ):
                loss = []
                target = y_test_[:, target_idx] if target_idx is not None else x_test_
                prediction = mean_idx[0][:, mean_idx[1]] if mean_idx[1] is not None else mean_idx[0][:]

                target = torch.from_numpy(target)
                prediction = torch.from_numpy(prediction)

                for i in range(len(target)):
                    loss.append(F.l1_loss(target[i], prediction[i]).item())

                # Apply physical unit transformations
                if metric in ["X", "Y"]:
                    loss = [round(val * 4, 2) for val in loss]
                elif metric in ["t", "u", "v"]:
                    loss = [round(val * 2, 2) for val in loss]

                if compute_quartiles:  # Use quartiles if enabled
                    median_loss, q1_loss, q3_loss = compute_median_quartiles(loss)
                    print(f'Median {metric} = {median_loss} ({q1_loss}, {q3_loss})')
                else:
                    mean_loss = round(np.mean(loss), 4 if metric == "cloud" else 2)
                    std_loss = round(np.std(loss), 4 if metric == "cloud" else 2)
                    print(f'MAE {metric} = {mean_loss} +/- {std_loss}')

                if var_idx[0] is not None:
                    uncertainty_data = var_idx[0][:, var_idx[1]] if var_idx[1] is not None else var_idx[0][:]

                    # Apply physical unit transformations
                    if metric in ["X", "Y"]:
                        uncertainty_data = [round(val * 4, 2) for val in uncertainty_data]
                    elif metric in ["t", "u", "v"]:
                        uncertainty_data = [round(val * 2, 2) for val in uncertainty_data]

                    if compute_quartiles:
                        median_unc, q1_unc, q3_unc = compute_median_quartiles(uncertainty_data)
                        print(f'Uncertainty {metric} = {round(median_unc, 2)} ({round(q1_unc, 2)}, {round(q3_unc, 2)})')
                    else:
                        uncertainty_mean = round(np.mean(uncertainty_data), 2)
                        uncertainty_std = round(np.std(uncertainty_data), 2)
                        print(f'Uncertainty {metric} = {uncertainty_mean:.2f} +/- {uncertainty_std:.2f}')

        except Exception as e:
            print(f"An error occurred: {e}")

        sys.stdout = original_stdout  # Reset stdout


if (config.get('model_params', {}).get('name') == 'DDL') and (config.get('test_params', {}).get('test_params')):
    compute_quartiles = True
    test_results = runner.predict(experiment, datamodule=data)

    mean1_list = []
    mean2_list = []
    mean3_list = []

    for results in test_results:
        mean1_list.append(results[0])
        mean2_list.append(results[1])
        mean3_list.append(results[2])

    # Convert to NumPy arrays
    mean1 = [torch.squeeze(torch.cat(x, dim=0)).numpy() for x in [mean1_list]][0]
    mean2 = [torch.squeeze(torch.cat(x, dim=0)).numpy() for x in [mean2_list]][0]
    mean3 = [torch.squeeze(torch.cat(x, dim=0)).numpy() for x in [mean3_list]][0]


    # Set output file path
    output_file_path = os.path.join(tb_logger.log_dir, 'parameter_estimation_error_new.txt')

    with open(output_file_path, 'w') as f:
        original_stdout = sys.stdout  # Save original stdout
        sys.stdout = f  # Redirect stdout to file
        print("Median and Uncertainty results over the testing dataset (T = 20)")

        try:
            for metric, target_idx, mean_idx in zip(
                ["X", "Y", "t", "u", "v"],
                [0, 1, 2, 3, 4, None],
                [(mean1, 0), (mean1, 1), (mean2, None), (mean3, 0), (mean3, 1)],
            ):
                loss = []
                target = y_test_[:, target_idx] if target_idx is not None else x_test_
                prediction = mean_idx[0][:, mean_idx[1]] if mean_idx[1] is not None else mean_idx[0][:]

                target = torch.from_numpy(target)
                prediction = torch.from_numpy(prediction)

                for i in range(len(target)):
                    loss.append(F.l1_loss(target[i], prediction[i]).item())

                # Apply physical unit transformations
                if metric in ["X", "Y"]:
                    loss = [round(val * 4, 2) for val in loss]
                elif metric in ["t", "u", "v"]:
                    loss = [round(val * 2, 2) for val in loss]

                if compute_quartiles:  # Use quartiles if enabled
                    median_loss, q1_loss, q3_loss = compute_median_quartiles(loss)
                    print(f'Median {metric} = {median_loss} ({q1_loss}, {q3_loss})')
                else:
                    mean_loss = round(np.mean(loss), 4 if metric == "cloud" else 2)
                    std_loss = round(np.std(loss), 4 if metric == "cloud" else 2)
                    print(f'MAE {metric} = {mean_loss} +/- {std_loss}')

        except Exception as e:
            print(f"An error occurred: {e}")

        sys.stdout = original_stdout  # Reset stdout


# if (config.get('model_params', {}).get('name') == 'DDL') and (config.get('test_params', {}).get('test_params')):
#     test_results = runner.predict(experiment, datamodule=data)

#     mean1_list = []
#     mean2_list = []
#     mean3_list = []
#     for i in range(len(test_results)):
#         results = test_results[i]
#         mean1_list.append(results[0])
#         mean2_list.append(results[1])
#         mean3_list.append(results[2])
#     mean1 = torch.squeeze(torch.cat(mean1_list, dim=0)).numpy()
#     mean2 = torch.squeeze(torch.cat(mean2_list, dim=0)).numpy()
#     mean3 = torch.squeeze(torch.cat(mean3_list, dim=0)).numpy()

#     # Set the output file path
#     output_file_path = os.path.join(tb_logger.log_dir, f'parameter_estimation_error.txt')

#     # Redirect stdout to the file
#     with open(output_file_path, 'w') as f:
#         original_stdout = sys.stdout  # Save the original standard output
#         sys.stdout = f  # Redirect stdout to file
#         print("MSE results averaged over the testing dataset.")  # Initial line in the file
        
#         try:
#             for metric, target_idx, mean_idx in zip(
#                 ["X", "Y", "t", "u", "v"],
#                 [0, 1, 2, 3, 4],
#                 [(mean1, 0), (mean1, 1), (mean2, None), (mean3, 0), (mean3, 1)]
#             ):
#                 loss = []
#                 target = y_test_[:, target_idx]
                
#                 if mean_idx[1] is not None:
#                     prediction = mean_idx[0][:, mean_idx[1]]
#                 else:
#                     prediction = mean_idx[0][:]
                
#                 target = torch.from_numpy(target)
#                 prediction = torch.from_numpy(prediction)
                
#                 for i in range(len(target)):
#                     loss.append(F.mse_loss(target[i], prediction[i]).item())  # Use `.item()` to get a Python number from a tensor containing a single value
                
#                 mean_loss = round(np.mean(loss), 2)
#                 std_loss = round(np.std(loss), 2)
                
#                 print(f'MSE {metric} = {mean_loss} +/- {std_loss}')
        
#         except Exception as e:
#             print(f"An error occurred: {e}")
        
#         sys.stdout = original_stdout  # Reset stdout to its original state


# if (config.get('model_params', {}).get('name') == 'VAE') and (config.get('test_params', {}).get('test_params')):
#     test_results = runner.predict(experiment, datamodule=data)

#     output_list = []
#     mean1_list = []
#     var1_list = []
#     z1_list = []
#     mean2_list = []
#     var2_list = []
#     z2_list = []
#     mean3_list = []
#     var3_list = []
#     z3_list = []
#     mean_new_list = []
#     var_new_list = []
#     z_new_list = []

#     for i in range(len(test_results)):
#         results = test_results[i]

#         output_list.append(results[0])
#         mean1_list.append(results[1])
#         var1_list.append(results[2])
#         z1_list.append(results[3])
#         mean2_list.append(results[4])
#         var2_list.append(results[5])
#         z2_list.append(results[6])
#         mean3_list.append(results[7])
#         var3_list.append(results[8])
#         z3_list.append(results[9])
#         if config['model_params']['arch_option'] != 'NO-Z2':
#             mean_new_list.append(results[10])
#             var_new_list.append(results[11])
#             z_new_list.append(results[12])

#     output = torch.squeeze(torch.cat(output_list, dim=0)).numpy()
#     mean1 = torch.squeeze(torch.cat(mean1_list, dim=0)).numpy()
#     var1 = torch.squeeze(torch.cat(var1_list, dim=0)).numpy()
#     z1 = torch.squeeze(torch.cat(z1_list, dim=0)).numpy()
#     mean2 = torch.squeeze(torch.cat(mean2_list, dim=0)).numpy()
#     var2 = torch.squeeze(torch.cat(var2_list, dim=0)).numpy()
#     z2 = torch.squeeze(torch.cat(z2_list, dim=0)).numpy()
#     mean3 = torch.squeeze(torch.cat(mean3_list, dim=0)).numpy()
#     var3 = torch.squeeze(torch.cat(var3_list, dim=0)).numpy()
#     z3 = torch.squeeze(torch.cat(z3_list, dim=0)).numpy()
#     if config['model_params']['arch_option'] != 'NO-Z2':
#         mean_new = torch.squeeze(torch.cat(mean_new_list, dim=0)).numpy()
#         var_new = torch.squeeze(torch.cat(var_new_list, dim=0)).numpy()
#         z_new = torch.squeeze(torch.cat(z_new_list, dim=0)).numpy()

#     # Set the output file path
#     output_file_path = os.path.join(tb_logger.log_dir, f'parameter_estimation_error.txt')

#     # Redirect stdout to the file
#     with open(output_file_path, 'w') as f:
#         original_stdout = sys.stdout  # Save the original standard output
#         sys.stdout = f  # Redirect stdout to file
#         print("MAE and Uncertainty results averaged over the testing dataset.")  # Initial line in the file

#         try:
#             for metric, target_idx, mean_idx, var_idx in zip(
#                 ["X", "Y", "t", "u", "v", "cloud"],
#                 [0, 1, 2, 3, 4, None],
#                 [(mean1, 0), (mean1, 1), (mean2, None), (mean3, 0), (mean3, 1), (output, None)],
#                 [(var1, 0), (var1, 1), (var2, None), (var3, 0), (var3, 1), (None, None)]
#             ):
#                 loss = []
#                 target = y_test_[:, target_idx] if target_idx is not None else x_test_
#                 prediction = mean_idx[0][:, mean_idx[1]] if mean_idx[1] is not None else mean_idx[0][:]
                
#                 target = torch.from_numpy(target)
#                 prediction = torch.from_numpy(prediction)
                
#                 for i in range(len(target)):
#                     loss.append(F.l1_loss(target[i], prediction[i]).item())  # Use `.item()` to get a Python number from a tensor containing a single value 
                
#                 mean_loss = round(np.mean(loss), 4 if metric == "cloud" else 2)
#                 std_loss = round(np.std(loss), 4 if metric == "cloud" else 2)
                
#                 print(f'MAE {metric} = {mean_loss} +/- {std_loss}')

#                 if var_idx[0] is not None:
#                     uncertainty_data = var_idx[0][:, var_idx[1]] if var_idx[1] is not None else var_idx[0][:]
#                     uncertainty_mean = round(np.mean(uncertainty_data), 2)
#                     uncertainty_std = round(np.std(uncertainty_data), 2)
                    
#                     print(f'Uncertainty {metric} = {uncertainty_mean:.2f} +/- {uncertainty_std:.2f}')

#         except Exception as e:
#             print(f"An error occurred: {e}")

#         sys.stdout = original_stdout  # Reset stdout to its original state



if (config.get('model_params', {}).get('name') == 'VAE') and (config.get('test_params', {}).get('test_cloud')):

    j = config['test_params']['cloud_id']
    target = x_test[j:j+1]
    target_ = np.squeeze(target.numpy())
    print(y_test_[j])

    import time

    # Start timing for the first command
    start_time_1 = time.time()
    # Execute the first command
    results = experiment.model.encode(target)
    # End timing for the first command
    end_time_1 = time.time()
    execution_time_1 = end_time_1 - start_time_1


    conv33 = results[0]
    mean1  = results[1]
    var1   = results[2]
    z1     = results[3]
    mean2  = results[4]
    var2   = results[5]
    z2     = results[6]
    mean3  = results[7]
    var3   = results[8]
    z3     = results[9]
    mean_new  = results[10]
    var_new   = results[11]
    z_new     = results[12]

    # Start timing for the second command
    start_time_2 = time.time()
    # Execute the second command
    output_mu_mu = experiment.model.decode([mean1, mean2, mean3, mean_new])
    # End timing for the second command
    end_time_2 = time.time()
    execution_time_2 = end_time_2 - start_time_2
    # Calculate overall execution time
    overall_time = (execution_time_1 + execution_time_2)

    # Print the results
    print(f"Time for first command: {execution_time_1} seconds")
    print(f"Time for second command: {execution_time_2} seconds")
    print(f"Overall time: {overall_time} seconds")

    output_mu_mu_ = output_mu_mu.squeeze(1).detach().numpy()
    loss_mu_mu = F.mse_loss(target , output_mu_mu)
    print(loss_mu_mu)

    print('np.max(target_[0,-1,:,:]: ',np.max(target_[-1,:,:]))
    print('np.max(output_mu_mu_[0,-1,:,:]: ',np.max(output_mu_mu_[0,-1,:,:]))

    array = np.arange(0, 1)

    for i in array:
        for t in range(len(target_)):
            print(t)
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            # Function to remove ticks from an axis
            def remove_ticks(ax):
                ax.set_xticks([])
                ax.set_yticks([])

            # Remove ticks for both axes
            for ax in axes:
                remove_ticks(ax)

            vmin_ = 0
            vmax_ = np.max(target_[t, ...])
            if vmax_ < 0.1:
                vmax_ = 1

            # Display the true image
            true_img = axes[0].imshow(target_[t, ...].reshape((128,128)), cmap='inferno', vmin=vmin_, vmax=vmax_)
            axes[0].set_title(f'True - Timestep {t}')
            fig.colorbar(true_img, ax=axes[0], fraction=0.046, pad=0.04, extend='max')

            # Display the predicted mu mu image
            pred_img = axes[1].imshow(output_mu_mu_[i, t, ...].reshape((128,128)), cmap='inferno', vmin=vmin_, vmax=vmax_)
            axes[1].set_title(f'Predicted - loss: {loss_mu_mu:.5f}')
            fig.colorbar(pred_img, ax=axes[1], fraction=0.046, pad=0.04, extend='max')

            plt.savefig(os.path.join(tb_logger.log_dir,f'Predicted_cloud_frame_{t}.png'), format='png', bbox_inches='tight')
            plt.show()
            plt.pause(1)

if (config.get('model_params', {}).get('name') == 'VAE') and (config.get('test_params', {}).get('generate_cloud')):

    # Generate a 1x2 tensor with random integers between -50 and 50 for a1
    a1 = torch.randint(-50, 51, (1, 2))
    # Generate a 1x1 tensor with random integers between 0 and 19 for a2
    a2 = torch.randint(0, config["data_params"]["temporal_resolution"], (1, 1))
    # Generate a 1x2 tensor with random integers between -2 and 2 for a3
    a3 = torch.randint(-2, 3, (1, 2))
    a_new = torch.randn((64)).unsqueeze(0)
    output_new_z_z = experiment.model.decode([a1,a2,a3,a_new])
    output_new_z_z_ = output_new_z_z.squeeze(1).detach().numpy()

    array = np.arange(0, 1)
    for i in array:
        for t in range(output_new_z_z_.shape[1]):
            print(t)
            fig, ax = plt.subplots(figsize=(4, 4))

            # Function to remove ticks from an axis
            def remove_ticks(ax):
                ax.set_xticks([])
                ax.set_yticks([])

            # Remove ticks for the axis
            remove_ticks(ax)

            vmin_ = 0
            vmax_ = np.max(output_new_z_z_[i, t, ...])
            if vmax_ < 0.1:
                vmax_ = 1

            # Display the predicted new mu mu image
            pred_img = ax.imshow(output_new_z_z_[i, t, ...].reshape((128,128)), cmap='inferno', vmin=vmin_, vmax=vmax_)
            ax.set_title(f'x={a1[0,0].numpy()}, y={a1[0,1].numpy()}, t={a2[0,0].numpy()}, u={a3[0,0].numpy()}, v={a3[0,1].numpy()}')
            fig.colorbar(pred_img, ax=ax, fraction=0.046, pad=0.04, extend='max')

            plt.savefig(os.path.join(tb_logger.log_dir,f'New_cloud_frame_{t}.png'), format='png', bbox_inches='tight')
            plt.show()
            plt.pause(1)

def delete_extra_files(directory_path):
    # Patterns for TensorBoard and hparams.yaml files
    tensorboard_pattern = os.path.join(directory_path, 'events.out.tfevents.*')
    hparams_file_path = os.path.join(directory_path, 'hparams.yaml')
    
    # Use glob to find all TensorBoard files
    tensorboard_files = glob.glob(tensorboard_pattern)
    
    # Delete TensorBoard files
    for file_path in tensorboard_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {str(e)}")
            
    # Delete hparams.yaml
    try:
        os.remove(hparams_file_path)
        print(f"Deleted: {hparams_file_path}")
    except Exception as e:
        print(f"Failed to delete {hparams_file_path}: {str(e)}")


# Then, call the function to delete TensorBoard files
delete_extra_files(tb_logger.log_dir)

print('Done!')

# Plotting
from matplotlib.ticker import FuncFormatter, MaxNLocator
def plot_generated_image(image,x,y,frame,max_value,xs,ys,ts,ue,ve):
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='inferno', vmax=max_value)

    # Remove x,y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar closer to the image and with LaTeX-formatted ticks
    cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax, format=FuncFormatter(lambda x, _: f"${x:.2f}$"))
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = MaxNLocator(nbins=5)  # Set the number of ticks
    cbar.update_ticks()

    # Scatter plot with reduced marker size
    ax.scatter(x, y, c='green', s=10)  # Note that x and y are not swapped here

    # Add LaTeX-formatted text above the point with increased font size
    ax.text(x, y-5, f"$({{{x-64}}}, {{{y-64}}})$", color='white', ha='center', va='bottom', fontsize=20)
    # Save the image without extra whitespace
    plt.savefig(f'image2_T={config["data_params"]["temporal_resolution"]}_xs={xs}_ys={ys}_ts={ts}_ue={ue}_ve={ve}_max={max_value:.2f}_frame={frame}.png', bbox_inches='tight')

    # Clear the figure
    plt.clf()

# import numpy as np
# for i in range(100):
#     # Generate a 1x2 tensor with random integers between -50 and 50 for a1
#     a1 = torch.randint(-50, 51, (1, 2))
#     # Generate a 1x1 tensor with random integers between 0 and 19 for a2
#     a2 = torch.randint(0, config["data_params"]["temporal_resolution"], (1, 1))
#     # Generate a 1x2 tensor with random integers between -2 and 2 for a3
#     a3 = torch.randint(-2, 3, (1, 2))
#     a_new = torch.randn((64)).unsqueeze(0)
#     output_ = experiment.model.decode([a1,a2,a3,a_new])

#     # Convert PyTorch tensors to NumPy arrays
#     a1_np = a1.numpy()
#     a2_np = a2.numpy()
#     a3_np = a3.numpy()
#     a_new_np = a_new.numpy()
#     output = output_.squeeze().detach().numpy()

#     xs = a1_np[0,0]
#     ys = a1_np[0,1]
#     ts = a2_np[0][0]
#     ue = a3_np[0,0]
#     ve = a3_np[0,1]

#     max_value = np.max(output)
#     print(f"The peak intensity is {max_value}")
#     if max_value > 4.7:
#         break


# data = loadmat('generated_cube2_T=10_ICSSP_.mat')
# a1 = data['a1']
# a2 = data['a2']
# a3 = data['a3']
# a_new = data['a_new']
# output = data['output']
# xs = a1[0,0]
# ys = a1[0,1]
# ts = a2[0][0]
# ue = a3[0,0]
# ve = a3[0,1]


# image ts
# t = ts
# image = output[ts,...]
# image[image<0]=0
# max_value = np.max(image)
# ref_x = xs + 64
# ref_y = ys + 64

# print(f"The peak intensity is {max_value}")
# print(f"ts is {ts}")
# print(f"The reference coordinates relative to the center are ({xs}, {ys})")
# plot_generated_image(image,ref_x,ref_y,t,max_value,xs,ys,ts,ue,ve)


# # image -1
# t = config["data_params"]["temporal_resolution"]
# image = output[-1,...]
# image[image<0]=0
# max_value = np.max(image)
# ref_x = xs + 64 + (t - ts) * ue
# ref_y = ys + 64 + (t - ts) * ve

# print(f"The peak intensity is {max_value}")
# print(f"ts is {ts}")
# print(f"The reference coordinates relative to the center are ({ref_x}, {ref_y})")
# plot_generated_image(image,ref_x,ref_y,t,max_value,xs,ys,ts,ue,ve)


# from scipy.io import savemat, loadmat
# # Save to .mat file
# savemat('generated_cube2_T='+str(t)+'_ICSSP_.mat', {'a1': a1_np, 'a2': a2_np, 'a3': a3_np, 'a_new': a_new_np, 'output':output})