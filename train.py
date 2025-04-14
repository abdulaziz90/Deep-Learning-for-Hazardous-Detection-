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
from torchinfo import summary
import glob

'''
python train.py -c configs/unet.yaml --temporal_resolution 10 --save_name 'unet_T=10' --max_epochs 3 --gpu 0
python train.py -c configs/unet.yaml --temporal_resolution 10 --save_name 'unet_T=10' --max_epochs 15 --gpu 0,1
python train.py -c configs/ddl.yaml --temporal_resolution 10 --save_name 'ddl_T=10_new' --max_epochs 3 --gpu 2
python train.py -c configs/vae.yaml --temporal_resolution 10 --save_name 'vae_T=10' --max_epochs 1 --gpu 0

'''

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/ddl.yaml')

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

model = vae_models[config['model_params']['name']](**config['model_params'], data_path=config["data_params"]["data_path"], temporal_resolution
=config["data_params"]["temporal_resolution"])

device = next(model.parameters()).device
model = model.to(device)

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


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)

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