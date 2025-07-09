import fire
import random
import subprocess
import wandb

import numpy as np
import torch
import torch.distributed as dist
from torchsummary import summary

# # Add the project root directory to the Python path
import os
import pathlib
import sys
# Add this line to help with debugging CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import trainer and scheduler
from dna.datasets import get_imagenet_dataloaders, get_cifar10_dataloaders
from dna.models import build_model
from dna.optim import setup_scheduler, setup_optimizer, _low_lr_for_non_matrix
from dna.utils import load_config, get_slurm_id, build_name_str
from dna.training import BaseTrainer, DNATrainer


def main_worker(
    config_path: str, 
    save_dir: str, 
    data_dir: str, 
    resume: bool = False, 
    resume_id: str = None, 
    overrides: dict = None,
    name_prefix: str='',
):
    """
    Main worker function for each GPU.
    This function initializes the process group, sets the device,
    and runs training.
    """
    
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    slurm_id = get_slurm_id() if resume_id is None else resume_id

    print(f"Using distributed environment variables: local_rank={local_rank}, world_size={world_size}")
    
    # Let torchrun handle the environment variables for distributed initialization
    print(f"Local rank: {local_rank}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # Initialize the process group (using NCCL backend for GPUs)
    if not dist.is_initialized():
        print(f"Initializing process group: rank={local_rank}, world_size={world_size}")
        try:
            # Let torchrun handle the initialization with its environment variables
            dist.init_process_group(backend="nccl")
            print(f"Process group initialized successfully for rank {local_rank}")
        except Exception as e:
            print(f"Error initializing process group for rank {local_rank}: {e}")
            raise
    if local_rank >= torch.cuda.device_count():
        raise ValueError(f"Device {local_rank} is not available")
    
    # Set the CUDA device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"[Rank {local_rank}] Using device: {device}")
    
    # Load configuration
    model_config, training_config, data_config, logging_config = load_config(config_path, overrides)
    
    torch.manual_seed(training_config.seed + local_rank)
    np.random.seed(training_config.seed + local_rank)
    random.seed(training_config.seed + local_rank)
    
    # Override data directory if provided
    if data_dir is not None:
        data_config.data_dir = data_dir
    
    # Get model_type and build a name for saving
    model_type = getattr(model_config, "model_type", "vit")  # Default to 'vit' if not specified
    model_type = model_type.lower()

    name_str = build_name_str(model_config=model_config, training_config=training_config, data_config=data_config)

    if save_dir is None:
        save_dir = pathlib.Path(logging_config.save_dir) / data_config.dataset / logging_config.project / (name_prefix + name_str)
        
    # Create save directory only on rank 0, modified to save in separate folders.
    if local_rank == 0:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Copy data to scratch folder first if use h5py
    target_dir = data_config.root_dir
    if data_config.h5py is True:
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
        target_dir = f'/scratch/slurm_tmpdir/{slurm_job_id}_{slurm_task_id}'
        # Only rank 0 performs the copy operation
        if local_rank == 0:
            # Check if the target directory already exists
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                # Use rsync to copy the dataset
                rsync_command = ['rsync', '-av', data_config.root_dir, target_dir]
                subprocess.run(rsync_command, check=True)
                print(f"Dataset copied to {target_dir}")
            else:
                print(f"Target directory {target_dir} already exists. Skipping copy.")
        # Synchronize all processes
        dist.barrier()
    else:
        target_dir = data_config.root_dir

    # get dataloaders
    match data_config.dataset.lower():
        case "imagenet":
            train_loader, val_loader, _ = get_imagenet_dataloaders(
                batch_size=training_config.batch_size,
                root_dir=target_dir,
                image_size=data_config.image_size,
                dataset_fraction=data_config.dataset_frac,
                h5py=data_config.h5py,  # Only use on aws
                num_workers=data_config.num_workers,  # each process gets its own workers
                distributed=dist.is_initialized(),  # flag to use DistributedSampler in the dataloader
                rank=local_rank,
                world_size=world_size,
                mixup=data_config.mixup,
                pin_memory=True
            )
        case "cifar10":
            train_loader, val_loader, _ = get_cifar10_dataloaders(
                batch_size=training_config.batch_size,
                root_dir=target_dir,
                num_workers=data_config.num_workers,  # each process gets its own workers
                distributed=dist.is_initialized(),  # flag to use DistributedSampler in the dataloader
                rank=local_rank,
                world_size=world_size
            )

    if not all(isinstance(loader, torch.utils.data.DataLoader) for loader in [train_loader, val_loader]):
        raise ValueError("Data loaders have not been created successfully")
    
    # Setup model, optimizer, and scheduler
    model = build_model(model_config).to(device)
    if local_rank == 0:
        summary(model)
    optimizer = setup_optimizer(model, training_config, customized_param_func=None)
    scheduler = setup_scheduler(optimizer, training_config, train_loader, init_lr=1e-7, ending_lr=1e-6)
    
    if local_rank == 0:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key is None:
            raise ValueError("WANDB_API_KEY can not be located.")
        # Set the host and login
        wandb.login(host="https://fairwandb.org/", key=api_key)

        # for wandb grouping access
        config_dict = {}
        config_dict.update({'model_' + key: val for key, val in model_config.__dict__.items()})
        config_dict.update({'training_' + key: val for key, val in training_config.__dict__.items()}) 

        config_dict['model_config'] = model_config
        config_dict['training_config'] = training_config
        config_dict['data_config'] = data_config
        config_dict['logging_config'] = logging_config

        wandb_logger = wandb.init(
            project=logging_config.project,
            entity=logging_config.entity,
            dir=logging_config.save_dir + '/wandb_log',
            config=config_dict,
            group=f"{model_type}",  # all runs for the experiment in one group
            id=slurm_id,
            resume="allow"
        )
        wandb_logger.name = name_prefix + name_str
    else:
        wandb_logger = None

    # Create trainer. (Your Trainer class should wrap the model with DDP if the process group is initialized.)
    if "dna" in model_type:
        trainer = DNATrainer(
            training_config=training_config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            slurm_id=slurm_id,
            save_dir=save_dir,
            device=device,
            rank=local_rank,
            resume=resume,
            wandb_logger=wandb_logger,
            find_unused_parameters=True,  # ddp unfortunately ocasionally for some batch zero tokens go through one specific experts, which would trigger training failure without setting this to True
            static_graph=False,  # ddp
            use_compile=False,
            print_freq=logging_config.print_freq
        )
    else:
        trainer = BaseTrainer(
            training_config=training_config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            slurm_id=slurm_id,
            save_dir=save_dir,
            device=device,
            rank=local_rank,
            resume=resume,
            wandb_logger=wandb_logger,
            find_unused_parameters=False,  # ddp
            static_graph=True,  # ddp
            use_compile=False,
            print_freq=logging_config.print_freq
        )

    dist.barrier()
    trainer.train()
    dist.destroy_process_group()

def main(
    config_path: str = "configs/config.yaml", 
    save_dir: str = None, 
    data_dir: str = None, 
    resume: bool = False,
    resume_id: str = None,
    name_prefix: str='',
    **overrides
):
    """
    Main training function.
    
    Args:
        config_path: Path to the configuration YAML file.
        save_dir: Directory to save checkpoints and logs.
        data_dir: Optional override for data directory from config.
        world_size: Number of GPUs to use.
    """
    # Spawn one process per GPU.
    main_worker(config_path, save_dir, data_dir, resume, resume_id, 
                name_prefix=name_prefix, overrides=overrides)
        
if __name__ == '__main__':
    fire.Fire(main)
