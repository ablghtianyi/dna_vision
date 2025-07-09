import math
import time

import torch
import torch.nn as nn

import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from pathlib import Path
from typing import Dict, Any, Optional, Union, Dict


class BaseTrainer:
    def __init__(
        self,
        training_config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        slurm_id: str,
        save_dir: str = 'checkpoints',
        device: str = 'cuda',
        rank: int = 0,  # New parameter for distributed training
        resume: bool = False,
        wandb_logger: Dict = None,
        find_unused_parameters: bool = False,  # ddp
        static_graph: bool = True,  # ddp
        use_compile: bool = False,  # Enable torch.compile
        print_freq: int = 50,
    ):
        # Correctly extract local_rank from the device.
        if isinstance(device, torch.device):
            local_rank = device.index
        elif isinstance(device, str):
            local_rank = int(device.split(':')[-1])
        else:
            local_rank = 0

        # Initialize NVML on the local GPU (use local rank as the GPU index)    
        # pynvml.nvmlInit()
        # self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        self.rank = rank
        self.device = device
        
        # Load config
        self.training_config = training_config
        
        # Move model to device and initialize optimizer, scheduler, and loss function
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

        # wandb flags
        self.wandb_logger = wandb_logger

        # Compile flags
        self.use_compile = use_compile

        # These will be overwritten if resuming from checkpoint
        self.global_step = 0
        self.epoch = 0
        self.best_val_acc = 0.0

        # Handle resume logic before creating directories and log files
        if resume is True:
            # Load checkpoint first to get the original slurm job id
            self.load_checkpoint(f'{save_dir}/{slurm_id}/latest_checkpoint.pt')
            self.slurm_id = slurm_id
            print(f"Resuming from checkpoint with SLURM ID: {self.slurm_id}")
        else:
            self.slurm_id = slurm_id

        # Apply torch.compile if enabled (must be done before DDP wrapping)
        # if self.use_compile and hasattr(torch, 'compile'):
        #     print(f"Rank {self.rank}: Compiling model with options: {self.compile_options}")
        #     self.model = torch.compile(self.model, **self.compile_options)
        #     print(f"Rank {self.rank}: Model compilation complete")
        # elif self.use_compile and not hasattr(torch, 'compile'):
        #     print(f"Rank {self.rank}: Warning - torch.compile not available, skipping compilation")
        #     self.use_compile = False

        # Wrap the model with DistributedDataParallel if a process group is initialized.
        if dist.is_initialized():
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused_parameters,
                static_graph=static_graph
            )
        
        if self.use_compile and hasattr(torch, 'compile'):
            print(f"Rank {self.rank} compiling")
            self.model = torch.compile(self.model)
        
        if self.rank == 0:
            self.wandb_logger.watch(model, log="all", log_freq=1000)

        # Load other part of config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Create save directory with slurm_id
        self.save_dir = Path(save_dir) / self.slurm_id
        if self.rank == 0:
            self.save_dir.mkdir(exist_ok=True, parents=True)

        # Use scaler only for fp16, mixed-precision is independent of gradscaler
        torch.set_float32_matmul_precision("high")
        if training_config.dtype == "f16":
            self.amp_dtype = torch.float16
            self.use_scaler = True
        elif training_config.dtype == "bf16":
            self.amp_dtype = torch.bfloat16
            self.use_scaler = False
        else:
            self.amp_dtype = torch.float32
            self.use_scaler = False
            print("Using fp32 with 10 mantissa bits only")
            
        print(f"Using AMP dtype: {self.amp_dtype}")
        
        # Grad Scaler only for fp16
        if self.use_scaler:
            self.scaler = GradScaler(device=self.device)
            print("GradScaler enabled.")
        else:
            self.scaler = GradScaler(device=self.device, enabled=False)
            print("GradScaler disabled.")

        self.print_freq = print_freq
        self.steps_per_epoch = len(self.train_loader)

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint and optimizer state (only on rank 0)."""
        if self.rank != 0:
            return  # Only rank 0 saves checkpoints
        
        # If the model is wrapped in DDP, get the underlying module's state_dict.
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'training_config': self.training_config  # Save config for reproducibility
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / f'latest_checkpoint.pt')
        
        # Save best checkpoint if this is the best model so far
        if is_best:
            torch.save(checkpoint, self.save_dir / f'best_checkpoint.pt')

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint and optimizer state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state - handle compiled models carefully
        # For compiled models, we need to run a forward pass before loading state_dict
        # to ensure the compiled model is initialized
        if self.use_compile and hasattr(torch, 'compile'):
            # If we're using a compiled model, we need to ensure it's been traced
            # before loading the state dict. This is a workaround for the fact that
            # compiled models need to see a forward pass before state_dict loading works correctly.
            try:
                # Create a dummy input to initialize the compiled model
                # This assumes the model takes image inputs - adjust if needed
                dummy_input = torch.zeros(1, 3, 224, 224, device=self.device)
                with torch.inference_mode():
                    if hasattr(self.model, 'module'):
                        self.model.module(dummy_input)
                    else:
                        self.model(dummy_input)
                print(f"Rank {self.rank}: Initialized compiled model with dummy forward pass")
            except Exception as e:
                print(f"Rank {self.rank}: Warning - couldn't initialize compiled model: {e}")
        
        # Now load the state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_acc = checkpoint['best_val_acc']
        
        # Verify config matches
        if 'training_config' in checkpoint:
            assert self.training_config == checkpoint['training_config'], "Checkpoint config doesn't match current config"
        
        if self.rank == 0:
            print(f"Resumed from epoch {self.epoch}, global step {self.global_step}")      
        
    @torch.inference_mode()
    def evaluate(self) -> Dict[str, float]:
        if self.rank == 0:
            print(f"Starting evaluation...")
        
        self.model.eval()
        total_loss = 0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        if dist.is_initialized():
            self.val_loader.sampler.set_epoch(self.epoch)
        
        # Add progress bar that only shows on rank 0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=self.amp_dtype):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted_top5 = outputs.topk(5, dim=1)
            predicted_top1 = predicted_top5[:, 0]
            total += labels.size(0)
            correct_top1 += predicted_top1.eq(labels).sum().item()
            correct_top5 += (predicted_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        val_loss = total_loss / len(self.val_loader)
        val_acc_top1 = 100. * correct_top1 / total
        val_acc_top5 = 100. * correct_top5 / total
        
        self.model.train()
        return val_loss, val_acc_top1, val_acc_top5

    @torch.inference_mode()
    def compute_batch_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute accuracy for current batch."""
        if labels.dim() > 1:
            labels = labels.argmax(-1)
        _, predicted = outputs.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum()
        return (100. * correct / total).item()
    
    def train_epoch(self) -> None:
        """Train for one epoch."""
        self.model.train()
        global_batch_size = int(self.training_config.batch_size)
        
        train_loss_accum = 0.0
        train_acc_accum = 0.0
        num_batches_accum = 0

        end_time = time.time()

        if dist.is_initialized():
            self.train_loader.sampler.set_epoch(self.epoch)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward
            with autocast(device_type='cuda', dtype=self.amp_dtype):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Calculate metrics before backward pass
            with torch.inference_mode():
                loss_item = loss.detach().item()
                
                # Check for NaN loss
                if math.isnan(loss_item):
                    print(f"Rank {self.rank}: NaN loss encountered at epoch {self.epoch}, global step {self.global_step}. Stopping training.")
                    self.close()
                    raise ValueError("NaN Loss encountered")
                    
                run_acc = self.compute_batch_accuracy(outputs, labels)
                del outputs
                
                # Accumulate metrics
                train_loss_accum += loss_item
                train_acc_accum += run_acc
                num_batches_accum += 1

            # Backward pass
            if self.use_scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.training_config.clip_norm > 0.0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.training_config.clip_norm
                    )
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            
            
            if self.rank == 0:
                self.wandb_logger.log({
                    'step': self.global_step, 
                    'lr': max(self.scheduler.get_last_lr()),
                    'running_loss': loss_item, 
                })

                if self.global_step % self.print_freq == 0:
                    used_time = time.time() - end_time
                    end_time = time.time()

                    print("Epoch: [{0}][{1}/{2}]\t"
                          "Used Time {used_time:.3f}\t"
                          "Loss {loss_item:.4f}\t"
                          "Prec@1 {run_acc:.3f}".format(
                              self.epoch,
                              batch_idx,
                              self.steps_per_epoch,
                              used_time=used_time,
                              loss_item=loss_item,
                              run_acc=run_acc,
                            )
                        )
            
            self.global_step += 1

        # Eval and save checkpoint
        with torch.inference_mode():
            # Evaluate validation and log
            avg_train_loss = train_loss_accum / num_batches_accum
            avg_train_acc = train_acc_accum / num_batches_accum
            val_loss, val_acc_top1, val_acc_top5 = self.evaluate()
            
            if dist.is_initialized():
                world_size = dist.get_world_size()
                metrics = torch.tensor([avg_train_loss, avg_train_acc, val_loss, val_acc_top1, val_acc_top5], device=self.device)
                dist.barrier()
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                avg_train_loss, avg_train_acc, val_loss, val_acc_top1, val_acc_top5 = metrics.tolist()

            if self.rank == 0:
                self.wandb_logger.log({'epoch': self.epoch,
                                       'step': self.global_step, 
                                       'tr_loss': avg_train_loss / world_size, 
                                       'tr_acc': avg_train_acc / world_size,
                                       'val_loss': val_loss / world_size,
                                       'val_acc_top1': val_acc_top1 / world_size,
                                       'val_acc_top5': val_acc_top5 / world_size,
                                       }
                                    )

                # self.log_metrics(metrics)
                if val_acc_top1 > self.best_val_acc:
                    self.best_val_acc = val_acc_top1
                    self.save_checkpoint(is_best=True)
        
        # Save last checkpoint
        if self.rank == 0:
            self.save_checkpoint(is_best=False)
        
        torch.cuda.empty_cache()
            
    def train(self) -> None:
        """Train for the specified number of epochs from config. Any exception is allowed to propagate so that the full traceback is visible."""
        num_epochs = self.training_config.num_epochs

        # Print compilation status
        if self.use_compile:
            print(f"Rank {self.rank}: Training with torch.compile enabled")
            if hasattr(self.model, 'module'):
                print(f"Rank {self.rank}: Model is wrapped with DDP")

        while self.epoch < num_epochs:
            self.train_epoch()
            self.epoch += 1

            # Synchronize across processes after each epoch
            if dist.is_initialized():
                dist.barrier()
        self.close()
    
    def close(self):
        if self.rank == 0:
            self.wandb_logger.finish()
