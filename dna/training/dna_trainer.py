import math
import time
import traceback

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils._pytree import tree_flatten, tree_unflatten

from typing import Dict, Any, Optional, Union, Dict
from tqdm import tqdm

from .base_trainer import BaseTrainer

class DNATrainer(BaseTrainer):
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
        print_freq: int = 50
    ):
        super().__init__(
            training_config,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            slurm_id,
            save_dir,
            device,
            rank,
            resume,
            wandb_logger,
            find_unused_parameters,
            static_graph,
            use_compile,
            print_freq
        )
    
    @torch.inference_mode()
    def _calculate_compute_ratios(self, data):
        """
        Calculate the ratio of each identity module for each selector.
        Args:
            data (dict): A dictionary containing selector statistics.
        Returns:
            dict: A dictionary with the calculated ratios for each selector.
        """
        # Initialize an empty dictionary to store the results
        result = {}
        # Iterate over each selector in the data
        for i, (selector, stats) in enumerate(data.items()):
            # Extract the identity values
            identities = {key: value for key, value in stats.items() if key.startswith('identity_')}
            # Check if there are any identity values
            if not identities:
                result[f'selector_{i}_utilization'] = 1.
                continue
            # Calculate the total sum of all identity values
            total_sum = sum(stats.values())
            id_sum = sum(identities.values())
            # Store the result in the dictionary
            result[f'selector_{i}_utilization'] = 1. - id_sum / total_sum
        return result
    
    @torch.inference_mode()
    def _get_layer_usage(self):
        """Get layer-wise usage statistics in a format compatible with metrics logging"""
        # Get the base model, handling both DDP and compiled models
        if hasattr(self.model, 'module'):
            # DDP model
            base_model = self.model.module
        else:
            # Non-DDP model
            base_model = self.model
            
        # For compiled models, we need to access the original model
        if self.use_compile and hasattr(torch, 'compile') and hasattr(base_model, '_orig_mod'):
            base_model = base_model._orig_mod
            
        # Get usage stats from the base model
        usage_list = base_model.get_usage_stats()
        
        local_usage_dict = {}
        for layer_idx, block_usage in enumerate(usage_list):
            # Use the same naming convention as in template.py
            block_usage[f"ipr"] = sum(val ** 4 for val in block_usage.values()) / (sum(val ** 2 for val in block_usage.values()) ** 2) / dist.get_world_size()
            local_usage_dict[f"selector_{layer_idx}_statistics"] = block_usage
        
        if dist.is_initialized():
            leaves, treespec = tree_flatten(local_usage_dict)
            
            leaves = torch.as_tensor(leaves, device=self.device)
            dist.all_reduce(leaves, op=dist.ReduceOp.SUM)            
            
            leaves = leaves.tolist()

            return tree_unflatten(leaves, treespec)
        else:
            return local_usage_dict
    
    def train_epoch(self) -> None:
        """Train for one epoch."""
        
        self.model.train()
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
                # print(self.model.module.router_collection.routers[0].bias.grad)
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
            
            # self.model.module.sync_bias() if hasattr(self.model, 'module') else self.model.sync_bias()
            val_loss, val_acc_top1, val_acc_top5 = self.evaluate()
            layer_usage = self._get_layer_usage()
            
            if dist.is_initialized():
                world_size = dist.get_world_size()
                metrics = torch.tensor([avg_train_loss, avg_train_acc, val_loss, val_acc_top1, val_acc_top5], device=self.device)
                dist.barrier()
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                avg_train_loss, avg_train_acc, val_loss, val_acc_top1, val_acc_top5 = metrics.tolist()

            if self.rank == 0:
                compute_usage = self._calculate_compute_ratios(layer_usage)
                self.wandb_logger.log({
                    'epoch': self.epoch,
                    'step': self.global_step, 
                    'tr_loss': avg_train_loss / world_size, 
                    'tr_acc': avg_train_acc / world_size,
                    'val_loss': val_loss / world_size,
                    'val_acc_top1': val_acc_top1 / world_size,
                    'val_acc_top5': val_acc_top5 / world_size,
                    'layer_usage': layer_usage,
                    'utilization': compute_usage,
                    'mean_utilization': sum(compute_usage.values()) / len(compute_usage)
                })
                
                print(sum(compute_usage.values()) / len(compute_usage), val_acc_top1 / world_size, val_acc_top5 / world_size)
                print(compute_usage)
                # self.log_metrics(metrics)
                if val_acc_top1 > self.best_val_acc:
                    self.best_val_acc = val_acc_top1
                    self.save_checkpoint(is_best=True)
        
        # Save last checkpoint
        if self.rank == 0:
            self.save_checkpoint(is_best=False)

        torch.cuda.empty_cache()