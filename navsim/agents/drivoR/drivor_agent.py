from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from .drivor_model import DrivoRModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from .drivor_features import DrivoRTargetBuilder
from .drivor_features import DrivoRFeatureBuilder
import sys
from omegaconf import OmegaConf
import math


def _slice_batch(data, mask: torch.Tensor):
    if torch.is_tensor(data):
        if data.ndim > 0 and data.shape[0] == mask.shape[0]:
            return data[mask]
        return data
    if isinstance(data, dict):
        return {key: _slice_batch(value, mask) for key, value in data.items()}
    if isinstance(data, list):
        if all(torch.is_tensor(value) or isinstance(value, (dict, list, tuple)) for value in data):
            return [_slice_batch(value, mask) for value in data]
        if len(data) == mask.shape[0]:
            indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            return [data[index] for index in indices]
        return data
    if isinstance(data, tuple):
        if all(torch.is_tensor(value) or isinstance(value, (dict, list, tuple)) for value in data):
            return tuple(_slice_batch(value, mask) for value in data)
        if len(data) == mask.shape[0]:
            indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            return tuple(data[index] for index in indices)
        return data
    return data

class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - train {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - val {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(self, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k,v in metrics.items():
            if "train/" in k:
                train_metrics[k]=v
            elif "val/" in k:
                val_metrics[k]=v
            else:
                other_metrics[k]=v
        print(f"\n###########  Epoch {trainer.current_epoch} ##########")
        for k,v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")

class DrivoRAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
    ):
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.use_metric_cache = bool(config.use_metric_cache)
        self.loss = loss


        cache_data=False

        if not cache_data:
            self._drivor_model = DrivoRModel(config)

        if not cache_data and self._checkpoint_path == "": # only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d
            self.ray = False

            if self.use_metric_cache:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                from .score_module.compute_navsim_score import get_scores

                self.ray = True
                self.worker = RayDistributedNoTorch(threads_per_node=8)
                self.worker_map = worker_map

                metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
                try:
                    metric_cache_synthetic_0 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-0"))
                    metric_cache_synthetic_1 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-1"))
                    metric_cache_synthetic_2 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-2"))
                    metric_cache_synthetic_3 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-3"))
                    metric_cache_synthetic_4 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-4"))

                    self.train_metric_cache_paths_synthetic = metric_cache_synthetic_0.metric_cache_paths
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_0.metric_cache_paths)
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_1.metric_cache_paths)
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_2.metric_cache_paths)
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_3.metric_cache_paths)
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_4.metric_cache_paths)
                except Exception:
                    self.train_metric_cache_paths_synthetic = None

                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths
                self.get_scores = get_scores
            


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            self.load_state_dict({k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()})

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        # return SensorConfig(
        #     cam_f0=[3],
        #     cam_l0=[3],
        #     cam_l1=[],
        #     cam_l2=[],
        #     cam_r0=[3],
        #     cam_r1=[],
        #     cam_r2=[],
        #     cam_b0=[3],
        #     lidar_pc=[],
        # )
        return SensorConfig(
            cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
            cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
            cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
            cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
            cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
            cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
            cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
            cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )
    
    def get_target_builders(self) :
        return [DrivoRTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [DrivoRFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._drivor_model(features)

    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
            metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
        else:
            metric_cache_paths = self.test_metric_cache_paths
            metric_cache_paths_synthetic = self.test_metric_cache_paths_synthetic

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        
        data_points = [
            {
                "token": metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token],
                "poses": poses,
                "test": test
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        scoring_function = self.compute_score if self.use_metric_cache else None
        domain_mask = features.get("domain_alignment_mask")
        if domain_mask is None:
            return self.loss(targets, pred, self._config, scoring_function)

        domain_mask = domain_mask.bool().flatten()
        trajectory_mask = ~domain_mask
        total_loss = pred["trajectory"].new_tensor(0.0)
        loss_dict = {}

        if torch.any(trajectory_mask):
            trajectory_targets = _slice_batch(targets, trajectory_mask)
            trajectory_pred = _slice_batch(pred, trajectory_mask)
            trajectory_loss = self.loss(trajectory_targets, trajectory_pred, self._config, scoring_function)
            if isinstance(trajectory_loss, dict):
                loss_dict.update(trajectory_loss)
                total_loss = total_loss + trajectory_loss["loss"]
            else:
                loss_dict["trajectory_loss"] = trajectory_loss
                total_loss = total_loss + trajectory_loss

        if torch.any(domain_mask):
            if "domain_logits" not in pred or "domain_labels" not in pred:
                raise KeyError("Domain alignment batch is missing domain classifier outputs")
            domain_loss = F.binary_cross_entropy_with_logits(pred["domain_logits"], pred["domain_labels"])
            domain_weight = float(self._config.get("domain_alignment_weight", 1.0))
            domain_pred = pred["domain_logits"].detach() >= 0
            domain_acc = torch.mean((domain_pred == pred["domain_labels"].bool()).float())
            total_loss = total_loss + domain_weight * domain_loss
            loss_dict["domain_loss"] = domain_loss
            loss_dict["domain_acc"] = domain_acc
            loss_dict["domain_weighted_loss"] = domain_weight * domain_loss

        loss_dict["loss"] = total_loss
        return loss_dict

    def get_optimizers(self):

        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.Adam(self._drivor_model.parameters(), lr=lr)
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.AdamW(self._drivor_model.parameters(), lr=lr)
        else:
            raise NotImplementedError

        if self.scheduler_args is not None:

            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)

            # classic cosine
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=T_max, 
            #     eta_min=0.0, last_epoch=-1
            # )

            # Ramp + cosine
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_cosine, 
                eta_min=0.0, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )           

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):
        if self.use_metric_cache:
            checkpoint_cb_best = ModelCheckpoint(save_top_k=1,
                                            monitor='val/score',
                                            filename='best-{epoch}-{step}',
                                            mode="max"
                                            )
        else:
            checkpoint_cb_best = ModelCheckpoint(save_top_k=1,
                                            monitor='val/chosen_fde',
                                            filename='best-{epoch}-{step}',
                                            mode="min"
                                            )
        
        checkpoint_cb = ModelCheckpoint(save_last=True)

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)
        device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)
        
        if self.progress_bar:
            return [checkpoint_cb_best, checkpoint_cb, lr_monitor, device_stats_monitor]
        else:
            progress_bar = LitProgressBar()
            return [checkpoint_cb_best, checkpoint_cb, progress_bar, lr_monitor, device_stats_monitor]
