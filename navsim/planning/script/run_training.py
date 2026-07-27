import os
import random
from typing import Tuple
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch


from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


class DrivoRDomainDataset(torch.utils.data.Dataset):
    """Normalizes cache samples so Songdo rendered-only and NAVSIM real/rendered caches collate together."""

    def __init__(self, dataset: torch.utils.data.Dataset, domain_alignment: bool):
        self.dataset = dataset
        self.domain_alignment = domain_alignment

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        features, targets = self.dataset[idx]
        model_image = features["image"] if "image" in features else features["camera_feature"]

        normalized_features = {
            "camera_feature": model_image,
            "ego_status": features["ego_status"],
            "domain_alignment_mask": torch.tensor(self.domain_alignment, dtype=torch.bool),
        }
        if self.domain_alignment:
            if "rendered_camera_feature" not in features:
                raise KeyError("NAVSIM alignment cache sample is missing 'rendered_camera_feature'")
            normalized_features["rendered_camera_feature"] = features["rendered_camera_feature"]
        else:
            normalized_features["rendered_camera_feature"] = torch.zeros_like(model_image)

        normalized_targets = {
            "trajectory": targets["trajectory"],
            "token": targets["token"],
        }
        return normalized_features, normalized_targets


def _cache_log_names(cache_path: str, manifest_name: str, manifest_key: str) -> list[str]:
    manifest_path = Path(cache_path) / manifest_name
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_key in manifest:
            return manifest[manifest_key]
    return sorted(log_name.name for log_name in Path(cache_path).iterdir() if log_name.is_dir())


class FixedRatioBatchSampler(torch.utils.data.Sampler):
    """Yields fixed-composition batches from a Songdo/NAVSIM ConcatDataset."""

    def __init__(
        self,
        songdo_len: int,
        navsim_len: int,
        songdo_batch_size: int,
        navsim_batch_size: int,
        seed: int,
    ):
        if songdo_len <= 0:
            raise ValueError("Songdo dataset must be non-empty for fixed-ratio batching")
        if navsim_len <= 0:
            raise ValueError("NAVSIM dataset must be non-empty for fixed-ratio batching")
        if songdo_batch_size <= 0 or navsim_batch_size <= 0:
            raise ValueError("Both Songdo and NAVSIM batch sizes must be positive")

        self.songdo_len = songdo_len
        self.navsim_len = navsim_len
        self.songdo_batch_size = songdo_batch_size
        self.navsim_batch_size = navsim_batch_size
        self.batch_size = songdo_batch_size + navsim_batch_size
        self.seed = seed
        self.epoch = 0
        self.batches_per_epoch = songdo_len // songdo_batch_size
        if self.batches_per_epoch <= 0:
            raise ValueError("Fixed-ratio batching would produce zero batches; reduce songdo_batch_size")

    def __len__(self) -> int:
        _, world_size = self._distributed_info()
        return (self.batches_per_epoch + world_size - 1) // world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        songdo_indices = list(range(self.songdo_len))
        navsim_indices = list(range(self.songdo_len, self.songdo_len + self.navsim_len))
        rng.shuffle(songdo_indices)
        rng.shuffle(navsim_indices)

        songdo_cursor = 0
        navsim_cursor = 0
        rank, world_size = self._distributed_info()

        total_batches = len(self) * world_size
        for batch_idx in range(total_batches):
            songdo_batch, songdo_cursor = self._draw(
                songdo_indices,
                songdo_cursor,
                self.songdo_batch_size,
                rng,
            )
            navsim_batch, navsim_cursor = self._draw(
                navsim_indices,
                navsim_cursor,
                self.navsim_batch_size,
                rng,
            )
            batch = songdo_batch + navsim_batch
            rng.shuffle(batch)
            if batch_idx % world_size == rank:
                yield batch

    @staticmethod
    def _distributed_info() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    @staticmethod
    def _draw(indices: list[int], cursor: int, count: int, rng: random.Random) -> tuple[list[int], int]:
        batch = []
        while len(batch) < count:
            remaining = len(indices) - cursor
            take = min(count - len(batch), remaining)
            batch.extend(indices[cursor : cursor + take])
            cursor += take
            if len(batch) < count:
                rng.shuffle(indices)
                cursor = 0
        return batch, cursor


def _maybe_apply_cache_overrides(cfg: DictConfig) -> None:
    if not cfg.use_cache_without_dataset:
        return

    manifest_path = Path(cfg.cache_path) / ".songdo_drivor_manifest.json"
    if not manifest_path.is_file():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "image_size" in manifest:
        cfg.agent.config.image_size = manifest["image_size"]
        logger.info("Using Songdo DrivoR cached image_size override: %s", manifest["image_size"])
    songdo_split_path = cfg.get("songdo_split_path")
    if songdo_split_path:
        split_payload = json.loads(Path(songdo_split_path).read_text(encoding="utf-8"))
        cfg.train_logs = [Path(session_filename).stem for session_filename in split_payload["train"]]
        cfg.val_logs = [Path(session_filename).stem for session_filename in split_payload["test"]]
        logger.info("Using Songdo DrivoR split override from %s", songdo_split_path)
        return
    if "train_logs" in manifest:
        cfg.train_logs = manifest["train_logs"]
        logger.info("Using Songdo DrivoR cached train_logs override: %s", manifest["train_logs"])
    if "val_logs" in manifest:
        cfg.val_logs = manifest["val_logs"]
        logger.info("Using Songdo DrivoR cached val_logs override: %s", manifest["val_logs"])

def dist_ready():
    return dist.is_available() and dist.is_initialized()

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    
    print("Train without caching....")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    _maybe_apply_cache_overrides(cfg)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    train_batch_sampler = None
    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
        train_data = DrivoRDomainDataset(train_data, domain_alignment=False)
        val_data = DrivoRDomainDataset(val_data, domain_alignment=False)
        scheduler_dataset_size = len(train_data)

        navsim_cache_path = cfg.get("navsim_cache_path")
        if navsim_cache_path:
            navsim_train_logs = _cache_log_names(
                navsim_cache_path,
                ".navsim_drivor_alignment_manifest.json",
                "train_logs",
            )
            navsim_data = CacheOnlyDataset(
                cache_path=navsim_cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=navsim_train_logs,
            )
            navsim_data = DrivoRDomainDataset(navsim_data, domain_alignment=True)
            navsim_batch_size = cfg.get("navsim_batch_size")
            if navsim_batch_size is None:
                raise ValueError("navsim_batch_size must be set when navsim_cache_path is provided")

            total_batch_size = int(cfg.dataloader.params.batch_size)
            navsim_batch_size = int(navsim_batch_size)
            songdo_batch_size = cfg.get("songdo_batch_size")
            if songdo_batch_size is None:
                songdo_batch_size = total_batch_size - navsim_batch_size
            else:
                songdo_batch_size = int(songdo_batch_size)
                if songdo_batch_size + navsim_batch_size != total_batch_size:
                    raise ValueError(
                        "songdo_batch_size + navsim_batch_size must equal dataloader.params.batch_size"
                    )
            songdo_data = train_data
            train_data = ConcatDataset([train_data, navsim_data])
            train_batch_sampler = FixedRatioBatchSampler(
                songdo_len=len(songdo_data),
                navsim_len=len(navsim_data),
                songdo_batch_size=songdo_batch_size,
                navsim_batch_size=navsim_batch_size,
                seed=int(cfg.seed),
            )
            scheduler_dataset_size = len(train_batch_sampler) * train_batch_sampler.batch_size
            logger.info(
                "Using NAVSIM alignment cache %s with fixed batches: %d Songdo + %d NAVSIM",
                navsim_cache_path,
                songdo_batch_size,
                navsim_batch_size,
            )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)
        scheduler_dataset_size = len(train_data)

    if getattr(agent, "scheduler_args", None) is not None:
        agent.scheduler_args.dataset_size = scheduler_dataset_size
        logger.info("Using training dataset size for scheduler: %d", scheduler_dataset_size)

    logger.info("Building Datasets")
    trainer_params = dict(cfg.trainer.params)
    if train_batch_sampler is None:
        train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True,drop_last=True)
    else:
        train_dataloader_params = dict(cfg.dataloader.params)
        train_dataloader_params.pop("batch_size")
        train_dataloader = DataLoader(train_data, **train_dataloader_params, batch_sampler=train_batch_sampler)
        logger.info("Num fixed-ratio training batches: %d", len(train_batch_sampler))
        # FixedRatioBatchSampler already partitions batches across distributed ranks.
        trainer_params["use_distributed_sampler"] = False
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False,drop_last=True)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    # automatically resume training
    # find latest ckpt
    import glob
    def find_latest_checkpoint(search_pattern):
        # List all files matching the pattern
        list_of_files = glob.glob(search_pattern, recursive=True)
        # Find the file with the latest modification time
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file


    if cfg.train_ckpt_path is None:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = "/".join(str(cfg.output_dir).split("/")[:-1]) + "/*/**/checkpoints/" + '*.ckpt'
        print("/".join(str(cfg.output_dir).split("/")[:-1]))
        print("search_pattern ", search_pattern)
        cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
        print("cfg.train_ckpt_path ", cfg.train_ckpt_path)
    trainer = pl.Trainer(
        **trainer_params,
        callbacks=agent.get_training_callbacks(),
        logger=WandbLogger(
            project="drivor",
            name=cfg.experiment_name,
            save_dir=cfg.output_dir,
        ),
    )

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            val_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()
