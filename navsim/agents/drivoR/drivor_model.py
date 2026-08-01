from typing import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .lora import (
    inject_lora_into_mlp,
    lora_activation,
    set_lora_enabled,
    set_lora_trainable,
)
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)


class LambdaScheduler:
    def __init__(self, gamma=10.0):
        self.gamma = gamma

    def __call__(self, progress: float) -> float:
        return 2.0 / (1.0 + math.exp(-self.gamma * progress)) - 1.0


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class DomainClassifier(nn.Module):
    def __init__(self, d_in, hidden=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat, lambd: float):
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        elif feat.dim() != 2:
            raise ValueError(f"Expected domain features with dim 2 or 3, got {feat.shape}")
        feat = grad_reverse(feat, lambd)
        return self.classifier(feat).squeeze(-1)


class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            config_image_backbone["target_lora_rank"] = int(config.get("proposal_lora_rank", 0))
            config_image_backbone["target_lora_alpha"] = float(config.get("proposal_lora_alpha", 16.0))
            config_image_backbone["target_lora_enabled"] = bool(config.get("proposal_lora_enabled", False))
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            config_lidar_backbone["target_lora_rank"] = int(config.get("proposal_lora_rank", 0))
            config_lidar_backbone["target_lora_alpha"] = float(config.get("proposal_lora_alpha", 16.0))
            config_lidar_backbone["target_lora_enabled"] = bool(config.get("proposal_lora_enabled", False))
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )
        scorer_lora_rank = int(config.get("scorer_lora_rank", 0))
        if scorer_lora_rank > 0:
            inject_lora_into_mlp(
                self.pos_embed,
                rank=scorer_lora_rank,
                alpha=float(config.get("scorer_lora_alpha", 16.0)),
                dropout=float(config.get("scorer_lora_dropout", 0.0)),
                enabled=bool(config.get("scorer_lora_enabled", False)),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])
        proposal_lora_rank = int(config.get("proposal_lora_rank", 0))
        if proposal_lora_rank > 0:
            # Earlier heads are auxiliary only; the deployed proposals use the final head.
            inject_lora_into_mlp(
                self.traj_head[-1].mlp,
                rank=proposal_lora_rank,
                alpha=float(config.get("proposal_lora_alpha", 16.0)),
                dropout=float(config.get("proposal_lora_dropout", 0.0)),
                enabled=bool(config.get("proposal_lora_enabled", False)),
            )

        # scorer
        self.scorer = Scorer(config)
        self.domain_alignment = bool(config.get("domain_alignment", False))
        if self.domain_alignment:
            hidden = int(config.get("domain_classifier_hidden", 512))
            self.domain_classifier = DomainClassifier(config.tf_d_model, hidden=hidden)
            self.lambda_scheduler = LambdaScheduler(gamma=10.0)
            self.domain_alignment_progress = 0.0

        self.b2d=config.b2d


    def set_lora_enabled(self, proposal: bool, scorer: bool) -> None:
        """Select source or target adapters independently at inference time."""
        if proposal and int(self._config.get("proposal_lora_rank", 0)) <= 0:
            raise ValueError("proposal_lora_enabled requires proposal_lora_rank > 0.")
        if scorer and int(self._config.get("scorer_lora_rank", 0)) <= 0:
            raise ValueError("scorer_lora_enabled requires scorer_lora_rank > 0.")

        self._config.proposal_lora_enabled = proposal
        self._config.scorer_lora_enabled = scorer
        for backbone_name in ("image_backbone", "lidar_backbone"):
            if hasattr(self, backbone_name):
                getattr(self, backbone_name).set_target_lora_enabled(proposal)
        set_lora_enabled(self.trajectory_decoder, proposal)
        set_lora_enabled(self.traj_head[-1], proposal)
        set_lora_enabled(self.pos_embed, scorer)
        set_lora_enabled(self.scorer_attention, scorer)
        set_lora_enabled(self.scorer.pred_score["ego_progress"], scorer)

    def configure_lora_finetuning(self) -> None:
        """Freeze source weights and expose only target adapters and alignment modules."""
        self.requires_grad_(False)
        self.set_lora_enabled(
            bool(self._config.get("proposal_lora_enabled", False)),
            bool(self._config.get("scorer_lora_enabled", False)),
        )
        set_lora_trainable(
            self.trajectory_decoder,
            bool(self._config.get("proposal_lora_enabled", False)),
        )
        set_lora_trainable(
            self.traj_head[-1],
            bool(self._config.get("proposal_lora_enabled", False)),
        )
        set_lora_trainable(
            self.pos_embed,
            bool(self._config.get("scorer_lora_enabled", False)),
        )
        set_lora_trainable(
            self.scorer_attention,
            bool(self._config.get("scorer_lora_enabled", False)),
        )
        set_lora_trainable(
            self.scorer.pred_score["ego_progress"],
            bool(self._config.get("scorer_lora_enabled", False)),
        )

        if bool(self._config.get("backbone_lora_trainable", False)):
            for backbone_name in ("image_backbone", "lidar_backbone"):
                if hasattr(self, backbone_name):
                    getattr(self, backbone_name).set_lora_trainable(True)

        if self.domain_alignment:
            self.domain_classifier.requires_grad_(
                bool(self._config.get("proposal_lora_enabled", False))
            )


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]

        output={}


        scene_features = []
        # image features
        if self.num_cams > 0:
            
            if "image" in features :
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

            if self.domain_alignment and "rendered_camera_feature" in features and "domain_alignment_mask" in features:
                domain_mask = features["domain_alignment_mask"].bool().flatten()
                if torch.any(domain_mask):
                    rendered_img = features["rendered_camera_feature"][domain_mask]
                    domain_batch_size = rendered_img.shape[0]
                    rendered_scene_tokens = self.scene_embeds.repeat(domain_batch_size, 1, 1, 1)
                    rendered_image_scene_tokens = self.image_backbone(rendered_img, rendered_scene_tokens)
                    real_image_scene_tokens = image_scene_tokens[domain_mask]
                    output["rendered_image_scene_tokens"] = rendered_image_scene_tokens
                    output["real_image_scene_tokens"] = real_image_scene_tokens
                    domain_features = torch.cat([rendered_image_scene_tokens, real_image_scene_tokens], dim=0)
                    domain_labels = torch.cat(
                        [
                            domain_features.new_zeros(domain_batch_size),
                            domain_features.new_ones(domain_batch_size),
                        ],
                        dim=0,
                    )
                    output["domain_logits"] = self.domain_classifier(
                        domain_features,
                        lambd=self.lambda_scheduler(self.domain_alignment_progress),
                    )
                    output["domain_labels"] = domain_labels

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)
        
        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]
        

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        # scoring
        B,N,_,_=proposals.shape

        scorer_scene_features = scene_features
        scorer_ego_token = ego_token
        if bool(self._config.get("lora_finetune", False)) and bool(
            self._config.get("detach_scorer_inputs", True)
        ):
            # RFS supervision adapts only the scorer head, not shared proposal inputs.
            scorer_scene_features = scorer_scene_features.detach()
            scorer_ego_token = scorer_ego_token.detach()
        detached_proposals = proposals.reshape(B, N, -1).detach()
        rfs_feature = None
        if bool(self._config.get("scorer_lora_enabled", False)):
            with lora_activation(self.pos_embed, False), lora_activation(self.scorer_attention, False):
                source_embedded_traj = self.pos_embed(detached_proposals)
                tr_out = self.scorer_attention(source_embedded_traj, scorer_scene_features)
            rfs_embedded_traj = self.pos_embed(detached_proposals)
            rfs_feature = self.scorer_attention(rfs_embedded_traj, scorer_scene_features)
            rfs_feature = rfs_feature + scorer_ego_token
        else:
            embedded_traj = self.pos_embed(detached_proposals)
            tr_out = self.scorer_attention(embedded_traj, scorer_scene_features)
        tr_out = tr_out + scorer_ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out, rfs_feature=rfs_feature)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        pdm_score = pred_logit["ego_progress"].new_zeros(pred_logit["ego_progress"].shape)
        for weight, key in (
            (self._config.noc, "no_at_fault_collisions"),
            (self._config.dac, "drivable_area_compliance"),
            (self._config.ddc, "driving_direction_compliance"),
        ):
            if weight:
                pdm_score = pdm_score + weight * F.logsigmoid(pred_logit[key])

        weighted_score_terms = []
        for weight, key in (
            (self._config.ttc, "time_to_collision_within_bound"),
            (self._config.ep, "ego_progress"),
            (self._config.comfort, "comfort"),
        ):
            if weight:
                weight_log = pred_logit[key].new_tensor(float(weight)).log()
                weighted_score_terms.append(weight_log + F.logsigmoid(pred_logit[key]))
        pdm_score = pdm_score + torch.logsumexp(torch.stack(weighted_score_terms, dim=0), dim=0)

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output
