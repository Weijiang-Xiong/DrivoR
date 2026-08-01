import torch
import torch.nn as nn
from omegaconf import OmegaConf

from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import LoRA_ViT_timm, timm_ViT
from navsim.agents.drivoR.lora import (
    LoRALinear,
    LoRAMultiheadAttention,
    iter_lora_modules,
    set_lora_enabled,
)
from navsim.agents.drivoR.score_module.scorer import Scorer


def _model_config():
    return OmegaConf.create(
        {
            "num_poses": 2,
            "proposal_num": 3,
            "tf_d_model": 8,
            "tf_d_ffn": 16,
            "one_token_per_traj": True,
            "full_history_status": False,
            "ref_num": 1,
            "scorer_ref_num": 1,
            "refiner_num_heads": 1,
            "refiner_ls_values": 0.0,
            "cam_f0": [],
            "cam_l0": [],
            "cam_l1": [],
            "cam_l2": [],
            "cam_r0": [],
            "cam_r1": [],
            "cam_r2": [],
            "cam_b0": [],
            "lidar_pc": [],
            "b2d": False,
            "double_score": False,
            "agent_pred": False,
            "area_pred": False,
            "bev_map": False,
            "bev_agent": False,
            "domain_alignment": True,
            "domain_classifier_hidden": 4,
            "lora_finetune": True,
            "proposal_lora_enabled": True,
            "proposal_lora_rank": 2,
            "proposal_lora_alpha": 2.0,
            "proposal_lora_dropout": 0.0,
            "scorer_lora_enabled": True,
            "scorer_lora_rank": 2,
            "scorer_lora_alpha": 2.0,
            "scorer_lora_dropout": 0.0,
            "backbone_lora_trainable": False,
            "detach_scorer_inputs": True,
            "noc": 0.0,
            "dac": 0.0,
            "ddc": 0.0,
            "ttc": 0.0,
            "ep": 1.0,
            "comfort": 0.0,
        }
    )


class _DummyLoRABackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.source_lora = nn.Parameter(torch.ones(1))
        self.target_lora = nn.Parameter(torch.ones(1))
        self.target_enabled = True

    def set_lora_trainable(self, trainable: bool) -> None:
        self.target_lora.requires_grad_(trainable)

    def set_target_lora_enabled(self, enabled: bool) -> None:
        self.target_enabled = enabled

    def forward(self, images: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_cameras, num_tokens, channels = scene_tokens.shape
        image_scale = images.mean(dim=(2, 3, 4), keepdim=False).reshape(batch_size, num_cameras, 1, 1)
        features = scene_tokens + self.source_lora * image_scale
        if self.target_enabled:
            features = features + self.target_lora * image_scale
        return features.reshape(batch_size, num_cameras * num_tokens, channels)


def test_zero_initialized_lora_preserves_linear_output() -> None:
    torch.manual_seed(0)
    source = nn.Linear(5, 3)
    adapter = LoRALinear.from_linear(source, rank=2, alpha=4, dropout=0, enabled=True)
    inputs = torch.randn(7, 5)

    torch.testing.assert_close(adapter(inputs), source(inputs), rtol=0, atol=0)


def test_rfs_lora_changes_only_ego_progress_head() -> None:
    torch.manual_seed(1)
    config = _model_config()
    scorer = Scorer(config)
    proposals = torch.zeros(2, config.proposal_num, config.num_poses, 3)
    proposal_features = torch.ones(2, config.proposal_num, config.tf_d_model)

    disabled_logits = scorer(proposals, proposal_features)[0]
    for adapter in iter_lora_modules(scorer.pred_score["ego_progress"]):
        with torch.no_grad():
            adapter.lora_A.fill_(0.25)
            adapter.lora_B.fill_(0.25)
    enabled_logits = scorer(proposals, proposal_features)[0]

    assert not torch.equal(enabled_logits["ego_progress"], disabled_logits["ego_progress"])
    for key in scorer.pred_score:
        if key != "ego_progress":
            torch.testing.assert_close(enabled_logits[key], disabled_logits[key], rtol=0, atol=0)

    set_lora_enabled(scorer.pred_score["ego_progress"], False)
    restored_logits = scorer(proposals, proposal_features)[0]
    torch.testing.assert_close(restored_logits["ego_progress"], disabled_logits["ego_progress"])


def test_attention_lora_preserves_base_keys_and_zero_initialized_output() -> None:
    torch.manual_seed(2)
    source = nn.MultiheadAttention(8, 2, batch_first=True)
    base_keys = set(source.state_dict())
    adapter = LoRAMultiheadAttention.from_attention(source, rank=2, alpha=2, enabled=True)
    inputs = torch.randn(2, 3, 8)

    torch.testing.assert_close(adapter(inputs, inputs, inputs)[0], source(inputs, inputs, inputs)[0])
    assert base_keys.issubset(adapter.state_dict())

    with torch.no_grad():
        adapter.lora_B_q.fill_(0.25)
        adapter.lora_B_v.fill_(0.25)
    assert not torch.equal(adapter(inputs, inputs, inputs)[0], source(inputs, inputs, inputs)[0])


def test_finetuning_exposes_only_requested_adapters_and_domain_classifier() -> None:
    model = DrivoRModel(_model_config())
    model.configure_lora_finetuning()

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert any(name.startswith("traj_head.") and ".lora_" in name for name in trainable_names)
    assert all(
        not name.startswith("traj_head.") or name.startswith("traj_head.1.")
        for name in trainable_names
    )
    assert any(
        name.startswith("scorer.pred_score.ego_progress.") and ".lora_" in name
        for name in trainable_names
    )
    assert any(name.startswith("domain_classifier.") for name in trainable_names)
    assert all(
        (name.startswith("trajectory_decoder.") and ".lora_" in name)
        or (name.startswith("traj_head.") and ".lora_" in name)
        or (name.startswith("pos_embed.") and ".lora_" in name)
        or (name.startswith("scorer_attention.") and ".lora_" in name)
        or (name.startswith("scorer.pred_score.ego_progress.") and ".lora_" in name)
        or name.startswith("domain_classifier.")
        for name in trainable_names
    )


def test_backbone_lora_can_be_reenabled_after_model_freeze() -> None:
    vit = timm_ViT(
        img_size=16,
        patch_size=4,
        in_chans=3,
        num_classes=0,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
    )
    adapter = LoRA_ViT_timm(vit, r=2, target_r=3, target_alpha=6, target_enabled=True)
    adapter.requires_grad_(False)
    adapter.set_target_lora_trainable(True)

    trainable_names = {
        name for name, parameter in adapter.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all("lora_A_target_" in name or "lora_B_target_" in name for name in trainable_names)

    qkv = adapter.lora_vit.blocks[0].attn.qkv
    inputs = torch.randn(2, 4, 8)
    adapter.set_target_lora_enabled(False)
    source_output = qkv(inputs)
    with torch.no_grad():
        qkv.lora_B_target_q.weight.fill_(0.25)
        qkv.lora_B_target_v.weight.fill_(0.25)
    adapter.set_target_lora_enabled(True)
    assert not torch.equal(qkv(inputs), source_output)
    adapter.set_target_lora_enabled(False)
    torch.testing.assert_close(qkv(inputs), source_output, rtol=0, atol=0)


def test_score_gradient_isolated_from_proposal_and_shared_backbone() -> None:
    config = _model_config()
    config.backbone_lora_trainable = True
    model = DrivoRModel(config)
    model.num_cams = 1
    model.image_backbone = _DummyLoRABackbone()
    model.scene_embeds = nn.Parameter(torch.randn(1, 1, 2, config.tf_d_model))
    model.configure_lora_finetuning()
    model.eval()

    features = {
        "camera_feature": torch.randn(2, 1, 3, 2, 2),
        "ego_status": torch.randn(2, 1, 11),
    }
    prediction = model(features)
    prediction["pred_logit"]["ego_progress"].sum().backward()
    score_grad_names = {
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    assert score_grad_names
    assert all(
        name.startswith("pos_embed.")
        or name.startswith("scorer_attention.")
        or name.startswith("scorer.pred_score.ego_progress.")
        for name in score_grad_names
    )

    model.zero_grad(set_to_none=True)
    model(features)["proposals"].sum().backward()
    proposal_grad_names = {
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    assert any(name.startswith("traj_head.1.") for name in proposal_grad_names)
    assert any(name.startswith("trajectory_decoder.") for name in proposal_grad_names)
    assert "image_backbone.target_lora" in proposal_grad_names
    assert "image_backbone.source_lora" not in proposal_grad_names
    assert all(not name.startswith("scorer.") for name in proposal_grad_names)


def test_scorer_context_lora_changes_only_ego_progress_output() -> None:
    config = _model_config()
    model = DrivoRModel(config)
    model.num_cams = 1
    model.image_backbone = _DummyLoRABackbone()
    model.scene_embeds = nn.Parameter(torch.randn(1, 1, 2, config.tf_d_model))
    model.configure_lora_finetuning()
    model.eval()
    features = {
        "camera_feature": torch.randn(2, 1, 3, 2, 2),
        "ego_status": torch.randn(2, 1, 11),
    }

    model.set_lora_enabled(proposal=True, scorer=False)
    source_logits = model(features)["pred_logit"]
    for module in (model.pos_embed, model.scorer_attention, model.scorer.pred_score["ego_progress"]):
        for adapter in iter_lora_modules(module):
            for name, parameter in adapter.named_parameters(recurse=False):
                if name.startswith("lora_B"):
                    with torch.no_grad():
                        parameter.fill_(0.25)
    model.set_lora_enabled(proposal=True, scorer=True)
    adapted_logits = model(features)["pred_logit"]

    assert not torch.equal(adapted_logits["ego_progress"], source_logits["ego_progress"])
    for key in source_logits:
        if key != "ego_progress":
            torch.testing.assert_close(adapted_logits[key], source_logits[key], rtol=0, atol=0)
