# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00TN1d6_ActionHeader import get_action_model, Gr00tN1d6ActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

# TODO make this configurable from yaml
multi_robot_action_heads = {
"franka": {"action_dim": 7, "NUM_ACTIONS_CHUNK": 16, "robo_info": "single arm, delta eef", "robot_mlp_id": 1}, # LIBERO
"oxe_bridge": {"action_dim": 7, "NUM_ACTIONS_CHUNK": 16, "robo_info": "single arm, delta eef", "robot_mlp_id": 2}, # OXE-Bridge
"oxe_rt1": {"action_dim": 7, "NUM_ACTIONS_CHUNK": 16, "robo_info": "single arm, delta eef", "robot_mlp_id": 3}, # OXE-RT1
"gr1": {"action_dim": 29, "NUM_ACTIONS_CHUNK": 16, "robo_info": "dual-arm dexterous hands, Joint", "robot_mlp_id": 4}, # Robocasa-GR1
"robotwin": {"action_dim": 14, "NUM_ACTIONS_CHUNK": 16, "robo_info": "arm gripper, Joint", "robot_mlp_id": 5}, # Robotwin
}

def collate_fn_extend_dim(batch, max_dim=32):

    for b in batch:
        if "action" in b.keys() and b["action"].shape[-1] < max_dim:
            b["action"] = np.concatenate([b["action"], np.zeros((b["action"].shape[0], max_dim - b["action"].shape[-1]))], axis=-1)
        if "state" in b.keys() and b["state"].shape[-1] < max_dim:
            b["state"] = np.concatenate([b["state"], np.zeros((b["state"].shape[0], max_dim - b["state"].shape[-1]))], axis=-1)
    return batch


@FRAMEWORK_REGISTRY.register("QwenGR00TN1d6_epx3_randominti")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Add a function to randomly reinitialize self.qwen_vl_interface parameters
        self._randomly_initialize_qwen_vl_interface()
        # align dims --> we should put them to config or no?
        self.config.framework.qwenvl.vl_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: Gr00tN1d6ActionHead = get_action_model(config=self.config)  # fix subsequent type references

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        

    def _randomly_initialize_qwen_vl_interface(self, seed: Optional[int] = None, std: float = 0.02) -> None:
        """Randomly re-initialize parameters of the Qwen-VL interface."""
        if seed is not None:
            torch.manual_seed(seed)

        model = self.qwen_vl_interface.model if hasattr(self.qwen_vl_interface, "model") else self.qwen_vl_interface
        total_params = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if not param.is_floating_point():
                    continue
                total_params += param.numel()
                if "bias" in name:
                    torch.nn.init.zeros_(param)
                else:
                    torch.nn.init.normal_(param, mean=0.0, std=std)

        logger.info(
            "Randomly initialized Qwen-VL interface parameters (params=%s, std=%.4f)",
            total_params,
            std,
        )


    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """

        # print(examples)
        examples = collate_fn_extend_dim(examples, max_dim=self.config.framework.action_model.action_dim)
        # print(examples)

        batch_images = [example["image"] for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        # 2. Enhance instructions (add action tokens)
        # print(len(instructions))
        instructions = self._add_robo_meta_tokens_to_instructions(examples, instructions) # TODO should only need examples
        # print(len(instructions))
        # print("----")
        # print(len(instructions))
        # print(len(batch_images))
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions_target = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)


            # align inputs for GR00T action model
            backbone_attention_mask = qwen_inputs.get("attention_mask", None)
            backbone_image_mask = self.qwen_vl_interface.get_image_input_mask(qwen_inputs["input_ids"])
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1)
                backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
            if backbone_image_mask is not None:
                backbone_image_mask = backbone_image_mask.repeat(repeated_diffusion_steps, 1)
                backbone_image_mask = backbone_image_mask.to(dtype=torch.bool)
            backbone_output = BatchFeature(
                data={
                    "backbone_features": last_hidden_repeated,
                    "backbone_attention_mask": backbone_attention_mask,
                    "image_mask": backbone_image_mask,
                }
            )
            # image mask is None for VLA input
            batch_size = actions_target_repeated.shape[0]
            embodiment_id = torch.zeros(batch_size, device=last_hidden.device, dtype=torch.long)
            action_mask = torch.ones_like(actions_target_repeated, device=last_hidden.device, dtype=last_hidden.dtype)

            action_input = BatchFeature(
                data={
                    "action": actions_target_repeated,
                    "state": state_repeated,
                    "embodiment_id": embodiment_id,
                    "action_mask": action_mask,
                }
            )

            action_outputs = self.action_model(backbone_output=backbone_output, action_input=action_input)
            action_loss = action_outputs["loss"]

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """

        if type(examples) is not list:
            examples = [examples]

        examples = collate_fn_extend_dim(examples, max_dim=self.config.framework.action_model.action_dim)
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
                
        # 2. Enhance instructions (add action tokens)
        instructions = self._add_robo_meta_tokens_to_instructions(examples, instructions) # TODO should only need examples
    
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Prepare state
        if state is not None:
            state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)

        # Build masks
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        backbone_image_mask = self.qwen_vl_interface.get_image_input_mask(qwen_inputs["input_ids"])
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        if backbone_image_mask is not None:
            backbone_image_mask = backbone_image_mask.to(dtype=torch.bool)

        # Assemble BatchFeatures for action head
        backbone_output = BatchFeature(
            data={
                "backbone_features": last_hidden,
                "backbone_attention_mask": backbone_attention_mask,
                "image_mask": backbone_image_mask,
            }
        )

        embodiment_id = torch.zeros(last_hidden.shape[0], device=last_hidden.device, dtype=torch.long)
        action_input = BatchFeature(
            data={
                "state": state,
                "embodiment_id": embodiment_id,
            }
        )

        with torch.autocast("cuda", dtype=torch.float32):
            action_outputs = self.action_model.get_action(backbone_output=backbone_output, action_input=action_input)
            pred_actions = action_outputs["action_pred"]

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _add_robo_meta_tokens_to_instructions(self, examples, instructions):
        """Add robot-specific meta info to instructions"""
        enhanced_instructions = []
        
        for example, instruction in zip(examples, instructions):
            robot_tag = example["robot_tag"]
            
            # robo info
            robo_info = "Robot name: {}. Action Dim: {}. Robot info: {}".format(
                robot_tag,
                multi_robot_action_heads[robot_tag]["action_dim"],
                multi_robot_action_heads[robot_tag]["robo_info"] if "robo_info" in multi_robot_action_heads[robot_tag] else "N/A",
            )
            chunk_len = multi_robot_action_heads[robot_tag]["NUM_ACTIONS_CHUNK"]
            
            prompt_suffix = f" Please predict the next {chunk_len} robot actions for the robot {robo_info}."
            enhanced_instructions.append(instruction + prompt_suffix)
        
        return enhanced_instructions

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    args.config_yaml = "examples/MultiRobot/train_files/starvla_cotrain_multiRobot_exp3.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
        "robot_tag": "franka",
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
        "robot_tag": "franka",
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample, but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # forward model with dataloader
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        # break

    action = model.predict_action(examples=batch)
    print("Finished")
