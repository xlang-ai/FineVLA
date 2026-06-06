# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025]. 

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md (adpat a little code)
  
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images

# build MLP action head for each robot
# TODO make this configurable from yaml
multi_robot_action_heads = {
"franka": {"action_dim": 7, "NUM_ACTIONS_CHUNK":16, "action_token": "<robot_action_100>" }, # LIBERO
"oxe_bridge": {"action_dim": 7, "NUM_ACTIONS_CHUNK":16, "action_token": "<robot_action_8>"}, # OXE-Bridge
"oxe_rt1": {"action_dim": 7, "NUM_ACTIONS_CHUNK":16, "action_token": "<robot_action_24>"}, # OXE-RT1
"gr1": {"action_dim": 29, "NUM_ACTIONS_CHUNK":16, "action_token": "<robot_action_40>"}, # Robocasa-GR1
"robotwin": {"action_dim": 14, "NUM_ACTIONS_CHUNK":16, "action_token": "<robot_action_56>"}, # Robotwin
}

from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model_default
def get_robot_action_head_list(robot_names: list[str], config: dict):

    for robot_name in robot_names:
        if robot_name not in multi_robot_action_heads:
            raise ValueError(f"Robot name {robot_names} not supported in multi_robot_action_heads") 
    action_heads = {}
    for robot_name in robot_names:
        action_dim = multi_robot_action_heads[robot_name]["action_dim"]
        NUM_ACTIONS_CHUNK = multi_robot_action_heads[robot_name]["NUM_ACTIONS_CHUNK"]
        action_hidden_dim = config.framework.action_model.action_hidden_dim
        action_head = get_action_model_default(
            input_dim=action_hidden_dim,
            action_dim=action_dim,
            NUM_ACTIONS_CHUNK=NUM_ACTIONS_CHUNK,
        )
        action_heads[robot_name] = action_head
    action_heads = nn.ModuleDict(action_heads)
    return action_heads





@FRAMEWORK_REGISTRY.register("QwenOFT_multiRobo")
class Qwenvl_xOFT(baseframework):
    """
    Multimodal vision-language-action model (Qwen-OFT).

    Overview:
      - Qwen2.5 vision-language backbone
      - Inject an action placeholder token into the text prompt
      - Regress continuous actions from the hidden states at those token positions
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            config: Framework/training configuration (dict/OmegaConf-like).
            **kwargs: Reserved for future use.
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Align action head hidden dim with the VLM hidden size.
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        robot_names = config.datasets.vla_data.robo_names 
        # ["franka", "oxe_bridge", "oxe_rt1", "gr1"]  # TODO: make this configurable from yaml

        self.action_models = get_robot_action_head_list(robot_names=robot_names, config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size


        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Training forward: regress future actions (no diffusion).

        Expected keys per example:
            - image: List[PIL.Image] (multi-view)
            - lang: str instruction
            - action: array-like with shape [T, action_dim]

        Returns:
            {"action_loss": torch.Tensor} scalar L1 loss.
        """
        batch_images = [example["image"] for example in examples]  # [B, List[PIL.Image]]
        instructions = [example["lang"] for example in examples]  # [B]
        actions = [example["action"] for example in examples]  # [B, T, action_dim]

        # 2. Enhance instructions (add action tokens)
        enhanced_instructions = self._add_action_tokens_to_instructions(examples, instructions)
    
        # 3. Build QwenVL inputs. and 4. VLM forward pass (get hidden states)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=enhanced_instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # 5. Extract input_ids and create action_query_mask
        input_ids = qwen_inputs.get("input_ids", None)
        action_query_mask = self.search_action_token_mask(input_ids)

        # 5. Collect data for each robot
        robot_data_dict = self._collect_robot_data(
            examples, last_hidden, action_query_mask, actions
        )

        # 6. Forward pass for each robot's data
        with torch.autocast("cuda", dtype=torch.float32):
            action_loss_dict = {}
            for robo_data in robot_data_dict.values():
                robo_name = robo_data["robo_name"]
                action_model = self.action_models[robo_name]
                action_queries = robo_data["action_queries"]  # [B_robo, chunk_len, vlm_dim]
                pred_actions = action_model.predict_action(action_queries)  # [B, chunk_len, action_dim]

                # Compute L1 loss
                action_robo_taget = robo_data["actions"]  # List of arrays

                action_robo_taget = torch.tensor(np.array(action_robo_taget), device=pred_actions.device, dtype=pred_actions.dtype)
                # actions_target = action_robo # TODO past action will not handleed now
                robo_action_loss = self.l1_loss(pred_actions, action_robo_taget)
                action_loss_dict[robo_name] = robo_action_loss

        # 7. Aggregate losses
        total_action_loss = sum(action_loss_dict.values()) / len(action_loss_dict)
        return {"action_loss": total_action_loss}
    

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> dict:
        """
        Inference: a single forward pass to regress actions (no sampling).

        Expected keys per example:
            - image: PIL.Image | np.ndarray | bytes-like (handled by to_pil_preserve)
            - lang: str instruction

        Returns:
            {"normalized_actions": np.ndarray} with shape [B, chunk_len, action_dim].
        """
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # [B] of PIL-like
        instructions = [example["lang"] for example in examples]  # [B]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)


        # 2. Enhance instructions (add action tokens)
        enhanced_instructions = self._add_action_tokens_to_instructions(examples, instructions)
    
        # 3. Build QwenVL inputs. and 4. VLM forward pass (get hidden states)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=enhanced_instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # 5. Extract input_ids and create action_query_mask
        input_ids = qwen_inputs.get("input_ids", None)
        action_query_mask = self.search_action_token_mask(input_ids)

        # 5. Collect data for each robot
        robot_data_dict = self._collect_robot_data(
            examples, last_hidden, action_query_mask
        )

        # 6. Forward pass for each robot's data
        robo_prodictions = {}
        with torch.autocast("cuda", dtype=torch.float32):
            for robo_data in robot_data_dict.values():
                robo_name = robo_data["robo_name"]
                action_model = self.action_models[robo_name]
                action_queries = robo_data["action_queries"]  # [B_robo, chunk_len, vlm_dim]
                pred_actions = action_model.predict_action(action_queries)  # [B, chunk_len, action_dim]

                normalized_actions = pred_actions.detach().cpu().numpy()
                mse_score = 0
                if "actions" in robo_data:
                    actions = robo_data["actions"]
                    actions = np.array(actions)  # convert actions to numpy.ndarray
                    # B, Chunk, dim = actions.shape
                    num_pots = np.prod(actions.shape)
                    # Compute the metric score
                    score = TrainerUtils.euclidean_distance(normalized_actions, actions)
                    mse_score = score / num_pots

                robo_prodictions[robo_name] = {"normalized_actions": normalized_actions, "mse_score": mse_score}
        
        avg_mse = sum([v["mse_score"] for v in robo_prodictions.values()]) / len(robo_prodictions)
        all_mse_scores = {k: v["mse_score"] for k, v in robo_prodictions.items()}
        return {"normalized_actions": normalized_actions, "mse_score": avg_mse, "all_mse_scores": all_mse_scores}

    def search_action_token_mask(self, input_ids, action_token_min=None, action_token_max=None):
        """
        Search for the action token mask.
        Args:
            input_ids: [batch_size, seq_len] input token IDs
            action_token_min: minimum action token ID (optional)
            action_token_max: maximum action token ID (optional)
        Returns:
            mask: [batch_size, seq_len] boolean mask, True indicates positions where loss should be computed
        """
        if action_token_min is None:
            action_token_min = getattr(self.qwen_vl_interface, '_ACTION_TOKEN_MIN', None)
        if action_token_max is None:
            action_token_max = getattr(self.qwen_vl_interface, '_ACTION_TOKEN_MAX', None)
        
        if action_token_min is None or action_token_max is None:
            raise ValueError("Action token range not specified. Please set _ACTION_TOKEN_MIN and _ACTION_TOKEN_MAX.")
        
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Process each sequence
        for i in range(batch_size):
            seq = input_ids[i]
            
            # Find action token positions
            action_mask = (seq >= action_token_min) & (seq <= action_token_max)
            nonzero_indices = torch.nonzero(action_mask, as_tuple=False)

            mask[i, nonzero_indices] = True

        return mask

    def _add_action_tokens_to_instructions(self, examples, instructions):
        """Add action token placeholders to instructions"""
        enhanced_instructions = []
        
        for example, instruction in zip(examples, instructions):
            robot_tag = example["robot_tag"]
            action_token = multi_robot_action_heads[robot_tag]["action_token"]
            action_token_start = int(action_token.split("_")[-1].replace(">", ""))
            chunk_len = multi_robot_action_heads[robot_tag]["NUM_ACTIONS_CHUNK"]
            
            action_query = ""
            for action_index in range(chunk_len):
                action_query += f"<robot_action_{action_index+action_token_start}>"
            
            prompt_suffix = f" Please predict the next {chunk_len} robot actions for {robot_tag}: {action_query}."
            enhanced_instructions.append(instruction + prompt_suffix)
        
        return enhanced_instructions

    def _collect_robot_data(self, examples, last_hidden, action_query_mask, actions=None):
        """Collect data for each robot"""
        robot_data_dict = {}
        if "action" in examples[0]:
            actions = [example["action"] for example in examples]
        for robot_name in self.action_models.keys():
            indices = [i for i, ex in enumerate(examples) if ex["robot_tag"] == robot_name]
            if not indices:
                continue

            
            robot_last_hidden = last_hidden[indices]  # [B_robo, L, H]
            robot_query_mask = action_query_mask[indices] # # [B_robo, L]
            B_robo, L, Dim = robot_last_hidden.shape
            chunk_len = multi_robot_action_heads[robot_name]["NUM_ACTIONS_CHUNK"]
            action_queries = robot_last_hidden[robot_query_mask] # total_k, Dim
            action_queries = action_queries.reshape(B_robo,chunk_len, Dim)

            robot_data_dict[robot_name] = {
                "robo_name": robot_name,
                "indices": indices,
                "action_queries": action_queries,
                "mask": robot_query_mask,
                "actions": [actions[i] for i in indices] if actions is not None else None,
                "chunk_len": chunk_len
            }
        
        return robot_data_dict

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/MultiRobot/train_files/starvla_cotrain_multiRobot_exp3.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.action_model.action_hidden_dim = 2048


    # try get model
    model = Qwenvl_xOFT(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(8, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # two views
        "lang": "This is a fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
        "robot_tag": "franka",
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], 
        "lang": "For testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
        "robot_tag": "franka",
    }


    sample3 = {
        "action": np.random.uniform(-1, 1, size=(16, 14)).astype(np.float16), # action_chunk, action_dim
        "image": [image], 
        "lang": "For testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
        'robot_tag': "gr1",
    }


    batch  = [sample, sample2, sample3]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample3])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")


    # try forward model
    # can be fake sample, but here get from dataloader for simpler
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        # break

        # try get model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model(batch)
        pass
        action = model.predict_action(batch)