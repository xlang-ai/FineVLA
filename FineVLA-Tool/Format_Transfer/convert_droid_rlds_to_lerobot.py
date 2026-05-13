"""
Convert DROID RLDS dataset (TFRecord format) to LeRobot format.

This script:
1. Reads DROID data from TFRecord RLDS format
2. Samples only a specified number of tfrecords for testing
3. Encodes images into MP4 videos directly in LeRobot's videos folder
4. Saves everything locally (no HuggingFace operations)

Usage:
python convert_droid_rlds_to_lerobot.py

Modify the paths at the bottom of the script as needed.
"""

import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from tqdm import tqdm

# Disable GPU for TensorFlow (we only need CPU for data loading)
tf.config.set_visible_devices([], 'GPU')


def decode_tf_example(serialized_example):
    """Decode a TFRecord example from DROID RLDS format."""
    # Define the feature description based on RLDS format
    # The RLDS format uses nested structure with steps as a sequence
    feature_description = {
        'episode_metadata/file_path': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'episode_metadata/recording_folderpath': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    
    # Try to parse just the metadata first
    try:
        parsed = tf.io.parse_single_example(serialized_example, feature_description)
        return parsed
    except:
        return None


def load_rlds_episode(tfrecord_path):
    """
    Load a single episode from a TFRecord file using tensorflow_datasets approach.
    Returns a list of steps, where each step contains observations, actions, etc.
    """
    # Read the raw dataset
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    episodes = []
    for raw_record in raw_dataset:
        # For RLDS format, we need to use the appropriate deserialization
        # RLDS stores episodes with steps as a sequence feature
        episode_data = {
            'steps': [],
            'metadata': {}
        }
        
        try:
            # Parse as SequenceExample which is common in RLDS
            context, sequences = tf.io.parse_single_sequence_example(
                raw_record,
                context_features={
                    'episode_metadata/file_path': tf.io.FixedLenFeature([], tf.string, default_value=b''),
                    'episode_metadata/recording_folderpath': tf.io.FixedLenFeature([], tf.string, default_value=b''),
                },
                sequence_features={
                    'steps/observation/exterior_image_1_left': tf.io.FixedLenSequenceFeature([], tf.string),
                    'steps/observation/exterior_image_2_left': tf.io.FixedLenSequenceFeature([], tf.string),
                    'steps/observation/wrist_image_left': tf.io.FixedLenSequenceFeature([], tf.string),
                    'steps/observation/joint_position': tf.io.FixedLenSequenceFeature([7], tf.float64),
                    'steps/observation/gripper_position': tf.io.FixedLenSequenceFeature([1], tf.float64),
                    'steps/observation/cartesian_position': tf.io.FixedLenSequenceFeature([6], tf.float64),
                    'steps/action': tf.io.FixedLenSequenceFeature([7], tf.float64),
                    'steps/action_dict/joint_velocity': tf.io.FixedLenSequenceFeature([7], tf.float64),
                    'steps/action_dict/gripper_position': tf.io.FixedLenSequenceFeature([1], tf.float64),
                    'steps/action_dict/gripper_velocity': tf.io.FixedLenSequenceFeature([1], tf.float64),
                    'steps/language_instruction': tf.io.FixedLenSequenceFeature([], tf.string),
                    'steps/is_first': tf.io.FixedLenSequenceFeature([], tf.bool),
                    'steps/is_last': tf.io.FixedLenSequenceFeature([], tf.bool),
                    'steps/is_terminal': tf.io.FixedLenSequenceFeature([], tf.bool),
                }
            )
            
            episode_data['metadata']['file_path'] = context['episode_metadata/file_path'].numpy().decode('utf-8')
            episode_data['metadata']['recording_folderpath'] = context['episode_metadata/recording_folderpath'].numpy().decode('utf-8')
            
            num_steps = len(sequences['steps/action'])
            
            for i in range(num_steps):
                step = {
                    'observation': {
                        'exterior_image_1_left': tf.io.decode_jpeg(sequences['steps/observation/exterior_image_1_left'][i]).numpy(),
                        'exterior_image_2_left': tf.io.decode_jpeg(sequences['steps/observation/exterior_image_2_left'][i]).numpy(),
                        'wrist_image_left': tf.io.decode_jpeg(sequences['steps/observation/wrist_image_left'][i]).numpy(),
                        'joint_position': sequences['steps/observation/joint_position'][i].numpy(),
                        'gripper_position': sequences['steps/observation/gripper_position'][i].numpy(),
                        'cartesian_position': sequences['steps/observation/cartesian_position'][i].numpy(),
                    },
                    'action': sequences['steps/action'][i].numpy(),
                    'action_dict': {
                        'joint_velocity': sequences['steps/action_dict/joint_velocity'][i].numpy(),
                        'gripper_position': sequences['steps/action_dict/gripper_position'][i].numpy(),
                        'gripper_velocity': sequences['steps/action_dict/gripper_velocity'][i].numpy(),
                    },
                    'language_instruction': sequences['steps/language_instruction'][i].numpy().decode('utf-8'),
                    'is_first': sequences['steps/is_first'][i].numpy(),
                    'is_last': sequences['steps/is_last'][i].numpy(),
                    'is_terminal': sequences['steps/is_terminal'][i].numpy(),
                }
                episode_data['steps'].append(step)
            
            episodes.append(episode_data)
            
        except Exception as e:
            print(f"Error parsing record: {e}")
            continue
    
    return episodes


def load_rlds_with_tfds(data_dir, num_shards=10):
    """
    Load RLDS dataset using tensorflow_datasets.
    This is more robust than manual parsing.
    """
    import tensorflow_datasets as tfds
    
    # Build the dataset from the directory
    builder = tfds.builder_from_directory(data_dir)
    
    # Get the dataset - we can limit the number of shards by specifying split
    # For sampling, we'll just take the first num_shards worth of data
    ds = builder.as_dataset(split='train')
    
    return ds


def encode_video_frames(frames, output_path, fps=15):
    """
    Encode a list of image frames into an MP4 video.
    
    Args:
        frames: List of numpy arrays (H, W, C) in RGB format
        output_path: Path to save the MP4 file
        fps: Frames per second
    """
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    
    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    
    writer.release()


def create_lerobot_dataset_structure(output_dir):
    """Create the LeRobot dataset directory structure."""
    output_dir = Path(output_dir)
    
    # Create directories
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    
    return output_dir


def convert_droid_rlds_to_lerobot(
    data_dir: str,
    output_dir: str,
    num_shards: int = 10,
    fps: int = 15,
):
    """
    Convert DROID RLDS dataset to LeRobot format.
    
    Args:
        data_dir: Path to DROID RLDS dataset (containing tfrecord files)
        output_dir: Path to output LeRobot dataset
        num_shards: Number of tfrecord shards to process (for testing)
        fps: Frames per second for output videos
    """
    import tensorflow_datasets as tfds
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Clean up existing output
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Create directory structure
    create_lerobot_dataset_structure(output_dir)
    
    print(f"Loading DROID RLDS dataset from: {data_dir}")
    
    # Load with tensorflow_datasets
    try:
        builder = tfds.builder_from_directory(str(data_dir))
        ds = builder.as_dataset(split='train')
        print(f"Successfully loaded dataset with tfds")
    except Exception as e:
        print(f"Failed to load with tfds: {e}")
        print("Falling back to manual TFRecord loading...")
        ds = None
    
    if ds is not None:
        # Process with tfds
        convert_with_tfds(ds, output_dir, num_shards, fps)
    else:
        # Fallback to manual TFRecord loading
        convert_with_manual_loading(data_dir, output_dir, num_shards, fps)


def convert_with_tfds(ds, output_dir, num_shards, fps):
    """Convert using tensorflow_datasets iterator."""
    output_dir = Path(output_dir)
    videos_dir = output_dir / "videos"
    data_dir = output_dir / "data"
    meta_dir = output_dir / "meta"
    
    # Collect data for parquet files
    all_frames_data = []
    episode_info = []
    
    # Camera names
    camera_names = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    
    # Create video subdirectories for each camera
    for cam in camera_names:
        (videos_dir / cam).mkdir(parents=True, exist_ok=True)
    
    episode_idx = 0
    total_frames = 0
    
    # Count episodes to process
    max_episodes = num_shards * 50  # Rough estimate: ~50 episodes per shard
    
    print(f"Processing up to {max_episodes} episodes (from {num_shards} shards)...")
    
    for episode in tqdm(ds.take(max_episodes), desc="Converting episodes"):
        try:
            steps = list(episode['steps'])
            if len(steps) == 0:
                continue
            
            # Collect frames for video encoding
            frames_by_camera = {cam: [] for cam in camera_names}
            episode_frames_data = []
            
            # Get language instruction from first step
            language_instruction = steps[0]['language_instruction'].numpy().decode('utf-8') if 'language_instruction' in steps[0] else "Do something"
            
            for step_idx, step in enumerate(steps):
                obs = step['observation']
                action_dict = step.get('action_dict', step.get('action', {}))
                
                # Collect images for video
                for cam in camera_names:
                    if cam in obs:
                        img = obs[cam].numpy()
                        frames_by_camera[cam].append(img)
                
                # Collect frame data for parquet
                frame_data = {
                    'episode_index': episode_idx,
                    'frame_index': step_idx,
                    'timestamp': step_idx / fps,
                    'index': total_frames + step_idx,
                }
                
                # Add state data
                if 'joint_position' in obs:
                    joint_pos = obs['joint_position'].numpy().astype(np.float32)
                    for i, val in enumerate(joint_pos):
                        frame_data[f'observation.state.joint_position.{i}'] = float(val)
                
                if 'gripper_position' in obs:
                    gripper_pos = obs['gripper_position'].numpy().astype(np.float32)
                    frame_data['observation.state.gripper_position'] = float(gripper_pos[0]) if len(gripper_pos) > 0 else 0.0
                
                if 'cartesian_position' in obs:
                    cart_pos = obs['cartesian_position'].numpy().astype(np.float32)
                    for i, val in enumerate(cart_pos):
                        frame_data[f'observation.state.cartesian_position.{i}'] = float(val)
                
                # Add action data
                if isinstance(action_dict, dict):
                    if 'joint_velocity' in action_dict:
                        joint_vel = action_dict['joint_velocity'].numpy().astype(np.float32)
                        for i, val in enumerate(joint_vel):
                            frame_data[f'action.joint_velocity.{i}'] = float(val)
                    if 'gripper_position' in action_dict:
                        gripper_action = action_dict['gripper_position'].numpy().astype(np.float32)
                        frame_data['action.gripper_position'] = float(gripper_action[0]) if len(gripper_action) > 0 else 0.0
                else:
                    # Action is a tensor directly
                    action = action_dict.numpy().astype(np.float32) if hasattr(action_dict, 'numpy') else np.array(action_dict, dtype=np.float32)
                    for i, val in enumerate(action):
                        frame_data[f'action.{i}'] = float(val)
                
                # Add task/language instruction
                frame_data['task'] = language_instruction
                
                episode_frames_data.append(frame_data)
            
            # Encode and save videos for this episode
            for cam in camera_names:
                if len(frames_by_camera[cam]) > 0:
                    video_path = videos_dir / cam / f"episode_{episode_idx:06d}.mp4"
                    encode_video_frames(frames_by_camera[cam], video_path, fps)
            
            # Add video paths to frame data
            for frame_data in episode_frames_data:
                for cam in camera_names:
                    frame_data[f'observation.images.{cam}'] = f"videos/{cam}/episode_{episode_idx:06d}.mp4"
            
            all_frames_data.extend(episode_frames_data)
            
            # Record episode info
            episode_info.append({
                'episode_index': episode_idx,
                'num_frames': len(steps),
                'language_instruction': language_instruction,
            })
            
            total_frames += len(steps)
            episode_idx += 1
            
            # Stop after processing enough shards worth of episodes
            if episode_idx >= num_shards * 40:  # Approximate based on shardLengths
                break
                
        except Exception as e:
            print(f"Error processing episode {episode_idx}: {e}")
            continue
    
    print(f"Processed {episode_idx} episodes with {total_frames} total frames")
    
    # Save data to parquet
    if all_frames_data:
        save_parquet_data(all_frames_data, data_dir)
    
    # Save metadata
    save_metadata(output_dir, episode_info, fps, camera_names, total_frames)
    
    print(f"Dataset saved to: {output_dir}")


def convert_with_manual_loading(data_dir, output_dir, num_shards, fps):
    """Fallback: manually load TFRecord files."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    videos_dir = output_dir / "videos"
    out_data_dir = output_dir / "data"
    
    # Find tfrecord files
    tfrecord_files = sorted(glob.glob(str(data_dir / "*.tfrecord*")))
    print(f"Found {len(tfrecord_files)} tfrecord files")
    
    # Sample first N shards
    tfrecord_files = tfrecord_files[:num_shards]
    print(f"Processing {len(tfrecord_files)} tfrecord files")
    
    camera_names = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    
    # Create video subdirectories
    for cam in camera_names:
        (videos_dir / cam).mkdir(parents=True, exist_ok=True)
    
    all_frames_data = []
    episode_info = []
    episode_idx = 0
    total_frames = 0
    
    for tf_path in tqdm(tfrecord_files, desc="Processing tfrecord files"):
        episodes = load_rlds_episode(tf_path)
        
        for episode_data in episodes:
            steps = episode_data['steps']
            if len(steps) == 0:
                continue
            
            frames_by_camera = {cam: [] for cam in camera_names}
            episode_frames_data = []
            
            language_instruction = steps[0].get('language_instruction', "Do something")
            
            for step_idx, step in enumerate(steps):
                obs = step['observation']
                action_dict = step.get('action_dict', {})
                
                for cam in camera_names:
                    if cam in obs:
                        frames_by_camera[cam].append(obs[cam])
                
                frame_data = {
                    'episode_index': episode_idx,
                    'frame_index': step_idx,
                    'timestamp': step_idx / fps,
                    'index': total_frames + step_idx,
                }
                
                if 'joint_position' in obs:
                    for i, val in enumerate(obs['joint_position']):
                        frame_data[f'observation.state.joint_position.{i}'] = float(val)
                
                if 'gripper_position' in obs:
                    gripper_pos = obs['gripper_position']
                    frame_data['observation.state.gripper_position'] = float(gripper_pos[0]) if len(gripper_pos) > 0 else 0.0
                
                if 'joint_velocity' in action_dict:
                    for i, val in enumerate(action_dict['joint_velocity']):
                        frame_data[f'action.joint_velocity.{i}'] = float(val)
                if 'gripper_position' in action_dict:
                    gripper_action = action_dict['gripper_position']
                    frame_data['action.gripper_position'] = float(gripper_action[0]) if len(gripper_action) > 0 else 0.0
                
                frame_data['task'] = language_instruction
                episode_frames_data.append(frame_data)
            
            # Encode videos
            for cam in camera_names:
                if len(frames_by_camera[cam]) > 0:
                    video_path = videos_dir / cam / f"episode_{episode_idx:06d}.mp4"
                    encode_video_frames(frames_by_camera[cam], video_path, fps)
            
            for frame_data in episode_frames_data:
                for cam in camera_names:
                    frame_data[f'observation.images.{cam}'] = f"videos/{cam}/episode_{episode_idx:06d}.mp4"
            
            all_frames_data.extend(episode_frames_data)
            
            episode_info.append({
                'episode_index': episode_idx,
                'num_frames': len(steps),
                'language_instruction': language_instruction,
            })
            
            total_frames += len(steps)
            episode_idx += 1
    
    print(f"Processed {episode_idx} episodes with {total_frames} total frames")
    
    if all_frames_data:
        save_parquet_data(all_frames_data, out_data_dir)
    
    save_metadata(output_dir, episode_info, fps, camera_names, total_frames)
    print(f"Dataset saved to: {output_dir}")


def save_parquet_data(frames_data, data_dir):
    """Save frame data to parquet files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to pyarrow table and save
    table = pa.Table.from_pylist(frames_data)
    
    # Save as a single parquet file (for simplicity in testing)
    # In production, you might want to chunk this
    pq.write_table(table, data_dir / "train-00000-of-00001.parquet")
    print(f"Saved parquet data with {len(frames_data)} frames")


def save_metadata(output_dir, episode_info, fps, camera_names, total_frames):
    """Save LeRobot metadata files."""
    output_dir = Path(output_dir)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "fps": fps,
        "total_episodes": len(episode_info),
        "total_frames": total_frames,
        "data_path": "data/train-00000-of-00001.parquet",
        "video_path": "videos",
        "features": {
            "observation.images.exterior_image_1_left": {
                "dtype": "video",
                "shape": [180, 320, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                }
            },
            "observation.images.exterior_image_2_left": {
                "dtype": "video",
                "shape": [180, 320, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                }
            },
            "observation.images.wrist_image_left": {
                "dtype": "video",
                "shape": [180, 320, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                }
            },
            "observation.state.joint_position": {
                "dtype": "float32",
                "shape": [7],
                "names": ["joint_position"],
            },
            "observation.state.gripper_position": {
                "dtype": "float32",
                "shape": [1],
                "names": ["gripper_position"],
            },
            "action.joint_velocity": {
                "dtype": "float32",
                "shape": [7],
                "names": ["joint_velocity"],
            },
            "action.gripper_position": {
                "dtype": "float32",
                "shape": [1],
                "names": ["gripper_position"],
            },
        }
    }
    
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # episodes.json
    episodes_data = {
        "episodes": [
            {
                "episode_index": ep["episode_index"],
                "length": ep["num_frames"],
                "task": ep["language_instruction"],
            }
            for ep in episode_info
        ]
    }
    
    with open(meta_dir / "episodes.json", "w") as f:
        json.dump(episodes_data, f, indent=2)
    
    # tasks.json - collect unique tasks
    unique_tasks = list(set(ep["language_instruction"] for ep in episode_info))
    tasks_data = {"tasks": unique_tasks}
    
    with open(meta_dir / "tasks.json", "w") as f:
        json.dump(tasks_data, f, indent=2)
    
    print(f"Saved metadata to {meta_dir}")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/cpfs01/data/shared/Group-m6/dannyXSC/data/droid/1.0.1"
    OUTPUT_DIR = "/cpfs02/shared/Group-m6/xuhong.hxh/data/droid"
    NUM_SHARDS = 10  # Only process 10 tfrecord shards for testing
    FPS = 15  # DROID is typically 15fps
    
    convert_droid_rlds_to_lerobot(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        num_shards=NUM_SHARDS,
        fps=FPS,
    )
