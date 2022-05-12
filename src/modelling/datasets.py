import io
import json
import math
import os
import re

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, RandomCrop, CenterCrop, Resize, ToTensor, ColorJitter
from torchvision.transforms import functional as TF
from utils.data_utils import IdentityTransform, VideoColorJitter, fix_box, \
    get_test_layout_indices, pad_sequence, sample_appearance_indices, sample_train_layout_indices
from modelling.configs import DataConfig


class StltDataset(Dataset):
    def __init__(self, config: DataConfig):
        self.config = config
        self.json_file = json.load(open(self.config.dataset_path))
        self.labels = self.config.labels  # Shallow copy of labels
        self.videoid2size = json.load(open(self.config.videoid2size_path))
        # Find max num objects
        max_objects = -1
        for video in self.json_file:
            for video_frame in video["frames"]:
                cur_num_objects = 0
                for frame_object in video_frame["frame_objects"]:
                    if frame_object["score"] >= self.config.score_threshold:
                        cur_num_objects += 1
                max_objects = max(max_objects, cur_num_objects)
        self.config.max_num_objects = max_objects

    def __len__(self):
        return self.config.debug_size if self.config.debug_size is not None else len(self.json_file)

    def __getitem__(self, idx: int):
        video_id = self.json_file[idx]["id"]
        video_size = torch.tensor(self.videoid2size[video_id]).repeat(2)
        boxes, categories, scores, frame_types = [], [], [], []
        num_frames = len(self.json_file[idx]["frames"])
        indices = (
            sample_train_layout_indices(self.config.layout_num_frames, num_frames)
            if self.config.train
            else get_test_layout_indices(self.config.layout_num_frames, num_frames)
        )
        for index in indices:
            frame = self.json_file[idx]["frames"][index]
            # Prepare CLS object
            frame_types.append(
                self.config.frame2type["empty"]
                if len(frame["frame_objects"]) == 0
                else self.config.frame2type["regular"]
            )
            frame_boxes = [torch.tensor([0.0, 0.0, 1.0, 1.0])]
            frame_categories = [self.config.categories["cls"]]
            frame_scores = [1.0]
            # Iterate over the other objects
            for element in frame["frame_objects"]:
                if element["score"] < self.config.score_threshold:
                    continue
                # Prepare box
                box = [element["x1"], element["y1"], element["x2"], element["y2"]]
                box = fix_box(
                    box, (video_size[1].item(), video_size[0].item())
                )  # Height, Width
                box = torch.tensor(box) / video_size
                frame_boxes.append(box)
                # Prepare category
                frame_categories.append(self.config.categories[element["category"]])
                # Prepare scores
                frame_scores.append(element["score"])
            # Ensure that everything is of the same length and pad to the max number of objects
            assert len(frame_boxes) == len(frame_categories)
            while len(frame_boxes) != self.config.max_num_objects + 1:
                frame_boxes.append(torch.full((4,), 0.0))
                frame_categories.append(0)
                frame_scores.append(0.0)
            categories.append(torch.tensor(frame_categories))
            scores.append(torch.tensor(frame_scores))
            boxes.append(torch.stack(frame_boxes, dim=0))
        # Prepare extract element
        # Boxes
        extract_box = torch.full((self.config.max_num_objects + 1, 4), 0.0)
        extract_box[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        boxes.append(extract_box)
        # Categories
        extract_category = torch.full((self.config.max_num_objects + 1,), 0)
        extract_category[0] = self.config.categories["cls"]
        categories.append(extract_category)
        # Scores
        extract_score = torch.full((self.config.max_num_objects + 1,), 0.0)
        extract_score[0] = 1.0
        scores.append(extract_score)
        # Length
        length = torch.tensor(len(categories))
        # Frame types
        frame_types.append(self.config.frame2type["extract"])
        # Get action(s)
        actions = self.get_actions(self.json_file[idx])

        return {
            "video_id": video_id,
            "categories": torch.stack(categories, dim=0),
            "boxes": torch.stack(boxes, dim=0),
            "scores": torch.stack(scores, dim=0),
            "frame_types": torch.tensor(frame_types),
            "lengths": length,
            "labels": actions,
        }

    def get_actions(self, sample):
        if self.config.dataset_name == "something":
            actions = torch.tensor(int(self.labels[re.sub("[\[\]]", "", sample["template"])]))
        else:
            action_list = [int(action[1:]) for action in sample["actions"]]
            actions = torch.zeros(len(self.labels), dtype=torch.float)
            actions[action_list] = 1.0
        return actions


class MouseDataset(Dataset):

    def __init__(self, config: DataConfig):
        self.config = config
        self.json_file = json.load(open(self.config.dataset_path))
        self.labels = self.config.labels  # Shallow copy of labels
        self.videoid2size = json.load(open(self.config.videoid2size_path))
        self.config.max_num_objects = 4 if config.include_hopper else 3  # 3 Mice + Hopper
        self.identity_flip = np.random.default_rng()

    def __len__(self):
        return self.config.debug_size if self.config.debug_size is not None else len(self.json_file)

    def __getitem__(self, idx: int):
        video_id = self.json_file[idx]["id"]
        video_size = torch.tensor(self.videoid2size[video_id]).repeat(2)
        boxes, categories, scores, frame_types = [], [], [], []
        num_frames = len(self.json_file[idx]["frames"])
        indices = (
            sample_train_layout_indices(self.config.layout_num_frames, num_frames)
            if self.config.train
            else get_test_layout_indices(self.config.layout_num_frames, num_frames)
        )
        # Define Mapping: Note that
        #   1: Augmentation is only done in the Training set
        #   2: ID Switching (at random) is done on a consistent per-BTI basis
        _category_map = self.config.categories.copy()
        if self.config.train and self.config.maintain_identities and (self.identity_flip.binomial(1, 0.5) == 1):
            _category_map['cagemate_1'] = self.config.categories['cagemate_2']
            _category_map['cagemate_2'] = self.config.categories['cagemate_1']

        for index in indices:
            frame = self.json_file[idx]["frames"][index]
            # Prepare CLS object
            frame_boxes = [torch.tensor([0.0, 0.0, 1.0, 1.0])]
            frame_categories = [self.config.categories["cls"]]
            frame_scores = [1.0]
            _empty_frame = True
            # Iterate over the other objects
            for element in frame["frame_objects"]:
                if element["score"] < self.config.score_threshold:
                    continue  # Skip if score does not match
                if (element["category"] == "hopper") and not self.config.include_hopper:
                    continue  # skip if not including hopper
                # Prepare box
                box = [element["x1"], element["y1"], element["x2"], element["y2"]]
                box = fix_box(box, (video_size[1].item(), video_size[0].item()))  # Height, Width
                box = torch.tensor(box) / video_size
                frame_boxes.append(box)
                # Prepare category
                frame_categories.append(_category_map[element["category"]])
                # Prepare scores
                frame_scores.append(element["score"])
                _empty_frame = False
            # Ensure that everything is of the same length and pad to the max number of objects
            assert len(frame_boxes) == len(frame_categories)
            while len(frame_boxes) != self.config.max_num_objects + 1:
                frame_boxes.append(torch.full((4,), 0.0))
                frame_categories.append(_category_map['pad'])
                frame_scores.append(0.0)
            categories.append(torch.tensor(frame_categories))
            scores.append(torch.tensor(frame_scores))
            boxes.append(torch.stack(frame_boxes, dim=0))
            frame_types.append(self.config.frame2type["empty" if _empty_frame else "regular"])
        # Prepare extract element
        # Boxes
        extract_box = torch.full((self.config.max_num_objects + 1, 4), 0.0)
        extract_box[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        boxes.append(extract_box)
        # Categories
        extract_category = torch.full((self.config.max_num_objects + 1,), 0)
        extract_category[0] = self.config.categories["cls"]
        categories.append(extract_category)
        # Scores
        extract_score = torch.full((self.config.max_num_objects + 1,), 0.0)
        extract_score[0] = 1.0
        scores.append(extract_score)
        # Length
        length = torch.tensor(len(categories))
        # Frame types
        frame_types.append(self.config.frame2type["extract"])
        # Get action(s)
        actions = self.get_actions(self.json_file[idx])

        return {
            "video_id": video_id,
            "categories": torch.stack(categories, dim=0),
            "boxes": torch.stack(boxes, dim=0),
            "scores": torch.stack(scores, dim=0),
            "frame_types": torch.tensor(frame_types),
            "lengths": length,
            "labels": actions,
        }

    def get_actions(self, sample):
        return torch.tensor(int(self.labels[re.sub("[\[\]]", "", sample["template"])]))


class FrameDataSet(Dataset):
    """
    Note, that for this class, the Videos Path is the absolute path to the base Frames

    Note also, that currently, this supports only all videos of the same size.
    """
    def __init__(self, config: DataConfig, json_file=None):
        self.config = config
        self.json_file = json_file if json_file is not None else json.load(open(self.config.dataset_path))
        self.labels = self.config.labels
        # Resolve output Frame-Size: this is inferred from the first video and then scaled
        _frame_size = np.asarray(list(json.load(open(self.config.videoid2size_path)).values())[0])
        _frame_size = (_frame_size / np.min(_frame_size) * self.config.min_scale)
        # Resolve enlarge to allow Random-Crop
        _frame_enlarge = (_frame_size * self.config.crop_scale).astype(int).tolist()
        _frame_size = _frame_size.astype(int).tolist()
        # Note that in both Training/Validation, we need to follow the same procedure of scaling
        #    up and then down, as this ensures that the scale is always the same. The difference
        #    is that in Validation/Test, we do a fixed center-crop.
        if self.config.train:
            self.transforms = Compose([
                Resize(_frame_enlarge),  # First Scale to appropriate scale
                RandomCrop(_frame_size),  # Now crop (return original if crop_scale == 1)
                ColorJitter(brightness=(0.75, 1.25), contrast=(0.75, 1.25)),
                ToTensor(),
                Normalize(mean=self.config.normaliser_means, std=self.config.normaliser_stds),
            ])
        else:
            self.transforms = Compose([
                Resize(_frame_enlarge),  # Again resize
                CenterCrop(_frame_size), # And Crop
                ToTensor(),
                Normalize(mean=self.config.normaliser_means, std=self.config.normaliser_stds),
            ])

    def __len__(self):
        return self.config.debug_size if self.config.debug_size is not None else len(self.json_file)

    def __sample_frames(self, sample):
        """
        Wrapper for Sampling in my specific use-case

        :param sample: Sample to which this pertains
        :return:
        """
        _num_frames = len(sample['frames'])
        _num_sampled = self.config.appearance_num_frames
        if _num_frames < _num_sampled:
            return np.arange(_num_frames)
        else:
            return np.sort(np.random.choice(np.arange(_num_frames), _num_sampled, replace=False))

    def __getitem__(self, idx: int):
        # Prepare MetaData
        _sample = self.json_file[idx]
        indices = self.__sample_frames(_sample)

        # Load all frames by their indices
        frm_pth = os.path.join(self.config.videos_path, _sample['video'], "img_{:05d}.jpg")
        video_frames = [Image.open(frm_pth.format(ix+_sample['offset'])) for ix in indices]

        # Build Batch
        video_frames = [self.transforms(frame) for frame in video_frames]
        video_frames = torch.stack(video_frames, dim=0).transpose(0, 1)

        # Obtain video label
        action = torch.tensor(int(self.labels[re.sub("[\[\]]", "", _sample["template"])]))

        return {"video_id": _sample['id'], "video_frames": video_frames, "labels": action}


class BBoxDataSet(Dataset):
    """
    TODO This Class extracts only the BBox regions in the frame of interest

    As for the Frame Dataset:
        1. The Videos Path is the absolute path to the base Frames
        2. Currently, this supports only all videos of the same size.
    """
    def __init__(self, config: DataConfig, json_file=None):
        self.config = config
        self.json_file = json_file if json_file is not None else json.load(open(self.config.dataset_path))
        self.labels = self.config.labels  # As a convenience

        # Resolve input Frame-Size: this is inferred from the first video
        _frame_size = np.asarray(list(json.load(open(self.config.videoid2size_path)).values())[0])
        _frame_size = (_frame_size / np.min(_frame_size) * self.config.min_scale)
        # Resolve enlarge to allow Random-Crop
        _frame_enlarge = (_frame_size * self.config.crop_scale).astype(int).tolist()
        _frame_size = _frame_size.astype(int).tolist()
        # Note that in both Training/Validation, we need to follow the same procedure of scaling
        #    up and then down, as this ensures that the scale is always the same. The difference
        #    is that in Validation/Test, we do a fixed center-crop.
        if self.config.train:
            self.transforms = Compose([
                Resize(_frame_enlarge),  # First Scale to appropriate scale
                RandomCrop(_frame_size),  # Now crop (return original if crop_scale == 1)
                ColorJitter(brightness=(0.75, 1.25), contrast=(0.75, 1.25)),
                ToTensor(),
                Normalize(mean=self.config.normaliser_means, std=self.config.normaliser_stds),
            ])
        else:
            self.transforms = Compose([
                Resize(_frame_enlarge),  # Again resize
                CenterCrop(_frame_size), # And Crop
                ToTensor(),
                Normalize(mean=self.config.normaliser_means, std=self.config.normaliser_stds),
            ])

    def __len__(self):
        return self.config.debug_size if self.config.debug_size is not None else len(self.json_file)


class AppearanceDataset(Dataset):
    def __init__(self, config: DataConfig, json_file=None):
        self.config = config
        self.json_file = json_file
        if not self.json_file:
            self.json_file = json.load(open(self.config.dataset_path))
        self.labels = json.load(open(self.config.labels_path))['labels']
        self.videoid2size = json.load(open(self.config.videoid2size_path))
        self.resize = Resize(math.floor(self.config.min_scale * 1.15))
        self.transforms = Compose(
            [
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.json_file)

    def open_videos(self):
        self.videos = h5py.File(
            self.config.videos_path, "r", libver="latest", swmr=True
        )

    def __getitem__(self, idx: int):
        if not hasattr(self, "videos"):
            self.open_videos()
        video_id = self.json_file[idx]["id"]
        num_frames = len(self.videos[video_id])
        indices = sample_appearance_indices(
            self.config.appearance_num_frames, num_frames, self.config.train
        )
        # Load all frames
        raw_video_frames = [
            self.resize(
                Image.open(io.BytesIO(np.array(self.videos[video_id][str(index)])))
            )
            for index in indices
        ]
        augment = IdentityTransform()
        if self.config.train:
            augment = VideoColorJitter()
            top, left, height, width = RandomCrop.get_params(
                raw_video_frames[0],
                (self.config.min_scale, self.config.min_scale),
            )

        video_frames = []
        for i in range(len(raw_video_frames)):
            frame = raw_video_frames[i]
            frame = augment(frame)
            frame = (
                TF.crop(frame, top, left, height, width)
                if self.config.train
                else TF.center_crop(frame, self.config.spatial_size)
            )
            frame = self.transforms(frame)
            video_frames.append(frame)

        video_frames = torch.stack(video_frames, dim=0).transpose(0, 1)
        # Obtain video label
        video_label = torch.tensor(
            int(self.labels[re.sub("[\[\]]", "", self.json_file[idx]["template"])])
        )

        return {
            "video_id": video_id,
            "video_frames": video_frames,
            "labels": video_label,
        }


class MultimodalDataset(Dataset):
    def __init__(self, config: DataConfig):
        if config.dataset_name == 'mouse':
            self.layout_dataset = MouseDataset(config)
            self.appearance_dataset = FrameDataSet(config, self.layout_dataset.json_file)
        else:
            self.layout_dataset = StltDataset(config)
            self.appearance_dataset = AppearanceDataset(config, self.layout_dataset.json_file)
        # Making a shallow copy of the labels for easier interfacing in train.py and inference.py
        self.labels = self.layout_dataset.labels

    def __len__(self):
        return self.layout_dataset.__len__()

    def __getitem__(self, idx: int):
        layout_dict = self.layout_dataset[idx]
        appearance_dict = self.appearance_dataset[idx]

        # layout_dict and appearance_dict have overlapping video_id and actions
        return {"layout": layout_dict, "appearance": appearance_dict}


datasets_factory = {
    "frames": FrameDataSet,
    "appearance": AppearanceDataset,
    "layout": StltDataset,
    "mouse": MouseDataset,
    "multimodal": MultimodalDataset,
}


class StltCollater:
    def __init__(self, config: DataConfig):
        self.config = config

    def __call__(self, batch):
        batch = {key: [e[key] for e in batch] for key in batch[0].keys()}
        # https://github.com/pytorch/pytorch/issues/24816
        # Pad categories
        pad_categories_tensor = torch.full((self.config.max_num_objects + 1,), 0)
        pad_categories_tensor[0] = self.config.categories["cls"]
        batch["categories"] = pad_sequence(
            batch["categories"], pad_tensor=pad_categories_tensor
        )
        # Hack for padding and using scores only if the dataset is Action Genome :(
        if self.config.dataset_name == "action_genome":
            pad_scores_tensor = torch.full((self.config.max_num_objects + 1,), 0.0)
            pad_scores_tensor[0] = 1.0
            batch["scores"] = pad_sequence(
                batch["scores"], pad_tensor=pad_scores_tensor
            )
        else:
            del batch["scores"]
        # Pad boxes
        pad_boxes_tensor = torch.full((self.config.max_num_objects + 1, 4), 0.0)
        pad_boxes_tensor[0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        batch["boxes"] = pad_sequence(batch["boxes"], pad_tensor=pad_boxes_tensor)
        # Pad frame types
        batch["frame_types"] = pad_sequence(
            batch["frame_types"],
            pad_tensor=torch.tensor([self.config.frame2type["pad"]]),
        )
        # Prepare length and labels
        batch["lengths"] = torch.stack(batch["lengths"], dim=0)
        batch["labels"] = torch.stack(batch["labels"], dim=0)
        # Generate mask for the padding of the boxes
        src_key_padding_mask_boxes = torch.zeros_like(
            batch["categories"], dtype=torch.bool
        )
        src_key_padding_mask_boxes[torch.where(batch["categories"] == 0)] = True
        batch["src_key_padding_mask_boxes"] = src_key_padding_mask_boxes
        # Generate mask for padding of the frames
        src_key_padding_mask_frames = torch.zeros(
            batch["frame_types"].size(), dtype=torch.bool
        )
        src_key_padding_mask_frames[
            torch.where(batch["frame_types"] == self.config.frame2type["pad"])
        ] = True
        batch["src_key_padding_mask_frames"] = src_key_padding_mask_frames

        return batch


class AppearanceCollater:
    def __init__(self, config: DataConfig):
        self.config = config

    def __call__(self, batch):
        batch = {key: [e[key] for e in batch] for key in batch[0].keys()}
        batch["video_frames"] = torch.stack(batch["video_frames"], dim=0)
        batch["labels"] = torch.stack(batch["labels"], dim=0)

        return batch


class MultiModalCollater:
    def __init__(self, config: DataConfig):
        self.config = config
        self.layout_collater = StltCollater(config)
        self.appearance_collater = AppearanceCollater(config)

    def __call__(self, batch):
        # Layout batch
        layout_batch = [e["layout"] for e in batch]
        layout_batch = self.layout_collater(layout_batch)
        # Appearance batch
        appearance_batch = [e["appearance"] for e in batch]
        appearance_batch = self.appearance_collater(appearance_batch)
        # Combine batches
        joint_batch = {**layout_batch, **appearance_batch}

        return joint_batch


collaters_factory = {
    "appearance": AppearanceCollater,
    "layout": StltCollater,
    "multimodal": MultiModalCollater,
}
