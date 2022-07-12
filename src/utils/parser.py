import argparse


class Parser:
    def __init__(self, parser_description: str):
        self.parser = argparse.ArgumentParser(description=parser_description)
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="The name of the dataset, either something or action_genome",
        )
        self.parser.add_argument(
            "--dataset_type",
            type=str,
            default=None,
            help="The type of the dataset - layout, appearance or multimodal.",
        )
        self.parser.add_argument(
            "--log_filepath",
            type=str,
            default=None,
            help="Where to log the progress.",
        )
        self.parser.add_argument(
            "--train_dataset_path",
            type=str,
            default=None,
            help="Path to the train dataset.",
        )
        self.parser.add_argument(
            "--val_dataset_path",
            type=str,
            default=None,
            help="Path to the val dataset.",
        )
        self.parser.add_argument(
            "--test_dataset_path",
            type=str,
            default=None,
            help="Path to the test dataset.",
        )
        self.parser.add_argument(
            "--labels_path",
            type=str,
            default=None,
            help="Path to the labels.",
        )
        self.parser.add_argument(
            "--videos_path",
            type=str,
            default=None,
            help="Path to the videos saved as HDF5.",
        )
        self.parser.add_argument(
            "--layout_samples",
            type=int,
            default=12,
            help="The number of layout frames to sample on either side of the centre-frame: the "
                 "total number of frames will thus be 1 + 2N.",
        )
        self.parser.add_argument(
            "--layout_stride",
            type=int,
            default=1,
            help="The stride to use when sampling frames on either side of the centre-frame. A "
                 "stride of 1 indicates sampling every frame."
        )
        self.parser.add_argument(
            "--appearance_samples",
            type=int,
            default=12,
            help="The number of appearance frames to sample on either side of the centre-frame: "
                 "the total number of frames will thus be 1 + 2N."
        )
        self.parser.add_argument(
            "--appearance_stride",
            type=int,
            default=1,
            help="The stride to use when sampling frames on either side of the centre-frame. A "
                 "stride of 1 indicates sampling every frame."
        )
        self.parser.add_argument(
            "--score_threshold",
            type=float,
            default=0.5,
            help="The score threshold for the categories.",
        )
        self.parser.add_argument(
            "--num_spatial_layers",
            type=int,
            default=4,
            help="The number of spatial transformer layers.",
        )
        self.parser.add_argument(
            "--num_temporal_layers",
            type=int,
            default=8,
            help="The number of temporal transformer layers.",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="The batch size.",
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=5e-5,
            help="The learning rate.",
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-3,
            help="The weight decay.",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="The number of processor workers.",
        )
        self.parser.add_argument(
            "--clip_val",
            type=float,
            default=5.0,
            help="The gradient clipping value.",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=20,
            help="The number of epochs to train the model.",
        )
        self.parser.add_argument(
            "--warmup_epochs",
            type=int,
            default=2,
            help="The number warmup epochs.",
        )
        self.parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="The name of the model.",
        )
        self.parser.add_argument(
            "--resnet_model_path",
            type=str,
            default=None,
            help="Path to the pre-trained ResNet3D.",
        )
        self.parser.add_argument(
            "--save_model_path",
            type=str,
            default="models/best.pt",
            help="Where to save the model.",
        )
        self.parser.add_argument(
            "--save_backbone_path",
            type=str,
            default=None,
            help="Where to save the STLT backbone.",
        )
        self.parser.add_argument(
            "--load_backbone_path",
            type=str,
            default=None,
            help="From where to load the STLT backbone.",
        )
        self.parser.add_argument(
            "--freeze_backbone",
            action="store_true",
            help="Whether to freeze the backbone.",
        )
        self.parser.add_argument(
            "--features_path",
            type=str,
            default=None,
            help="Whether to use video features.",
        )
        self.parser.add_argument(
            "--checkpoint_path",
            type=str,
            default="models/best.pt",
            help="Checkpoint to a trained model.",
        )
        self.parser.add_argument(
            "--num_appearance_layers",
            type=int,
            default=4,
            help="Number of Appearance Layers (Resnet)"
        )
        self.parser.add_argument(
            "--num_fusion_layers",
            type=int,
            default=4,
            help="Number of Fusion Layers"
        )
        self.parser.add_argument(
            "--hidden_size",
            type=int,
            default=768,
            help="Default/Base Hidden dimension size"
        )
        self.parser.add_argument(
            "--maintain_identities",
            action="store_true",
            default=False,
            help="Whether to maintain identities in Mouse Dataset"
        )
        self.parser.add_argument(
            "--include_hopper",
            action="store_true",
            default=False,
            help="Whether to include the Hopper location as a cage structure"
        )
        self.parser.add_argument(
            "--normaliser_mean",
            type=float,
            nargs=3,
            help="The normalisation Means (for images)"
        )
        self.parser.add_argument(
            "--normaliser_std",
            type=float,
            nargs=3,
            help="The normalisation standard deviations (for images)"
        )
        self.parser.add_argument(
            "--resize_height",
            type=int,
            default=112,
            help="Spatial Size to resize frames to (height)"
        )
        self.parser.add_argument(
            "--crop_scale",
            type=float,
            default=1.15,
            help="Scaling prior to cropping"
        )
        self.parser.add_argument(
            "--bbox_scale",
            type=int,
            default=128,
            help="Resize BBoxes to this size (width and height)"
        )
        self.parser.add_argument(
            "--size_jitter",
            type=int,
            nargs=2,
            default=(0, 0),
            help="Range of values (pixels) to add to BBox co-ordinates"
        )
        self.parser.add_argument(
            "--output_path",
            type=str,
            default=None,
            help="Path for storing Inference Classifications: this is a CSV file, in which the "
                 "first column is the video-id and the subsequent columns the logits for each "
                 "class. Note that the extension will be added automatically on a per-logit basis.",
        )
        self.parser.add_argument(
            "--which_score",
            type=str,
            default='average',
            help="On which model output to base the score: if average, based on average over all."
        )
        self.parser.add_argument(
            "--which_logits",
            type=str,
            nargs='+',
            default=('stlt',),
            help='For which models to output logits'
        )
        self.parser.add_argument(
            "--select_best",
            type=str,
            default='average',
            help="How to select the best model through the epochs: average, top1 or top5."
        )
        self.parser.add_argument(
            "--debug_size",
            type=int,
            default=None,
            help="For Debugging, size of datasets to operate on"
        )
        self.parser.add_argument(
            "--video_size",
            type=int,
            nargs=2,
            default=[1280, 720],
            help="Video-Size (HxW)"
        )

    def parse_args(self):
        return self.parser.parse_args()
