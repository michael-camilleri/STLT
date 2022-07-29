# Stores the data and model configs
import json


class DataConfig:
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        labels_path: str,
        videos_path: str,
        train: bool,
        **kwargs,
    ):
        assert (dataset_name in ("something", "action_genome", "mouse")), f"{dataset_name} does not exist!"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.videos_path = videos_path
        self.train = train
        self.layout_samples = kwargs.pop("layout_samples", 12)
        self.layout_num_frames = self.layout_samples * 2 + 1
        self.layout_stride = kwargs.pop("layout_stride", 1)
        self.appearance_samples = kwargs.pop("appearance_samples", 12)
        self.appearance_num_frames = self.appearance_samples * 2 + 1
        self.appearance_stride = kwargs.pop("appearance_stride", 1)
        self.max_num_objects = kwargs.pop("max_num_objects", 7)
        self.score_threshold = kwargs.pop("score_threshold", 0.5)
        self.normaliser_means = kwargs.pop("normaliser_mean", (0.5, 0.5, 0.5))
        self.normaliser_stds = kwargs.pop("normaliser_std", (0.5, 0.5, 0.5))
        self.min_scale = kwargs.pop("spatial_size", 112)
        self.crop_scale = kwargs.pop("crop_scale", 1.0)
        _bbscale = kwargs.pop("bbox_scale", 128)
        self.bbox_scale = (_bbscale, _bbscale)
        self.size_jitter = kwargs.pop("size_jitter", (0, 0))
        self.bbox_pad = kwargs.pop("bbox_pad", 50),
        self.debug_size = kwargs.pop("debug_size", None)
        self.maintain_identities = kwargs.pop("maintain_identities", False)
        self.include_hopper = kwargs.pop("include_hopper", False)
        self.video_size = [float(vs) for vs in kwargs.pop("video_size", ["1280", "720"])]

        _schema = json.load(open(labels_path))
        self.labels = _schema['labels']
        if self.dataset_name == "mouse":
            self.categories = {'pad': 0, 'mouse': 1, 'cagemate_1': 2}
            _next_id = 3
            if self.maintain_identities:
                print('Including IDs')
                self.categories['cagemate_2'] = _next_id
                _next_id += 1
            else:
                self.categories['cagemate_2'] = _next_id - 1
            if self.include_hopper:
                print('Including Hopper')
                self.categories['hopper'] = _next_id
                _next_id += 1
            self.categories['cls'] = _next_id
            self.frame2type = {"pad": 0, "start": 1, "regular": 2, "empty": 3, "extract": 4}
        elif self.dataset_name == "something":
            self.categories = {'pad': 0, **_schema['categories'], 'cls': len(_schema['categories']) + 1}
            self.frame2type = {"pad": 0, "start": 1, "regular": 2, "empty": 3, "extract": 4}
        else:
            self.categories = {'pad': 0, 'cls': 1, **_schema['categories']}
            self.frame2type = {"pad": 0, "regular": 1, "extract": 2, "empty": 3}
        self.unique_categories = len(set(self.categories.values()))


class GeneralModelConfig:
    def __init__(self, **kwargs):
        self.num_classes = kwargs.pop("num_classes", None)
        assert self.num_classes, "num_classes must not be None!"
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.hidden_dropout_prob = kwargs.pop("hidden_dropout_prob", 0.1)
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-12)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 12)


class StltModelConfig(GeneralModelConfig):
    def __init__(self, **kwargs):
        super(StltModelConfig, self).__init__(**kwargs)
        self.unique_categories = kwargs.pop("unique_categories", None)
        assert self.unique_categories, "unique_categories must not be None!"
        self.num_spatial_layers = kwargs.pop("num_spatial_layers", 4)
        self.num_temporal_layers = kwargs.pop("num_temporal_layers", 8)
        self.layout_num_frames = kwargs.pop("layout_num_frames", 256)
        self.load_backbone_path = kwargs.pop("load_backbone_path", None)
        self.freeze_backbone = kwargs.pop("freeze_backbone", False)

    def __repr__(self):
        return (
            f"- Unique categories: {self.unique_categories}\n"
            f"- Number of classes: {self.num_classes}\n"
            f"- Hidden size: {self.hidden_size}\n"
            f"- Hidden dropout probability: {self.hidden_dropout_prob}\n"
            f"- Layer normalization epsilon: {self.layer_norm_eps}\n"
            f"- Number of attention heads: {self.num_attention_heads}\n"
            f"- Number of spatial layers: {self.num_spatial_layers}\n"
            f"- Number of temporal layers: {self.num_temporal_layers}\n"
            f"- Max number of layout frames: {self.layout_num_frames}\n"
            f"- The backbone path is: {self.load_backbone_path}\n"
            f"- Freezing the backbone: {self.freeze_backbone}"
        )


class AppearanceModelConfig(GeneralModelConfig):
    def __init__(self, **kwargs):
        super(AppearanceModelConfig, self).__init__(**kwargs)
        self.appearance_num_frames = kwargs.pop("appearance_num_frames", None)
        assert self.appearance_num_frames, "appearance_num_frames must not be None!"
        self.resnet_model_path = kwargs.pop("resnet_model_path", None)
        assert self.resnet_model_path, "resnet_model_path must be provided"
        self.num_appearance_layers = kwargs.pop("num_appearance_layers", 4)
        self.spatial_size = kwargs.pop("spatial_size", 112)

    def __repr__(self):
        return (
            f"- Number of classes: {self.num_classes}\n"
            f"- Max number of appearance frames: {self.appearance_num_frames}\n"
            f"- Hidden size: {self.hidden_size}\n"
            f"- If Transformer: Number of attention heads: {self.num_attention_heads}\n"
            f"- If Transformer: Number of layers: {self.num_appearance_layers}\n"
            f"- If Transformer: Hidden dropout probability: {self.hidden_dropout_prob}\n"
            f"- If Transformer: Layer norm eps: {self.layer_norm_eps}"
        )


class MultimodalModelConfig(GeneralModelConfig):
    def __init__(self, **kwargs):
        super(MultimodalModelConfig, self).__init__(**kwargs)
        # Not perfect way of creating the configs...
        self.stlt_config = StltModelConfig(**kwargs)
        self.appearance_config = AppearanceModelConfig(**kwargs)
        self.num_fusion_layers = kwargs.pop("num_fusion_layers", 4)
        self.load_backbone_path = kwargs.pop("load_backbone_path", None)
        self.freeze_backbone = kwargs.pop("freeze_backbone", False)

    def __repr__(self):
        stlt_repr = "*** Layout branch config: \n" + self.stlt_config.__repr__() + "\n"
        appearance_repr = (
            "*** Appearance branch config: \n"
            + self.appearance_config.__repr__()
            + "\n"
        )

        return (
            stlt_repr
            + appearance_repr
            + "*** Fusion module config: \n"
            + f"- Num fusion layers: {self.num_fusion_layers}\n"
            + f"- The backbone path is: {self.load_backbone_path}\n"
            + f"- Freezing the backbone: {self.freeze_backbone}"
        )


model_configs_factory = {
    "stlt": StltModelConfig,
    "resnet3d": AppearanceModelConfig,
    "resnet3d-transformer": AppearanceModelConfig,
    "lcf": MultimodalModelConfig,
    "caf": MultimodalModelConfig,
    "cacnf": MultimodalModelConfig,
}
