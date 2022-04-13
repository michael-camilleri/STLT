import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelling.datasets import collaters_factory, datasets_factory
from modelling.configs import DataConfig, model_configs_factory
from modelling.models import models_factory
from utils.evaluation import evaluators_factory
from utils.parser import Parser
from utils.train_inference_utils import get_device, move_batch_to_device

import pandas as pd


@torch.no_grad()
def inference(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = get_device(logger=logging.getLogger(__name__))
    logging.info("Preparing dataset...")
    data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.test_dataset_path,
        labels_path=args.labels_path,
        videoid2size_path=args.videoid2size_path,
        layout_num_frames=args.layout_num_frames,
        appearance_num_frames=args.appearance_num_frames,
        videos_path=args.videos_path,
        normaliser_mean=args.normaliser_mean,
        normaliser_std=args.normaliser_std,
        videos_as_frames=args.videos_as_frames,
        spatial_size=args.resize_height,
        crop_scale=args.crop_scale,
        debug_size=args.debug_size,
        train=False,
    )
    test_dataset = datasets_factory[args.dataset_type](data_config)
    num_samples = len(test_dataset)
    logging.info(f"Inference on {num_samples}")
    collater = collaters_factory[args.dataset_type](data_config)
    # Prepare loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    logging.info("Preparing model...")
    # Prepare model
    num_classes = len(test_dataset.labels)
    model_config = model_configs_factory[args.model_name](
        num_classes=num_classes,
        appearance_num_frames=args.appearance_num_frames,
        spatial_size=args.resize_height,
        unique_categories=len(data_config.categories), # len(data_config.category2id),
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
        layout_num_frames=args.layout_num_frames,  # Added this to also pass in num_frames.
        resnet_model_path=args.resnet_model_path,
        num_appearance_layers=args.num_appearance_layers,
        num_fusion_layers=args.num_fusion_layers,
        hidden_size=args.hidden_size,
    )
    logging.info("==================================")
    logging.info(f"The model's configuration is:\n{model_config}")
    logging.info("==================================")
    # clean up to make better use of memory
    torch.cuda.empty_cache()
    model = models_factory[args.model_name](model_config).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    except RuntimeError as e:
        logging.warning(
            "Default loading failed, loading with strict=False. If it's only "
            "score_embedding modules it's ok. Otherwise see exception below"
        )
        logging.warning(e)
        model.load_state_dict(
            torch.load(args.checkpoint_path, map_location=device), strict=False
        )
    model.train(False)
    logging.info("Starting inference...")
    evaluator = evaluators_factory[args.dataset_name](num_samples, num_classes, model.logit_names)
    # Handle Output Logits
    output = {l: {} for l in args.which_logits} if args.output_path is not None else None
    for batch in tqdm(test_loader):
        logits = model(move_batch_to_device(batch, device))
        evaluator.process(logits, batch["labels"])
        if args.output_path is not None:
            for lg in args.which_logits:
                output[lg].update(
                    {v: l.cpu().numpy() for v, l in zip(batch['video_id'], logits[lg])}
                )

    metrics = evaluator.evaluate()
    logging.info("=================================")
    logging.info("The metrics are:")
    for m, scores in metrics.items():
        for l, s in scores.items():
            logging.info(f"{m}/{l}: {round(s * 100, 2)}")
    logging.info("=================================")

    if args.output_path is not None:
        for lg, output_logits in output.items():
            pd.DataFrame(output_logits).T.to_csv(args.output_path + f'.{lg}.csv', header=False)


def main():
    parser = Parser("Inference with a model, currenly STLT, LCF, CAF, and CACNF.")
    inference(parser.parse_args())


if __name__ == "__main__":
    main()
