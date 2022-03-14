import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from layout_models.datasets import StltCollater, StltDataConfig, StltDataset
from layout_models.modelling import Stlt, StltModelConfig
from utils.data_utils import get_device
from utils.evaluation import Evaluator
from utils.parser import Parser

import pandas as pd

@torch.no_grad()
def inference(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = get_device(logger=logging.getLogger(__name__))
    # Prepare datasets
    logging.info("Preparing datasets...")
    data_config = StltDataConfig(
        dataset_path=args.test_dataset_path,
        labels_path=args.labels_path,
        videoid2size_path=args.videoid2size_path,
        num_frames=args.layout_num_frames,
        train=False,
    )
    test_dataset = StltDataset(data_config)
    logging.info(f"Inference on {len(test_dataset)}")
    collater = StltCollater(data_config)
    # Prepare loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    logging.info("Preparing model...")
    # Prepare model
    model_config = StltModelConfig(
        num_classes=len(test_dataset.labels),
        unique_categories=len(data_config.categories),
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
    )
    logging.info("==================================")
    logging.info(f"The model's configuration is:\n{model_config}")
    logging.info("==================================")
    model = Stlt(model_config).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.train(False)
    logging.info("Starting inference...")
    # Validation loop
    evaluator = Evaluator(len(test_dataset))
    output_logits = {} if args.output_path is not None else None
    for inputs, labels, videos in tqdm(test_loader):
        # Move tensors to device
        inputs = {key: val.to(device) for key, val in inputs.items()}
        logits = model(inputs)
        evaluator.process(logits, labels)
        if args.output_path is not None:
            output_logits.update({vid: lg.cpu().numpy() for vid, lg in zip(videos, logits)})

    top1_accuracy, top5_accuracy = evaluator.evaluate()
    logging.info("=================================")
    logging.info(
        f"{round(top1_accuracy * 100, 2)}% (TOP 1), "
        f"{round(top5_accuracy * 100, 2)}% (TOP 5)."
    )
    logging.info("=================================")

    if args.output_path is not None:
        pd.DataFrame(output_logits).T.to_csv(args.output_path, header=False)


def main():
    parser = Parser("Inference with STLT model.")
    inference(parser.parse_args())


if __name__ == "__main__":
    main()
