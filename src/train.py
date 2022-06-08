import logging
import os.path as op

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelling.datasets import collaters_factory, datasets_factory
from modelling.configs import model_configs_factory, DataConfig
from modelling.models import models_factory
from utils.evaluation import evaluators_factory
from utils.parser import Parser
from utils.train_inference_utils import add_weight_decay, get_device, \
    get_linear_schedule_with_warmup, move_batch_to_device, Criterion


def train(args):
    if args.log_filepath:
        # Set up logging
        if op.exists(args.log_filepath):
            raise ValueError(f"There is a log at {args.log_filepath}!")
        logging.basicConfig(
            level=logging.INFO, filename=args.log_filepath, filemode="w"
        )
    else:
        logging.basicConfig(level=logging.INFO)
    # Check for CUDA
    device = get_device(logger=logging.getLogger(__name__))
    # Prepare datasets
    logging.info("Preparing datasets...")
    if (args.dataset_name == "mouse") and (args.dataset_type == "layout"):
        ds_type = "mouse"
    else:
        ds_type = args.dataset_type
    # Prepare train dataset
    logging.info("  -> Training DataSet Config")
    train_data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.train_dataset_path,
        labels_path=args.labels_path,
        video_size=args.video_size,
        layout_samples=args.layout_samples,
        layout_stride=args.layout_stride,
        appearance_samples=args.appearance_samples,
        appearance_stride=args.appearance_stride,
        videos_path=args.videos_path,
        normaliser_mean=args.normaliser_mean,
        normaliser_std=args.normaliser_std,
        maintain_identities=args.maintain_identities,
        include_hopper=args.include_hopper,
        spatial_size=args.resize_height,
        crop_scale=args.crop_scale,
        debug_size=args.debug_size,
        train=True,
    )
    logging.info("  -> Loading Training DataSet")
    train_dataset = datasets_factory[ds_type](train_data_config)
    num_training_samples = len(train_dataset)
    logging.info(f"     (Training on {num_training_samples})")
    # Prepare validation dataset
    logging.info("  -> Validation DataSet Config")
    val_data_config = DataConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.val_dataset_path,
        labels_path=args.labels_path,
        video_size=args.video_size,
        layout_samples=args.layout_samples,
        layout_stride=args.layout_stride,
        appearance_samples=args.appearance_samples,
        appearance_stride=args.appearance_stride,
        videos_path=args.videos_path,
        normaliser_mean=args.normaliser_mean,
        normaliser_std=args.normaliser_std,
        maintain_identities=args.maintain_identities,
        include_hopper=args.include_hopper,
        spatial_size=args.resize_height,
        crop_scale=args.crop_scale,
        debug_size=args.debug_size,
        train=False,
    )
    logging.info("  -> Loading Validation DataSet")
    val_dataset = datasets_factory[ds_type](val_data_config)
    num_validation_samples = len(val_dataset)
    num_classes = len(val_dataset.labels)
    logging.info(f"     (Validating on {num_validation_samples})")
    # Prepare collaters
    train_collater = collaters_factory[args.dataset_type](train_data_config)
    val_collater = collaters_factory[args.dataset_type](val_data_config)
    # Prepare loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=val_collater,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers else False,
    )
    logging.info("Preparing model...")
    # Prepare model
    model_config = model_configs_factory[args.model_name](
        num_classes=num_classes,
        appearance_num_frames=val_data_config.appearance_num_frames,
        layout_num_frames=val_data_config.layout_num_frames,
        spatial_size=args.resize_height,
        unique_categories=val_data_config.unique_categories,
        num_spatial_layers=args.num_spatial_layers,
        num_temporal_layers=args.num_temporal_layers,
        resnet_model_path=args.resnet_model_path,
        num_appearance_layers=args.num_appearance_layers,
        num_fusion_layers=args.num_fusion_layers,
        hidden_size=args.hidden_size,
        load_backbone_path=args.load_backbone_path,
        freeze_backbone=args.freeze_backbone,
    )
    logging.info("==================================")
    logging.info(f"The model's configuration is:\n{model_config}")
    logging.info("==================================")
    model = models_factory[args.model_name](model_config).to(device)
    # Prepare loss and optimize
    criterion = Criterion(args.dataset_name)
    parameters = add_weight_decay(model, args.weight_decay)
    optimizer = optim.AdamW(parameters, lr=args.learning_rate)
    num_batches = len(train_dataset) // args.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_epochs * num_batches,
        num_training_steps=args.epochs * num_batches,
    )
    evaluator = evaluators_factory[args.dataset_name](
        num_validation_samples, num_classes, model.logit_names, args.which_score, args.select_best
    )
    logging.info("Starting training...")
    torch.cuda.empty_cache()
    for epoch in range(args.epochs):
        # Training loop
        model.train(True)
        for batch in tqdm(train_loader, miniters=50, maxinterval=60):
            # Remove past gradients
            optimizer.zero_grad()
            # Move tensors to device
            batch = move_batch_to_device(batch, device)
            # Obtain outputs
            logits = model(batch)
            # Measure loss and update weights
            loss = criterion(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_val)
            optimizer.step()
            # Update the scheduler
            scheduler.step()
            # Clean up
            del batch, logits
            torch.cuda.empty_cache() # Clear any unused cache items

        # Validation loop
        model.train(False)
        evaluator.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, miniters=50, maxinterval=60):
                batch = move_batch_to_device(batch, device)
                logits = model(batch)
                evaluator.process(logits, batch["labels"])
                del batch, logits
                torch.cuda.empty_cache()
        # Saving logic
        metrics = evaluator.evaluate()
        if evaluator.is_best():
            logging.info("=================================")
            logging.info(f"Found new best {evaluator.how_best} on epoch {epoch+1}!")
            logging.info("=================================")
            torch.save(model.state_dict(), args.save_model_path)
            if args.save_backbone_path:
                torch.save(model.backbone.state_dict(), args.save_backbone_path)
            with open(op.join(op.dirname(args.save_model_path), 'best_model.txt'), 'w') as f:
                f.write(f'Epoch: {epoch+1}\n')
                f.write(f'Top1: {metrics["top1"]["caf"]}')

        for m, scores in metrics.items():
            for l, s in scores.items():
                logging.info(f"{m}/{l}: {round(s * 100, 2)}")


def main():
    parser = Parser("Trains a model, currenly STLT, LCF, CAF, and CACNF.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
