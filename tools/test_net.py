#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import os
import pickle
import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter, EPICTestMeter

logger = logging.get_logger(__name__)


def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    N = len(test_loader) * cfg.TEST.BATCH_SIZE
    NUM_VERBS = 97
    NUM_NOUNS = 300
    N_SHARED_FEAT = 2304
    narration_ids = []
    true_verbs = np.zeros(N, dtype=int) - 1
    true_nouns = np.zeros(N, dtype=int) - 1
    verb_probs = np.zeros((N, NUM_VERBS))
    noun_probs = np.zeros((N, NUM_NOUNS))

    verb_feats = np.zeros((N, NUM_VERBS))
    noun_feats = np.zeros((N, NUM_NOUNS))
    shared_feats = np.zeros((N, N_SHARED_FEAT))

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        video_idx = video_idx.cuda()

        if cfg.DETECTION.ENABLE:
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach().cpu(),
                ori_boxes.detach().cpu(),
                metadata.detach().cpu(),
            )
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds = model(inputs, ret_extra=True)
            n_id = meta['narration_id']
            start_idx = cur_iter * cfg.TEST.BATCH_SIZE
            end_idx = start_idx + len(n_id)

            narration_ids += n_id
            true_verbs[start_idx:end_idx] = labels['verb'].cpu().numpy()
            true_nouns[start_idx:end_idx] = labels['noun'].cpu().numpy()
            verb_probs[start_idx:end_idx] = preds[0].detach().cpu().numpy()
            noun_probs[start_idx:end_idx] = preds[1].detach().cpu().numpy()

            verb_feats[start_idx:end_idx] = preds[2].mean((1,2,3)).detach().cpu().numpy()
            noun_feats[start_idx:end_idx] = preds[3].mean((1,2,3)).detach().cpu().numpy()
            shared_feats[start_idx:end_idx] = preds[4].mean((1,2,3)).detach().cpu().numpy()

            if isinstance(labels, (dict,)):
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    verb_preds, verb_labels, video_idx = du.all_gather(
                        [preds[0], labels['verb'], video_idx]
                    )

                    noun_preds, noun_labels, video_idx = du.all_gather(
                        [preds[1], labels['noun'], video_idx]
                    )
                    meta = du.all_gather_unaligned(meta)
                    metadata = {'narration_id': []}
                    for i in range(len(meta)):
                        metadata['narration_id'].extend(meta[i]['narration_id'])
                else:
                    metadata = meta
                    verb_preds, verb_labels, video_idx = preds[0], labels['verb'], video_idx
                    noun_preds, noun_labels, video_idx = preds[1], labels['noun'], video_idx
                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    (verb_preds.detach().cpu(), noun_preds.detach().cpu()),
                    (verb_labels.detach().cpu(), noun_labels.detach().cpu()),
                    metadata,
                    video_idx.detach().cpu(),
                )
                test_meter.log_iter_stats(cur_iter)
            else:
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    preds, labels, idx = du.all_gather(
                        [preds, labels, video_idx]
                    )

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach().cpu(),
                    labels.detach().cpu(),
                    video_idx.detach().cpu(),
                )
                test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # save resulting data
    data = dict(narration_ids=narration_ids,
        true_verbs=true_verbs, true_nouns=true_nouns,
        verb_probs=verb_probs, noun_probs=noun_probs,
        verb_feats=verb_feats, noun_feats=noun_feats, shared_feats=shared_feats)
    
    with open(os.path.join(cfg.OUTPUT_DIR, 'slow_fast_results_feat.pickle'), 'wb') as f:
        pickle.dump(data, f)

    # Log epoch stats and print the final testing results.
    if cfg.TEST.DATASET == 'epickitchens':
        preds, labels, metadata = test_meter.finalize_metrics()
    else:
        test_meter.finalize_metrics()
        preds, labels, metadata = None, None, None
    test_meter.reset()
    return preds, labels, metadata

def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model, cfg, is_train=False)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        if cfg.TEST.DATASET == 'epickitchens':
            test_meter = EPICTestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
            )
        else:
            test_meter = TestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
            )

    # # Perform multi-view test on the entire dataset.
    preds, labels, metadata = perform_test(test_loader, model, test_meter, cfg)

    if du.is_master_proc():
        if cfg.TEST.DATASET == 'epickitchens':
            results = {'verb_output': preds[0],
                       'noun_output': preds[1],
                       'narration_id': metadata}
            scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            file_path = os.path.join(scores_path, cfg.EPICKITCHENS.TEST_SPLIT + '.pkl')
            pickle.dump(results, open(file_path, 'wb'))
