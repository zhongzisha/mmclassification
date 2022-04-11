# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.image import tensor2imgs

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model



"""
CONFIG=vgg16_b32x2_ganta_with_tower_state
CONFIG=resnet50_b32x2_ganta_with_tower_state
for SUBSET in train val; do

python extract_features_and_classification.py \
/media/ubuntu/SSD/ganta_with_tower_state/${CONFIG}/${CONFIG}.py \
/media/ubuntu/SSD/ganta_with_tower_state/${CONFIG}/latest.pth  \
--metrics accuracy \
--out /media/ubuntu/SSD/ganta_ensemble/${CONFIG}/${SUBSET}_results.pkl \
--data_prefix data/ganta_with_tower_state/${SUBSET} \
--save_prefix ${SUBSET} \
--show-dir /media/ubuntu/SSD/ganta_ensemble/${CONFIG}/

done


CONFIG=mobilenet_v3_large_ganta_with_tower_state
SUBSET=train
python extract_features_and_classification.py \
/media/ubuntu/SSD/ganta_with_tower_state/${CONFIG}/${CONFIG}.py \
/media/ubuntu/SSD/ganta_with_tower_state/${CONFIG}/latest.pth \
--out /media/ubuntu/SSD/ganta_ensemble/${CONFIG}/${SUBSET}_results.pkl \
--data_prefix data/ganta_with_tower_state/${SUBSET} \
--save_prefix ${SUBSET} \
--show-dir /media/ubuntu/SSD/ganta_ensemble/${CONFIG}/

CONFIG=shufflenet_v2_1x_b64x16_linearlr_bn_nowd_ganta_with_tower_state
SUBSET=train
python extract_features_and_classification.py \
/media/ubuntu/SSD/ganta_with_tower_state/${CONFIG}/${CONFIG}.py \
/media/ubuntu/SSD/ganta_with_tower_state/${CONFIG}/latest.pth \
--out /media/ubuntu/SSD/ganta_ensemble/${CONFIG}/${SUBSET}_results.pkl \
--data_prefix data/ganta_with_tower_state/${SUBSET} \
--save_prefix ${SUBSET} \
--show-dir /media/ubuntu/SSD/ganta_ensemble/${CONFIG}/
"""

def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--data_prefix', default="", type=str)
    parser.add_argument('--save_prefix', default="", type=str)
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
             '"accuracy", "precision", "recall", "f1_score", "support" for single '
             'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
             'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be parsed as a dict metric_options for dataset.evaluate()'
             ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
             'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def single_gpu_test(args,
                    model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    **show_kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    feature_tensors = {}
    features_dict = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_tensors[name + '_feat'] = output.detach()
        return hook

    # import pdb
    # pdb.set_trace()

    config_filename = args.config.split(os.sep)[-1]

    if 'vgg16' in config_filename:
        model.module.backbone.features[-1].register_forward_hook(get_activation('conv5'))
        model.module.backbone.classifier[2].register_forward_hook(get_activation('fc1'))
        model.module.backbone.classifier[5].register_forward_hook(get_activation('fc2'))

        features_dict = {
            'conv5_feat': [],
            'fc1_feat': [],
            'fc2_feat': []
        }
    elif 'resnet18' in config_filename:
        pass
    elif 'resnet34' in config_filename:
        pass
    elif 'resnet50' in config_filename:

        model.module.neck.gap.register_forward_hook(get_activation('gap'))
        features_dict = {
            'gap_feat': [],
        }

        pass
    elif 'resnet101' in config_filename:
        pass
    elif 'resnext50' in config_filename:
        pass
    elif 'regnetx_3.2gf' in config_filename:
        pass
    elif 'seresnet50' in config_filename:
        pass
    elif 'seresnext50' in config_filename:
        pass
    elif 'mobilenet_v2' in config_filename:
        pass
    elif 'mobilenet_v3' in config_filename:

        model.module.neck.gap.register_forward_hook(get_activation('gap'))
        features_dict = {'gap_feat': []}

        pass
    elif 'shufflenet_v1' in config_filename:
        pass
    elif 'shufflenet_v2' in config_filename:

        model.module.neck.gap.register_forward_hook(get_activation('gap'))
        features_dict = {'gap_feat': []}

        pass
    else:
        print('has an error')
        import pdb
        pdb.set_trace()

    # pdb.set_trace()

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        for key in features_dict.keys():
            features_dict[key].append(feature_tensors[key].cpu().numpy().astype(np.float32))

        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [model.CLASSES[lb] for lb in pred_label]

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[i],
                    'pred_label': pred_label[i],
                    'pred_class': pred_class[i]
                }
                model.module.show_result(
                    img_show,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    for key in features_dict.keys():
        # np.savez(os.path.join(out_dir, '%s_%s.npz' % (args.save_prefix, key)), np.concatenate(features_dict[key]))
        filename = os.path.join(out_dir, '%s_%s.npz' % (args.save_prefix, key))
        with open(filename, 'wb') as fp:
            pickle.dump(np.concatenate(features_dict[key]), fp)

    # pdb.set_trace()

    return results


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.data_prefix != "":
        cfg.data.test.data_prefix = args.data_prefix

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    # if not distributed:
    #     if args.device == 'cpu':
    #         model = model.cpu()
    #     else:
    #         model = MMDataParallel(model, device_ids=[0])
    #     model.CLASSES = CLASSES
    #     show_kwargs = {} if args.show_options is None else args.show_options
    #     outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
    #                               **show_kwargs)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.CLASSES = CLASSES

    show_kwargs = {} if args.show_options is None else args.show_options
    outputs = single_gpu_test(args, model, data_loader, args.show, args.show_dir,
                              **show_kwargs)

    # import pdb
    # pdb.set_trace()

    rank, _ = get_dist_info()
    if rank == 0:

        results = {}
        if args.metrics:
            eval_results = dataset.evaluate(outputs, args.metrics,
                                            args.metric_options)
            results.update(eval_results)
            for k, v in eval_results.items():
                print(f'\n{k} : {v:.2f}')
        if args.out:
            # save gt_labels
            gt_labels = dataset.get_gt_labels()
            with open(args.out.replace('_results.pkl', '_gt_labels.npz'), 'wb') as fp:
                pickle.dump(gt_labels, fp)

            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [CLASSES[lb] for lb in pred_label]
            results.update({
                'class_scores': scores,
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            })
            print(f'\ndumping results to {args.out}')
            mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
