from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot

import glob,os
import time
import random
import copy
import mmcv
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import cv2
from osgeo import gdal, osr
from natsort import natsorted
from pathlib import Path
import json
import psutil
from yoloV5.myutils import load_gt_from_txt, load_gt_from_esri_xml, py_cpu_nms, \
    compute_offsets, save_predictions_to_envi_xml, LoadImages, \
    box_iou, ap_per_class, ConfusionMatrix
from yoloV5.utils.torch_utils import select_device, time_synchronized


"""
export PYTHONPATH=/media/ubuntu/Data/gd/:$PYTHONPATH
python demo/detect_gd_line.py \
--source /media/ubuntu/Data/gd_1024_aug_90_newSplit_4classes/val/val_list.txt \
--checkpoint work_dirs/resnet34_b16x8_gd_line/latest.pth \
--config configs/resnet/resnet34_b16x8_gd_line.py \
--img-size 224 --gap 32
"""

def main():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--cls-weights', type=str, default='', help='cls_model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--gt-xml-dir', type=str, default='', help='gt xml dir')
    parser.add_argument('--gt-prefix', type=str, default='', help='gt prefix')
    parser.add_argument('--gt-subsize', type=int, default=5120, help='train image size for labeling')
    parser.add_argument('--gt-gap', type=int, default=128, help='train gap size for labeling')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--big-subsize', type=int, default=51200, help='inference big-subsize (pixels)')
    parser.add_argument('--gap', type=int, default=128, help='overlap size')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')

    parser.add_argument('--score-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--hw-thres', type=float, default=5, help='height or width threshold for box')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    args = parser.parse_args()

    source, view_img, save_txt, imgsz, gap, \
    gt_xml_dir, gt_prefix, gt_subsize, gt_gap, \
    big_subsize, batchsize, score_thr, hw_thr = \
        args.source, args.view_img, args.save_txt, args.img_size, args.gap, \
        args.gt_xml_dir, args.gt_prefix, int(args.gt_subsize), int(args.gt_gap), args.big_subsize, \
        args.batchsize, args.score_thres, args.hw_thres

    # Directories
    save_dir = Path(args.project)  # increment run
    if not os.path.exists(save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)

    names = {0: 'nonline', 1: 'line'}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    shown_labels = [0, 1]  # 只显示中大型杆塔和绝缘子

    device = select_device(args.device)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    stride = 32

    tiffiles = None
    if os.path.isfile(source) and source[-4:] == '.txt':
        with open(source, 'r', encoding='utf-8-sig') as fp:
            tiffiles = [line.strip() for line in fp.readlines()]
    else:
        tiffiles = natsorted(glob.glob(source + '/*.tif'))
    print(tiffiles)

    seen = 0
    nc = len(names)
    inst_count = 1

    mean = np.array([125.307, 122.961, 113.8575], dtype=np.float32).reshape([1,3,1,1])
    std = np.array([51.5865, 50.847, 51.255], dtype=np.float32).reshape([1,3,1,1])

    for ti in range(len(tiffiles)):
        image_id = ti + 1
        tiffile = tiffiles[ti]
        file_prefix = tiffile.split(os.sep)[-1].replace('.tif', '')

        print(ti, '=' * 80)
        print(file_prefix)

        ds = gdal.Open(tiffile, gdal.GA_ReadOnly)
        print("Driver: {}/{}".format(ds.GetDriver().ShortName,
                                     ds.GetDriver().LongName))
        print("Size is {} x {} x {}".format(ds.RasterXSize,
                                            ds.RasterYSize,
                                            ds.RasterCount))
        print("Projection is {}".format(ds.GetProjection()))
        projection = ds.GetProjection()
        projection_sr = osr.SpatialReference(wkt=projection)
        projection_esri = projection_sr.ExportToWkt(["FORMAT=WKT1_ESRI"])
        geotransform = ds.GetGeoTransform()
        xOrigin = geotransform[0]
        yOrigin = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        orig_height, orig_width = ds.RasterYSize, ds.RasterXSize
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
            print("IsNorth = ({}, {})".format(geotransform[2], geotransform[4]))

        # 先计算可用内存，如果可以放得下，就不用分块了
        avaialble_mem_bytes = psutil.virtual_memory().available
        if orig_width * orig_height * ds.RasterCount < 0.8 * avaialble_mem_bytes:
            offsets = [[0, 0, orig_width, orig_height]]
        else:
            # 根据big_subsize计算子块的起始偏移
            offsets = compute_offsets(height=orig_height, width=orig_width, subsize=big_subsize, gap=2 * gt_gap)
        print('offsets: ', offsets)

        final_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        all_preds_filename = str(save_dir) + '/' + file_prefix + '_all_preds.pt'

        if True:

            all_preds = []
            for oi, (xoffset, yoffset, sub_width, sub_height) in enumerate(offsets):  # left, up

                print('processing sub image %d' % oi, xoffset, yoffset, sub_width, sub_height)
                dataset = LoadImages(gdal_ds=ds, xoffset=xoffset, yoffset=yoffset,
                                     width=sub_width, height=sub_height,
                                     batchsize=batchsize, subsize=imgsz, gap=gap, stride=stride,
                                     return_list=False, is_nchw=True)
                if len(dataset) == 0:
                    continue

                print('forward inference')
                sub_preds = []
                for img in dataset:
                    # img: BS x 3 x 224 x 224
                    img = img.astype(np.float32)
                    img -= mean
                    img /= std
                    img = torch.from_numpy(img)
                    img = img.to(device)

                    data={'img':img, 'img_metas':{}}
                    result = model(return_loss=False, **data)
                    pred_label = np.argmax(result, axis=1)

                    sub_preds.append(np.stack(pred_label))
                # import pdb
                # pdb.set_trace()
                sub_preds = np.concatenate(sub_preds)
                # pdb.set_trace()
                inds = np.where(sub_preds==1)[0]
                # pdb.set_trace()
                if len(inds) > 0:
                    for ind in inds:
                        x, y = dataset.start_positions[ind]
                        y1 = y+yoffset
                        y2 = y1 + 224
                        x1 = x+xoffset
                        x2 = x1 + 224
                        final_mask[y1:y2, x1:x2] = 255

                # pdb.set_trace()
                del dataset.img0
                del dataset
                import gc
                gc.collect()
        mask_savefilename = file_prefix+"_LineCls_result.png"
        # cv2.imwrite(mask_savefilename, mask)
        cv2.imencode('.png', final_mask)[1].tofile(mask_savefilename)



if __name__ == '__main__':
    main()
