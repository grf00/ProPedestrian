import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys
#sys.path.append( "/home/lhb/grf/" )
sys.path.append(".")
sys.path.append( "yolov7" )
sys.path.append( "fast_reid_master" )
sys.path.append( "yolov7/fast_reid_master" )
import argparse
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image
import glob
import os
import torchvision.transforms as T
import torch.nn.functional as F
import tqdm
import torchvision
from torch.backends import cudnn
from yolov7.fast_reid_master.fastreid.config import get_cfg
from yolov7.fast_reid_master.fastreid.utils.logger import setup_logger
from yolov7.fast_reid_master.fastreid.utils.file_io import PathManager
from yolov7.fast_reid_master.predictor import FeatureExtractionDemo

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#########################################################################################################
                    #####         fast—reid      ###############
cudnn.benchmark = True
setup_logger(name="fastreid")
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg
def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        default="yolov7/fast_reid_master/configs/Market1501/bagtricks_R50.yml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        default='./yolov7/fast_reid_master/datasets/query/*.jpg',
        #"E:/study/python/Study-myself/RE-ID/fast-reid-master/datasets/query/0001_c1s1_0_02.jpg",#0001_c1s1_0_02
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--dataset-name",  # 数据集名字
        default='Market1501',
        help="a test dataset name for visualizing ranking list."
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    #features = features.cpu().data.numpy()
    return features
def fast_detect(source,project):
    # gc.collect()
    # torch.cuda.empty_cache()
#########################################    yolov7参数设置   ################################################################
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7/weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=source, help='source')  # file/folder, 0 for webcam"./yolov7/inference/myself-detect/yolo-reid"
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt',default='true',action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', default='0',nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=project, help='save results to project/name')#'./runs/detect'
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default='true',action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--mask_hui', default='lama/test-image/', help='don`t trace model')
    parser.add_argument('--lama_pic', default='example/lama_s_pic/', help='don`t trace model')
    opt = parser.parse_args()
    dist_thres = 1.0
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

######################################################################################################################
    # ---------- 行人重识别模型初始化 --------------------------
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)  # 加载特征提取器，也就是加载模型
    query_feats = []  # 图像特征，用于保存每个行人的图像特征
    # 逐张保存读入行人图像，并保存相关信息
    if args.input:
        if PathManager.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input))
            assert args.input, "The input path(s) was not found"

        for path in tqdm.tqdm(args.input):
            img = cv2.imread(path)
            feat = demo.run_on_image(img)
            #feat = postprocess(feat)
            query_feats.append(feat)
    query_feats = torch.cat(query_feats, dim=0)
    query_feats = F.normalize(query_feats, p=2, dim=1)
#################################################################################################################################
#################################################################################################################################

    # --------------- 行人检测模型初始化 -------------------

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            img_size = im0.shape
            h0 = img_size[0]
            w0 = img_size[1]
            mask = np.zeros((h0, w0), dtype=np.uint8)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                gallery_img = []
                gallery_loc = []  # 这个列表用来存放框的坐标
                cls_l=[]
                gallery_feats = []
                # Write results
                for *xyxy, conf, cls in reversed(det):#一张图片的所有检测到的人
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    w = xmax - xmin
                    h = ymax - ymin
                    if w * h > 200:#太小的不行
                        gallery_loc.append((xmin, ymin, xmax, ymax))
                        crop_img = im0[ymin:ymax,
                                   xmin:xmax]  # HWC (602, 233, 3)  这个im0是读取的帧，获取该帧中框的位置 im0= <class 'numpy.ndarray'>

                        gallery_img.append(crop_img)
                        cls_l.append(cls)#
                if gallery_img:
                    for img in tqdm.tqdm(gallery_img):
                        g_feat= demo.run_on_image(img)
                        gallery_feats.append(g_feat)
                    gallery_feats = torch.cat(gallery_feats, dim=0)#图片中的人物的特征
                    gallery_feats = F.normalize(gallery_feats, p=2, dim=1)
                    print("The gallery feature is normalized")
                    distmat = 1 - torch.mm(query_feats, gallery_feats.t())
                    m= query_feats.shape[0]#m表示query的数量
                      # 这里distmat表示两张图像的距离，越小越接近  t() 方法会按照指定的维度进行转置。
                    #distmat = distmat.numpy()
                    # distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    #               torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                    distmat = distmat.detach().cpu().numpy()
                    #distmat = distmat.sum(axis=0) / len(query_feats)  # 平均一下query中同一行人的多个结果
                    index = distmat.argmin(axis=1)
                    for i in range(m):
                        if(distmat[i][index[i]]<0.1):#0.1是阈值，小于0.1才有可能是一个人
                            mask[gallery_loc[index[i]][1]-5:gallery_loc[index[i]][3]+5,
                            gallery_loc[index[i]][0]-5: gallery_loc[index[i]][2]+5] = 255#制作mask
                            masked = (im0 * (1 - mask[..., np.newaxis] / 255.0)).astype(np.uint8)
                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls_l[index[i]])]} '
                                plot_one_box(gallery_loc[index[i]], im0, label=label, color=colors[int(cls)],#cls是类别名称
                                              line_thickness=1)
                                line = (gallery_loc[index[i]][0], gallery_loc[index[i]][1],gallery_loc[index[i]][2], gallery_loc[index[i]][3])
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
if __name__ == '__main__':
    fast_detect("example/input-image/" + 'street',"./runs/detect2")



