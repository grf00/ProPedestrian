from pathlib import Path
import sys
sys.path.append( "reid" )
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

from reid.data.transforms import build_transforms
from reid.data import make_data_loader
from reid.modeling import build_model
from reid.config import cfg as reidCfg

from models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import torch

def detect(save_img=False):
    # gc.collect()
    # torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7/weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default="yolov7/inference/myself-detect/yolo-reid", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt',action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', default='0',nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='yolov7/runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default='true',action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--mask_hui', default='.../lama/test-image/', help='don`t trace model')
    opt = parser.parse_args()
    #print(opt)
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


    # ---------- 行人重识别模型初始化 --------------------------
    #query_loader, num_query = make_data_loader(reidCfg)  # 验证集预处理
    train_loader, query_loader, num_query, num_classes = make_data_loader(reidCfg)
    reidModel = build_model(reidCfg, num_classes=1501)  # 模型初始化
    reidModel.load_param(reidCfg.TEST.WEIGHT)  # 加载权重
    reidModel.to(device).eval()  # 模型测试

    query_feats = []  # 测试特征
    query_pids = []  # 测试ID

    for i, batch in enumerate(query_loader):
        with torch.no_grad():
            img, pid, camid = batch  # 返回图片，ID，相机ID
            img = img.to(device)  # 将图片放入gpu
            feat = reidModel(img)  #
            query_feats.append(feat)  # 获得特征值列表
            query_pids.extend(np.asarray(pid))  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

    query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
    print("The query feature is normalized")
    query_feats = F.normalize(query_feats, dim=1, p=2)  # 计算出查询图片的特征向量


    # --------------- 行人检测模型初始化 -------------------

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

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
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
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
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

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

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    w = xmax - xmin
                    h = ymax - ymin
                    if w * h > 200:
                        gallery_loc.append((xmin, ymin, xmax, ymax))
                        crop_img = im0[ymin:ymax,
                                   xmin:xmax]  # HWC (602, 233, 3)  这个im0是读取的帧，获取该帧中框的位置 im0= <class 'numpy.ndarray'>

                        crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                        crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                        gallery_img.append((crop_img))
                        cls_l.append(cls)
                if gallery_img:
                    gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
                    gallery_img = gallery_img.to(device)
                    gallery_feats = reidModel(gallery_img)  # torch.Size([7, 2048])
                    print("The gallery feature is normalized")
                    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量
                    m, n = query_feats.shape[0], gallery_feats.shape[0]
                    distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                                  torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()

                    distmat.addmm_(query_feats, gallery_feats.t(), beta=1, alpha=-2 )
                    distmat = distmat.detach().cpu().numpy()
                    distmat = distmat.sum(axis=0) / len(query_feats)  # 平均一下query中同一行人的多个结果
                    index = distmat.argmin()
                    #if distmat[index] <= 1.8:#dist_thres:# print('距离：%s' % distmat[index])
                    mask[gallery_loc[index][1]:gallery_loc[index][3], gallery_loc[index][0] : gallery_loc[index][2] ] = 255
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls_l[index])]} '
                        plot_one_box(gallery_loc[index], im0,label=label, color=colors[int(cls)], line_thickness=1)
                            #cv2.imshow(str(p), im0)
                            #cv2.waitKey(0)


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

                    cv2.imwrite(opt.mask_hui+p.name[0:-4]+"_mask.png",mask)
                    # cv2.imshow(str(p), im0)
                    # cv2.waitKey(1)

                    print(f" The image with the result is saved in: {save_path}")
                    print(f" The mask with the result is saved in: {opt.mask_hui}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    # gc.collect()
    # torch.cuda.empty_cache()


if __name__ == '__main__':
    detect()
    #check_requirements(exclude=('pycocotools', 'thop'))


