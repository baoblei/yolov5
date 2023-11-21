# !/usr/bin/python
# coding=utf-8
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
              
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
import socket
import threading
import struct
import select
import time


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

def calculate_checksum(data_):
    # 计算字节流的校验和
    return sum(data_) & 0xFF

def unpack(rev_data):
    # 解析数据包(文件头、类型、命令、url源类型、校验)i
    header, type_, cmdid, source_cls, check = struct.unpack('BBBBB', rev_data[:5])

    # 计算校验和
    calculated_checksum = calculate_checksum(rev_data[:-1])

    # 检查校验位是否准确，校验不正确返回-1，正确返回cmdid, source_cls
    if check == calculated_checksum:
        return cmdid, source_cls
    else:
        return -1, -1


class DetecResultQ:
    def __init__(self,results=[], m=0):
        self.header = 0xff 
        self.type = 0x000000aa    
        self.m = m  # 识别数量
        self.results = results# 识别结果，最多五个目标
        self.check = 0x00

    def sort_results_by_prob(self):
        if self.m > 5:
            # 按照 results[i].prob 从大到小排序
            sorted_results = sorted(self.results, key=lambda result: result.prob, reverse=True)
            # 保留前五个对象
            self.results = sorted_results[:5]  
            self.m = 5  
    
    def calculate_checksum(self, data):
        # 计算字节流的校验和
        return sum(data) & 0xFF
    
    def pack(self):
        # 使用struct.pack指定每个属性的大小和格式
        # '10s' 表示一个长度为10的字符串，'i' 表示一个整数
        packed_header = struct.pack('B', self.header)
        packed_type = struct.pack('i', self.type)
        packed_m = struct.pack('i', self.m)
        # 使用循环将每个类对象的属性打包
        packed_results = b""
        for result in self.results:
            packed_results += struct.pack('iiiiif', result.id.numpy(), result.x.numpy(), result.y.numpy(), result.w.numpy(), result.h.numpy(), result.prob)

        self.check = self.calculate_checksum(packed_header + packed_type + packed_m + packed_results)
        packed_check = struct.pack('B', self.check)
        return packed_header + packed_type + packed_m + packed_results + packed_check
    
    def unpack(self,data):
        # 使用struct.unpack解析二进制数据
        header, type, m = struct.unpack('Bii', data[:9])
        
        # 解析每个类对象的属性
        results = []
        offset = 9
        for _ in range(self.m):
            result_data = struct.unpack('iiiiif', data[offset:offset+24])
            results.append(result_data)
            offset += 24

        check = struct.unpack('B', data[-1:])
        return header, type, m, results, check

class DetectResult:   
    def __init__(self, id, x, y, w, h, prob):
        self.id = id.to('cpu').int()
        self.x = x.to('cpu').int()
        self.y = y.to('cpu').int()
        self.w = w.to('cpu').int()
        self.h = h.to('cpu').int()
        self.prob = prob

# 全局变量 cmdid =0: start detect =1: stop detect soruce_cls: 0,1,2,3
g_cmdid = 0
g_source_cls = 2
ip_port = ("192.168.8.200", 20005)
soc = socket.socket()
soc_recv=socket.socket()
lock = threading.Lock()
# def connect_to_server():
#     global soc_recv
#     while True:
#         try:
#             with lock:
#                 if soc_recv is None or soc_recv.fileno() == -1:
#                     soc_recv.connect(ip_port)
            
#             # 如果与服务器通信后退出循环
#             break
        
#         except Exception as e:
#             print(f"Connection failed: {e}")
            
#             # 等待一段时间后尝试重新连接
#             time.sleep(5)

#     # return soc

# 接收线程
def recv():
    global g_cmdid
    global g_source_cls
    global soc_recv
    while True:
        # 监听 g_soc 的可读事件
        # readable, _, _ = select.select([g_soc], [], [], 1)

        # if readable:
        #     # 如果 g_soc 可读，执行接收操作
        #     reply = g_soc.recv(5)
        #     print("Received data:", reply)
        # else:
        #     # 如果超时，执行相应的操作（例如停止监听）
        #     print("Timeout reached. Stopping listening.")
        #     # 此处可以添加停止监听的代码
        try:    
            reply = soc_recv.recv(5)  # 重写recv函数，不等待太久 or 单独开一个接收套接字
        except Exception as e:
            # 重新连接
            soc_recv = socket.socket()
            soc_recv.connect(ip_port)
            continue
        cmdid_, source_cls_ = unpack(reply)
        if cmdid_ != -1:   
            with lock:            
                g_cmdid, g_source_cls = cmdid_, source_cls_

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global g_cmdid
    global g_source_cls
    global soc

    save_img = True  # save inference images
    is_file = False
    is_url = True
    webcam = True
    screenshot = False

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    bs = 1  # batch_size
   
    # warmup model
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  

    while True:
        with lock:
            cmdid, source_cls = g_cmdid, g_source_cls

        if cmdid==1: continue
        else:
            # 获取rtsp地址
            if source_cls==0:  # 2688X1520P
                source = str('rtsp://admin:abcd1234@192.168.8.108:8555/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')
            elif source_cls==1: # 1920X1080P
                source = str('rtsp://admin:abcd1234@192.168.8.108:8555/cam/realmonitor?channel=1&subtype=2&unicast=true&proto=Onvif')
            elif source_cls==2: # 704X576P
                source = str('rtsp://admin:abcd1234@192.168.8.108:8555/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif')
            elif source_cls==3: # 红外
                source = str('rtsp://admin:abcd1234@192.168.8.63:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1')
            # dataloader()
            if webcam:
                # view_img = check_imshow(warn=True)
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                bs = len(dataset)
            vid_path, vid_writer = [None] * bs, [None] * bs

            # Run inference
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            if cmdid==0: # 开启识别
                for path, im, im0s, vid_cap, s in dataset:
                    with dt[0]:
                        im = torch.from_numpy(im).to(model.device)
                        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim

                    # Inference
                    with dt[1]:
                        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                        pred = model(im, augment=augment, visualize=visualize)

                    # NMS
                    with dt[2]:
                        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                    # Define the path for the CSV file
                    csv_path = save_dir / 'predictions.csv'

                    # Create or append to the CSV file
                    def write_to_csv(image_name, prediction, confidence):
                        data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
                        with open(csv_path, mode='a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=data.keys())
                            if not csv_path.is_file():
                                writer.writeheader()
                            writer.writerow(data)

                    # Process predictions
                    results=[]
                    for i, det in enumerate(pred):  # per image
                        seen += 1
                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        s += '%gx%g ' % im.shape[2:]  # print string
                        # normalization gain whwh
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
                        print('video source:%d' % (source_cls))
                        # if source_cls==0:
                        #     gn = torch.tensor([1520,2688,1520,2688])
                        # elif source_cls==1:
                        #     gn = torch.tensor([1080,1920,1080,1920])
                        imc = im0.copy() if save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, 5].unique():    # c is class_id
                                n = (det[:, 5] == c).sum()  # numbers per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):   # box confidence class
                                c = int(cls)  # integer class
                                label = names[c] if hide_conf else f'{names[c]}'
                                confidence = float(conf)
                                confidence_str = f'{confidence:.2f}'

                                if save_csv:
                                    write_to_csv(p.name, label, confidence_str)

                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or save_crop or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    result = DetectResult(cls, xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1], confidence)
                                    results.append(result)
                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        # Stream results
                        im0 = annotator.result()
                        resultQ = DetecResultQ(results=results,m=len(results))  
                        resultQ.sort_results_by_prob()
                        data_send = resultQ.pack()
                        with lock:
                            try:
                                soc.sendall(data_send)
                            except Exception as e:
                                print(f"Connection failed: {e}")
                                soc = socket.socket()
                                soc.connect(ip_port)
                                time.sleep(2)
                                continue

                        if view_img:
                            if platform.system() == 'Linux' and p not in windows:
                                windows.append(p)
                                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                            cv2.imshow(str(p), im0)
                            cv2.waitKey(1)  # 1 millisecond

                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                cv2.imwrite(save_path, im0)
                            else:  # 'video' or 'stream'
                                if vid_path[i] != save_path:  # new video
                                    vid_path[i] = save_path
                                    if isinstance(vid_writer[i], cv2.VideoWriter):
                                        vid_writer[i].release()  # release previous video writer
                                    if vid_cap:  # video, rtsp
                                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                vid_writer[i].write(im0)

                    # Print time (inference-only)
                    LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

                # Print results
                t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
                LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                if save_txt or save_img:
                    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
                if update:
                    strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp2/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')

    # parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default='rtsp://admin:abcd1234@192.168.8.108:8555/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--source', type=str, default='rtsp://192.168.8.88:554/av0_0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/myVOC.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=5, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', default=False, help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes',default=[0], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    # 启动一个线程来修改全局变量
    # thread1 = threading.Thread(target=recv)
    # thread1.start()
    # thread2 = threading.Thread(target=connect_to_server)
    # thread2.start()
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
