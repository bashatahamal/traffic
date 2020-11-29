import cv2
import torch
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import matplotlib.path as MPath
import random
#sys.path.insert(1, '/home/mhbrt/Desktop/Wind/Project/Traffic/yolov5/')
sys.path.insert(1, './yolov5/')
from utils.utils import *
from utils.datasets import *
from utils import google_utils
from flask import Flask, render_template, Response, jsonify, after_this_request, make_response, request
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
# from camera_breakdown import VideoCamera

app = Flask(__name__)

data = {}
data[0] = 0  # pedestrian
data[1] = 0  # motobike
data[2] = 0  # car
data[3] = 0  # bus
data[4] = 0  # truck
data[5] = 0  # invalid direction
data[6] = 0  # invalid turn
stop_is_pressed = False
url = ''
line_coordinat = []
polygon = []
invalid_move = []
type_process = []  #counting vehicle, invalid direction, invalid turn

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def index():
    return render_template('index.html')

def get_frame():
    image = cv2.imread('./static/images/last_im0.jpg')
    # cv2.imshow('e', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes()

def gen():
    frame = get_frame()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/last_im0')
def last_im0():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

import time
def get_size_and_frame(source):
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    cap=cv2.VideoCapture(0 if source == '0' else source)
    for i in range(10):
        ret, frame=cap.read()
        # cv2.imshow('s', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if i == 5:
            cv2.imwrite('./static/images/frame.jpg', frame)
        h, w, _ = frame.shape
    cap.release()
    time.sleep(2)
    return h, w
    # if webcam:
    #     # dataset = LoadStreams(source)
    #     cap=cv2.VideoCapture(0 if source == '0' else source)
    #     for i in range(10):
    #         ret, frame=cap.read()
    #         # cv2.imshow('s', frame)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    #         if i == 5:
    #             cv2.imwrite('./static/images/frame.jpg', frame)
    #         h, w, _ = frame.shape
    #     # cap.release()
    #     return h, w
    # else:
    #     dataset = LoadImages(source)
    #     k = 0
    #     for path, img, im0s, vid_cap in dataset:
    #         k+=1
    #         h, w, _ = im0s.shape
    #     # if k == 10:
    #     #     # cv2.imwrite('./static/images/frame.jpg', im0s)
    #     #     cv2.imshow('s', np.float32(im0s))
    #     #     cv2.waitKey(0)
    #     #     cv2.destroyAllWindows()
    #     #     h, w, _ = im0s.shape
    #     #     break
    
    #     return h, w

class IPoint: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

# Given three colinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False

def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
        # Clockwise orientation 
        return 1
    elif (val < 0): 
        # Counterclockwise orientation 
        return 2
    else:
        # Colinear orientation 
        return 0

# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
    # Special Cases 
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
    # If none of the cases 
    return False

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0   
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
    return img

def most_frequent(List): 
    return max(set(List), key = List.count) 

def detect(weights='',
           source='inferences/images',
           output='inferences/output', 
           img_size=640, 
           conf_thres=0.4,
           iou_thres=0.5, 
           device='', 
           view_img=False,
           save_img=False,
           save_txt=False,
           classes=None,
           agnostic_nms=True,
           augment=True,
           update=True,
           fps_count=1,
           line_coordinat = [],  # [[(x1,y1),(x2,y2)], ...]
           polygon=[],  # [[[(x1,y1),(x2,y2),(x3,y3),...], [0(u)/1(r)/2(d)/3(l)]], ...]
           invalid_move = []  # [[0,1], ...]
           ):
    global data
    global stop_is_pressed
    global type_process
    
    out, source, weights, view_img, save_txt, imgsz = \
        output, source, weights, view_img, save_txt, img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file('./deep_sort/configs/deep_sort.yaml')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = torch_utils.select_device(device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)[
        'model'].float().eval()  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        # view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
        # dataset = LoadWebcam(source, img_size=imgsz)
    else:
        save_img = True
        # view_img = True
        dataset = LoadImages(source, img_size=imgsz)
        # dataset = LoadStreams(source, img_size=imgsz)
    # save_img = True
    # view_img = True

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)]
                for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    # print(model)
    k = 0
    limit = 10
    id_limit = 50
    output_all_frames = {}
    counting_id = []
    invalid_direction_id = []
    invalid_turn_id = []
    for path, img, im0s, vid_cap in dataset:
        # print(stop_is_pressed)
        while(stop_is_pressed):
        #     cv2.imwrite('./static/images/last_im0.jpg', im0)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # break
        # else:
        k += 1
        if k == fps_count:
            k = 0
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # print(model(img, augment=augment)[0])
            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=augment)[0]
            # print('pred b4 nms', pred)
            # Apply NMS
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            t2 = torch_utils.time_synchronized()
            # print('pred', pred)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' %
                                                            dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    
                    bbox_xywh = []
                    confs = []
                    save_from_det = {}
                    for x in range(0, 5):
                        save_from_det[x] = []
                    # Write results
                    for *xyxy, conf, cls in det:
                        for x in range(0, 5):
                            if int(cls) == x:
                                # data[x] += 1
                                save_from_det[x].append(
                                    [int(xyxy[0].item()), int(xyxy[1].item()),
                                     int(xyxy[2].item()), int(xyxy[3].item())])
                                break

                        img_h, img_w, _ = im0.shape
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                        #         1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * 5 + '\n') %
                        #                 (cls, *xywh))  # label format

                        # if save_img or view_img:  # Add bbox to image
                        #     label = '%s %.2f' % (names[int(cls)], conf)
                        #     plot_one_box(xyxy, im0, label=label,
                        #                     color=colors[int(cls)], line_thickness=2)
                    # print('Save from det : ', save_from_det)
                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    for p, q in line_coordinat:
                        cv2.line(im0, p, q, (100, 255, 100), 2)
                    for x in range(len(polygon)):
                        poly = polygon[x][0]
                        pts = np.array(poly, np.int32)
                        pts = pts.reshape((-1, 1, 2)) 
                        cv2.polylines(im0, [pts], True, (255, 0, 0), 1) 
  
                    
                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)
                    print('Output Deep Sort: ', outputs)
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, identities)
                        # Save all results to dictionary
                        for i, box in enumerate(bbox_xyxy):
                            x1, y1, x2, y2 = [int(i) for i in box]
                            print('x1 y1 x2 y2 : ', x1,y1, x2, y2)
                            print('i : ', i)
                            print('int(identities[i]) : ', int(identities[i]))
                            ds_class = float('inf')
                            smallest = float('inf')
                            for x in save_from_det:
                                for sx1, sy1, sx2, sy2 in save_from_det[x]:
                                    diff = sum(abs(np.array([x1, y1, x2, y2])
                                            -np.array([sx1, sy1, sx2, sy2])))
                                    if diff < smallest:
                                        smallest = diff
                                        ds_class = x
                            if int(identities[i]) in output_all_frames.keys():
                                # check crossed line
                                if type_process[0]:
                                    (w1, h1) = (x2-x1, y2-y1)
                                    prev_xyxy = output_all_frames[int(identities[i])][0][-1]
                                    print('prev xyxy', prev_xyxy)
                                    (xp, yp) = (int(prev_xyxy[0]), int(prev_xyxy[1]))
                                    (wp, hp) = (int(prev_xyxy[2]-xp), int(prev_xyxy[3]-yp))
                                    # p1 = (int(x1 + (w1-x1)/2), int(y1 + (h1-y1)/2))
                                    # q1 = (int(xp + (wp-xp)/2), int(yp + (hp-yp)/2))
                                    p1 = (int(x1 + (w1)/2), int(y1 + (h1)/2))
                                    q1 = (int(xp + (wp)/2), int(yp + (hp)/2))
                                    print('p1 q1 : ', p1, q1)
                                    cv2.line(im0, p1, q1, (10, 255, 10), 3)
                                    p1 = IPoint(p1[0], p1[1])
                                    q1 = IPoint(q1[0], q1[1])
                                    for p2, q2 in line_coordinat:
                                        p2 = IPoint(p2[0], p2[1])
                                        q2 = IPoint(q2[0], q2[1])
                                        if doIntersect(p1, q1, p2, q2):
                                            if int(identities[i]) not in counting_id:
                                                counting_id.append(int(identities[i]))
                                                data[most_frequent(output_all_frames[int(identities[i])][1])] += 1
                                # check direction
                                if type_process[1]:
                                    minus_y1 = prev_xyxy[1] - y1 
                                    minus_y2 = prev_xyxy[3] - y2
                                    minus_x1 = prev_xyxy[0] - x1
                                    minus_x2 = prev_xyxy[2] - x2
                                    # 0=up, 1=right, 2=down, 3=left
                                    if minus_y1 > 0 and minus_y2 > 0:
                                        output_all_frames[int(identities[i])][4].append(0)
                                        label = '^'
                                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)[0]
                                        cv2.putText(im0, label, (x1, y1 - int(t_size[1]/2)), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
                                    if minus_y1 < 0 and minus_y2 < 0:
                                        output_all_frames[int(identities[i])][4].append(2)
                                        label = 'v'
                                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)[0]
                                        cv2.putText(im0, label, (x1, y1 - int(t_size[1]/2)), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
                                    if minus_x1 > 0 and minus_x2 > 0:
                                        output_all_frames[int(identities[i])][4].append(3)
                                        label = '<'
                                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)[0]
                                        cv2.putText(im0, label, (x1, y1 - int(t_size[1]/2)), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
                                    if minus_x1 < 0 and minus_x2 < 0:
                                        output_all_frames[int(identities[i])][4].append(1)
                                        label = '>'
                                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)[0]
                                        cv2.putText(im0, label, (x1, y1 - int(t_size[1]/2)), cv2.FONT_HERSHEY_PLAIN, 1.2, [255, 255, 255], 1)
                                # check region
                                if type_process[2]:
                                    for x in range(len(polygon)):
                                        path = MPath.Path(polygon[x][0])
                                        # inside2 = path.contains_points([[i[0], i[1]]])
                                        if path.contains_point((x1+int(w1/2), y1+int(h1/2))):
                                            output_all_frames[int(identities[i])][2].append(x)
                                            output_all_frames[int(identities[i])][3].append(polygon[x][1])
                                # check for invalid direction
                                if len(output_all_frames[int(identities[i])][3]) > int(3/4*limit)\
                                        and len(output_all_frames[int(identities[i])][4]) > int(3/4*limit)\
                                             and type_process[1]:
                                    # if most_frequent(output_all_frames[int(identities[i])][3]) \
                                    #         != most_frequent(output_all_frames[int(identities[i])][4]):
                                    unique, frequency = np.unique(output_all_frames[int(identities[i])][4],
                                                                  return_counts=True)
                                    true_direction = most_frequent(output_all_frames[int(identities[i])][3])
                                    if true_direction not in unique:
                                        if int(identities[i]) not in invalid_direction_id:
                                            invalid_direction_id.append(int(identities[i]))
                                            data[5] += 1
                                        label = '!'
                                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                                        cv2.putText(im0, label, (x1 + int(t_size[1]/2), y1), cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 255], 2)
                                    else:
                                        for x in range(len(unique)):
                                            if true_direction == unique[x]:
                                                id_true_in_unique = x
                                                break
                                        if frequency[id_true_in_unique] < int(1/3*limit):
                                            if int(identities[i]) not in invalid_direction_id:
                                                invalid_direction_id.append(int(identities[i]))
                                                data[5] += 1
                                            label = '!'
                                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                                            cv2.putText(im0, label, (x1 + int(t_size[1]/2), y1), cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 255], 2)
                                # check for invalid turn
                                if len(output_all_frames[int(identities[i])][2]) > int(3/4*limit) and type_process[2]:
                                    # unique, frequency = np.unique(output_all_frames[int(identities[i])][2],
                                    #                                 return_counts=True)
                                    first = True
                                    region_trace = []
                                    for r in output_all_frames[int(identities[i])][2]:
                                        if first:
                                            reg = r
                                            region_trace.append(r)
                                            first = False
                                        if reg != r :
                                            region_trace.append(r)
                                            reg = r
                                    if len(region_trace) > 1:
                                        for reg1, reg2 in invalid_move:
                                            for k in range(len(region_trace)):
                                                if k+1 > len(region_trace)-1:
                                                    break
                                                if (region_trace[k], region_trace[k+1]) == (reg1, reg2):
                                                    if int(identities[i]) not in invalid_turn_id:
                                                        invalid_turn_id.append(int(identities[i]))
                                                        data[6] += 1
                                                    label = 'X'
                                                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                                                    cv2.putText(im0, label, (x1 + int(t_size[1]/2), y1), cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 255], 2)
                            else:
                                # oaf[ID] = [[in frame coordinat], [class_type], [region]
                                #            [true_direction], [pred_direction]]
                                output_all_frames[int(identities[i])] = [[], [], [], [], []]

                            output_all_frames[int(identities[i])][0].append((x1, y1, x2, y2))
                            if len(output_all_frames[int(identities[i])][0]) > limit:
                                output_all_frames[int(identities[i])][0] = output_all_frames[int(identities[i])][0][-limit:]
                            output_all_frames[int(identities[i])][1].append(ds_class)
                            if len(output_all_frames[int(identities[i])][1]) > limit:
                                output_all_frames[int(identities[i])][1] = output_all_frames[int(identities[i])][1][-limit:]
                            if len(output_all_frames[int(identities[i])][2]) > limit:
                                output_all_frames[int(identities[i])][2] = output_all_frames[int(identities[i])][2][-limit:]
                            if len(output_all_frames[int(identities[i])][3]) > limit:
                                output_all_frames[int(identities[i])][3] = output_all_frames[int(identities[i])][3][-limit:]
                            if len(output_all_frames[int(identities[i])][4]) > limit:
                                output_all_frames[int(identities[i])][4] = output_all_frames[int(identities[i])][4][-limit:]

                        # delete output_all_frames oldest if more than n number of id
                        if len(output_all_frames) > id_limit:
                            unused = list(set(output_all_frames.keys())
                                     -set(sorted(output_all_frames.keys())[-id_limit:]))
                            for x in unused:
                                del output_all_frames[x]
                        if len(counting_id) > id_limit:
                            counting_id = counting_id[-id_limit:]
                        if len(invalid_direction_id) > id_limit:
                            invalid_direction_id = invalid_direction_id[-id_limit:]
                        if len(invalid_turn_id) > id_limit:
                            invalid_turn_id = invalid_turn_id[-id_limit:]
                        print('All Frame : ', output_all_frames)

                    # Write MOT compliant results to file
                    if save_txt and len(outputs) != 0:  
                        for j, output in enumerate(outputs):
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2]
                            bbox_h = output[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        print('saving img!')
                        cv2.imwrite(save_path, im0)
                    else:
                        print('saving video!')
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)
            # cv2.imshow(p, im0)
            # if cv2.waitKey(1) == ord('q'):  # q to quit
            #     raise StopIteration
            ret, jpeg = cv2.imencode('.jpg', im0)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
            

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    global stop_is_pressed
    global url
    global line_coordinat
    global polygon
    global invalid_move

    # print(stop_is_pressed)
    if stop_is_pressed:
        print('FROM IMAGE')
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        print('FROM VIDEO FEED')
        return Response(detect(
            weights='../best_Road.pt',
            # source='rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov',
            # source='/home/mhbrt/Desktop/EVA/eva/media/condong catur.mp4'
            source=url,
            # source='0',
            img_size=256,
            # augment=True,
            agnostic_nms=True,
            fps_count= 10,
            classes=None,       # Filter by class
            conf_thres=0.1,
            iou_thres=0.5,
            line_coordinat=line_coordinat,
            polygon=polygon,
            invalid_move=invalid_move
        ),mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/data', methods=['GET'])
def hello():
    global data

    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    print(jsonify(data))
    return jsonify(data)

@app.route('/bridge',  methods=['POST'])
def bridge():
    global stop_is_pressed
    global url
    global line_coordinat
    global polygon
    global invalid_move
    global type_process

    if request.method == "POST":
        if request.is_json:
            req = {}
            res = request.get_json()
            print(res)
            if len(res) > 1 and type(res) == type([]):
                print('PLAY')
                stop_is_pressed = False

                print('Configuration')
                for x1, y1, x2, y2 in res[0]:
                    line_coordinat.append([(x1, y1), (x2, y2)])
                for x in range(len(res[1])):
                    list_poly = []
                    for xy in res[1][x]:
                        list_poly.append((xy['x'], xy['y']))
                    if res[2][x] == 'UP':
                        dir_n = 0 
                    if res[2][x] == 'LEFT':
                        dir_n = 1
                    if res[2][x] == 'DOWN':
                        dir_n = 2
                    if res[2][x] == 'RIGHT':
                        dir_n = 3
                    poly_and_dir = [list_poly, dir_n]
                    polygon.append(poly_and_dir)
                for regA, regB in res[3]:
                    invalid_move.append([int(regA), int(regB)])
                type_process = res[4]
                print('1 : ', line_coordinat)
                print('2 : ', polygon)
                print('3 : ', invalid_move)
                print('4 : ', type_process)
                response = make_response(jsonify(res), 200)
                return response
            elif len(res) == 1 and type(res) == type([]):
                url = res[0]
                h, w = get_size_and_frame(url)
                shape = [h, w]
                response = make_response(jsonify(shape), 200)
                return response
            elif res == 'STOP':
                stop_is_pressed = True
                # sys.modules[__name__].__dict__.clear()
                response = make_response(jsonify('stop is accepted'), 200)
                return response
            else:
                response = make_response(jsonify(res), 200)
                return response
    return ('bridge')


if __name__ == '__main__':
    app.run(host = '127.0.0.3', debug = True)
