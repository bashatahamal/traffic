
import cv2
import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.insert(1, '/home/mhbrt/Desktop/Wind/Project/Traffic/yolov5/')
from utils.utils import *
from utils.datasets import *
from utils import google_utils
from flask import Flask, render_template, Response, jsonify, after_this_request, make_response, request
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
# from camera_breakdown import VideoCamera

app = Flask(__name__)

data = {}
data[0] = 0
data[1] = 0
data[2] = 0
data[3] = 0
data[4] = 0
data[5] = 0
stop_is_pressed = False
url = ''

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
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def detect(save_img=False, weights='', source='inferences/images',
           output='inferences/output', img_size=640, conf_thres=0.4,
           iou_thres=0.5, device='', view_img=False, save_txt=False,
           classes=None, agnostic_nms=True, augment=True, update=True, fps_count=1):
    global data
    global stop_is_pressed
    
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
    save_img = True
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
                    
                    # Write results
                    for *xyxy, conf, cls in det:
                        for x in range(0, 5):
                            if int(cls) == x:
                                data[x] += 1
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
                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)
                    
                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)
                    print('Output Deep Sort: ', outputs)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        draw_boxes(im0, bbox_xyxy, identities)

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
    # print(stop_is_pressed)
    if stop_is_pressed:
        print('FROM IMAGE')
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        print('FROM VIDEO FEED')
        return Response(detect(
            weights='/home/mhbrt/Desktop/Wind/Project/Traffic/best_Road.pt',
            # source='rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov',
            # source='/home/mhbrt/Desktop/EVA/eva/media/condong catur.mp4'
            source=url,
            # source='0',
            img_size=640,
            # augment=True,
            agnostic_nms=True,
            fps_count= 10,
            classes=None,       # Filter by class
            conf_thres=0.1,
            iou_thres=0.5
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

    if request.method == "POST":
        if request.is_json:
            req = {}
            res = request.get_json()
            print(res)
            if len(res) > 1 and type(res) == type([]):
                print('PLAY')
                stop_is_pressed = False
                response = make_response(jsonify('play'), 200)
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
