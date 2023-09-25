import time, cv2, torch, threading, argparse
import torch.backends.cudnn as cudnn
import numpy as np 
import pandas as pd

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords 
from utils.plots import Annotator, colors
from utils.torch_utils import select_device 
from utils.accessory_lib import system_info


class LoggingFile:
    def __init__(self, logging_header=pd.DataFrame(), file_name='logging_data'):
        self.start_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.logging_file_name = file_name + '_' + self.start_time
        self.logging_file_path = './logging_data/' + self.logging_file_name + '.csv'
        logging_header.to_csv(self.logging_file_path, mode='a', header=True)

    def start_logging(self, period=0.1): 
        logging_data = pd.DataFrame({'1': time.strftime('%Y/%m/%d', time.localtime(time.time())),
                                     '2': time.strftime('%H:%M:%S', time.localtime(time.time())),
                                     '3': round(elapsed_time, 2), '4': ref_frame, '5': round(fps, 2)}, index=[0])
        logging_data.to_csv(self.logging_file_path, mode='a', header=False) 
        logging_thread = threading.Timer(period, self.start_logging, (period, )) 
        logging_thread.daemon = True 
        logging_thread.start() 


parser = argparse.ArgumentParser()
parser.add_argument("--fp16", type=int, default=0)
args = parser.parse_args()

elapsed_time: float = 0.0
fps: float = 0.0
ref_frame: int = 0
prev_time = time.time()
start_time = time.time()


@torch.no_grad()
def main():
    global elapsed_time, fps, ref_frame, prev_time, start_time
    system_info()

    source = "D:\\video\\urban_street.mp4"
    weights = "./weights/yolov5s.pt"
    img_size = 640
    CONF_THRES = 0.4
    IOU_THRES = 0.45
    half = args.fp16
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False

    # Initialize
    device = select_device('')
    print(f'[1/3] Device Initialized {time.time()-prev_time:.2f}sec')
    prev_time = time.time()
    
    # Load model
    model = attempt_load(weights, device)  # load FP32 model
    model.eval()
    stride = int(model.stride.max())  # model stride
    img_size_chk = check_img_size(img_size, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    model(torch.zeros(1, 3, img_size_chk, img_size_chk).to(device).type_as(next(model.parameters())))  # run once
    print(f'[2/3] Yolov5 Detector Model Loaded {time.time()-prev_time:.2f}sec')
    prev_time = time.time()

    # Load image
    video = cv2.VideoCapture(source)
    font = cv2.FONT_HERSHEY_COMPLEX
    print(f'[3/3] Video Resource Loaded {time.time()-prev_time:.2f}sec')
    start_time = time.time()

    # logging file threading start
    logging_header = pd.DataFrame(columns=['absolute_data', 'absolute_time', 'time(sec)', 'frame', 'throughput(fps)'])
    logging_task = LoggingFile(logging_header, file_name='logging_data')
    logging_task.start_logging(period=0.1)

    normalize_tensor = torch.tensor(255.0).to(device)
    cv2.namedWindow('video', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)

    while video.isOpened():
        prev_time = time.time()
        ret, img0 = video.read()

        if ret:
            ref_frame = ref_frame + 1

            # Padded resize
            img = letterbox(img0, img_size_chk, stride=stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = torch.divide(img, normalize_tensor)

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False)

            # Process detections
            det = pred[0]
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(img0, line_width=1, example=str(names))

            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls)))

            fps = 1/(time.time() - prev_time)
            cv2.putText(img0, f'Elapsed Time(sec): {elapsed_time: .2f}', (5, 20), font, 0.5, [0, 0, 255], 1)
            cv2.putText(img0, f'Process Speed(FPS): {fps: .2f}', (5, 40), font, 0.5, [0, 0, 255], 1)
            cv2.putText(img0, f'Frame: {ref_frame}', (5, 60), font, 0.5, [0, 0, 255], 1)

            # Stream results
            cv2.imshow("video", img0)
            elapsed_time = time.time()-start_time

        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print('Video Play Done!')


if __name__ == "__main__":
    main()
