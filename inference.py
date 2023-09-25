import time, cv2, torch, threading, argparse, os, re, subprocess
import torch.backends.cudnn as cudnn
import numpy as np 
import pandas as pd

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords 
from utils.plots import Annotator, colors
from utils.torch_utils import select_device 
from utils.accessory_lib import system_info


def get_thermal_zone_paths():
    thermal_path = '/sys/devices/virtual/thermal/'

    return [os.path.join(thermal_path, m.group(0)) for m in [re.search('thermal_zone[0-9]', d)
                                                             for d in os.listdir(thermal_path)] if m]


def read_sys_value(pth):
    return subprocess.check_output(['cat', pth]).decode('utf-8').rstrip('\n')


def get_thermal_zone_names(zone_paths):
    return [read_sys_value(os.path.join(p, 'type')) for p in zone_paths]


def get_thermal_zone_temps(zone_paths):
    return [int(read_sys_value(os.path.join(p, 'temp'))) for p in zone_paths]


class LoggingFile:
    def __init__(self, logging_header=pd.DataFrame(), file_name='logging_data'):
        self.start_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.logging_file_name = file_name + '_' + self.start_time
        self.logging_file_path = './logging_data/' + self.logging_file_name + '.csv'
        logging_header.to_csv(self.logging_file_path, mode='a', header=True)

    def start_logging(self, period=0.1): 
        logging_data = pd.DataFrame({'1': time.strftime('%Y/%m/%d', time.localtime(time.time())),
                                     '2': time.strftime('%H:%M:%S', time.localtime(time.time())),
                                     '3': round(elapsed_time, 2), '4': ref_frame, '5': round(fps, 2),
                                     '6': jetson_temp.read_temps()[0], '7': jetson_temp.read_temps()[1]}, index=[0])

        logging_data.to_csv(self.logging_file_path, mode='a', header=False) 
        logging_thread = threading.Timer(period, self.start_logging, (period, )) 
        logging_thread.daemon = True 
        logging_thread.start() 


class JetsonTemperature:
    def __init__(self):
        self.zone_paths_modify = []
        self.zone_paths = get_thermal_zone_paths()
        self.zone_paths_modify.append(self.zone_paths[1])
        self.zone_paths_modify.append(self.zone_paths[4])
        self.zone_names = get_thermal_zone_names(self.zone_paths_modify)
        self.zone_temps = np.zeros(shape=len(self.zone_names), dtype=np.float64)

    def get_temps(self):
        zone_temps_raw = get_thermal_zone_temps(self.zone_paths_modify)
        for i, temp_raw in enumerate(zone_temps_raw):
            self.zone_temps[i] = np.divide(temp_raw, 1000.0)

    def read_temps(self):
        return self.zone_temps

    def read_zone_names(self):
        return self.zone_names


jetson_temp = JetsonTemperature()


def read_jetson_temp():
    while True:
        jetson_temp.get_temps()
        time.sleep(2)


jetson_temp_task = threading.Thread(target=read_jetson_temp)
jetson_temp_task.daemon = True
jetson_temp_task.start()

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
    global elapsed_time, fps, ref_frame, prev_time, start_time, zone_temps

    system_info()
    source = "./car.jpg"
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
    font = cv2.FONT_HERSHEY_COMPLEX
    print(f'[3/3] Video Resource Loaded {time.time()-prev_time:.2f}sec')
    start_time = time.time()

    # logging file threading start
    logging_header_names = ['absolute_data', 'absolute_time', 'time(sec)', 'frame', 'throughput(fps)']

    for zone_name in jetson_temp.read_zone_names():
        logging_header_names.append(zone_name)

    logging_header = pd.DataFrame(columns=logging_header_names)
    logging_task = LoggingFile(logging_header, file_name='logging_data')
    logging_task.start_logging(period=0.1)

    normalize_tensor = torch.tensor(255.0).to(device)
    cv2.namedWindow(winname='video', flags=cv2.WINDOW_NORMAL)

    while True:
        img0 = cv2.imread(source)
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
        prev_time = time.time()

        cv2.putText(img0, f'Elapsed Time(sec): {elapsed_time: .2f}', (5, 20), font, 0.5, [0, 0, 255], 1)
        cv2.putText(img0, f'Process Speed(FPS): {fps: .2f}', (5, 40), font, 0.5, [0, 0, 255], 1)
        cv2.putText(img0, f'Frame: {ref_frame}', (5, 60), font, 0.5, [0, 0, 255], 1)

        # Stream results
        cv2.imshow("video", img0)
        elapsed_time = time.time()-start_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print('Video Play Done!')


if __name__ == "__main__":
    main()
