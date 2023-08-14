import glob
import argparse

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


import generate_xml

from ultralytics import YOLO
import cv2
import os
import datetime

class AutoAnnotate():
 
    def __init__(self, model_path: str,images_path: str, xml_path: str = None, detection_threshold: float = 0.5) -> None:
        self.model = load_detection_model(model_path)
        self.images_path = images_path
        self.images = glob.glob(self.images_path+'/*')
        self.xml_path = xml_path if xml_path else self.images_path
        self.detection_threshold = detection_threshold
        self.xml_generator = generate_xml.GenerateXml(self.xml_path)

    #通过摄像头,按空格键截图,并打标签
    def generate_annotations_with_video(self) -> None:
        """Iterates over all images and generates the annotations for each one."""
        capture = cv2.VideoCapture(0)
        timeStr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        path = "image_"+timeStr
        folder = os.path.exists(path)
        if not folder:  
            os.makedirs(path)   
        self.xml_generator = generate_xml.GenerateXml(path)
        i=0
        while True:
            _, frame = capture.read()
            original_frame = frame.copy()
            im_height, im_width, _ = frame.shape
            results = self.model(frame)
            frame = results[0].plot()
            cv2.imshow('frame', frame)
            key = cv2.waitKey(10) 
            if key == ord(' '):   # 按空格键保存
                detections  = eval(results[0].tojson())
                i+=1
                file_name = timeStr+'_'+str(i)
                cv2.imwrite(path+'/'+file_name+'.jpg',original_frame)
                class_names, bounding_boxes = self._filter_detections_by_threshold(
                    detections,
                    im_height,
                    im_width
                )
                self.xml_generator.generate_xml_annotation(
                    file_name,
                    bounding_boxes,
                    im_width,
                    im_height,
                    class_names
                )
                print("\033[32m图片已保存,打标成功!\033[0m")

            if key==27:    # Esc key to stop
                break

        capture.release()
        cv2.destroyAllWindows()

 

    #通过本地图片资源,打标签
    def generate_annotations(self) -> None:
        """Iterates over all images and generates the annotations for each one."""

        print(f'Found {len(self.images)} images to annotate.')

        for image in tqdm(self.images, colour='green'):
            try:
                img = np.array(ImageOps.exif_transpose(Image.open(image)))
                im_height, im_width, _ = img.shape

                detections = self._get_model_detections(img)
                class_names, bounding_boxes = self._filter_detections_by_threshold(
                    detections,
                    im_height,
                    im_width
                )
                file_name = image.split('\\')[-1]
                self.xml_generator.generate_xml_annotation(
                    file_name,
                    bounding_boxes,
                    im_width,
                    im_height,
                    class_names
                )
            except Exception as error:
                print(error)

    def _get_model_detections(self, image) -> dict:
        detections = self.model(image)
        result  = eval(detections[0].tojson())
        return result

    def _filter_detections_by_threshold(self, detections: list, heigth: int, width: int) -> tuple:
        class_names = []
        bounding_boxes = []
 
        for item in detections:
            if item['confidence'] > self.detection_threshold:
                class_names.append(item['name'])
                box = item['box']
                output_boxes = {'xmin': box['x1'], 'xmax': box['x2'],'ymin': box['y1'], 'ymax': box['y2']}
                bounding_boxes.append(output_boxes)

        return class_names, bounding_boxes

    def _get_box_coordinates(self, boxes: list, heigth: int, width: int, index: int) -> tuple:
        xmin, xmax, ymin, ymax = boxes[index][1], boxes[index][3], boxes[index][0], boxes[index][2]
       
        return (xmin, xmax, ymin, ymax)


def load_detection_model(model_path: str):
    """Loads an object detection model from path

    Args: model_path (str): Path to saved model directory.

    Returns:
        A tf.saved_model.load object.
    """
    try:
        print('Loading model into memory...')
        return YOLO(model_path)
    except Exception as error:
        print(f'Error loading model: {error}')
        raise error


 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto annotation arguments.')
    parser.add_argument('--label_map_path', type=str, help='The path of the label_map.pbtxt file containing the classes.')
    parser.add_argument('--model_path', type=str, help='The path of the saved model folder.')
    parser.add_argument('--imgs_path', type=str, help='The path of the images that will be annotated.')
    parser.add_argument('--xml_path', type=str, help='The path where the xml files will be saved.', default=None)
    parser.add_argument('--threshold', type=float, help='The path where the xml files will be saved.', default=0.5)

    args = parser.parse_args()

    AutoAnnotate(args.model_path,
                 args.label_map_path,
                 args.imgs_path,
                 args.xml_path,
                 args.threshold).generate_annotations()
