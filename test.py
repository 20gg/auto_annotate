#-*-coding=utf-8-*-
from auto_annotate import AutoAnnotate

ann_tool = AutoAnnotate(
model_path = 'D:\\workspace_mediapipe\\auto-assign\\model\\direction_yolov8.pt',
images_path = 'D:\\workspace_mediapipe\\auto-assign\\images',
xml_path = 'D:\\workspace_mediapipe\\auto-assign\\images_annotate',
detection_threshold = 0.40)
 

#通过本地图片生成打标
#ann_tool.generate_annotations()

#通过摄像有截图并打标
ann_tool.generate_annotations_with_video()