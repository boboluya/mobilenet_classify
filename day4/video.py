# 打开摄像头识别摄像头前的物体分类和画框
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="yolov8n-640", video_reference=0, on_prediction=render_boxes, max_fps=1
)
pipeline.start()
pipeline.join()
