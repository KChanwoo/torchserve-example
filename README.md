# Torchserve Test

## Preference
```
yolo export model=yolov8n.pt imgsz=640  # create torchscript
mv yolov8n.torchscript yolov8n.torchscript.pt  # torchserve can recognize when file extension is 'pt'
```

## Run
```
torch-model-archiver --model-name yolov8n \
--version 0.1 \
--serialized-file yolov8n.torchscript.pt \
--handler handler.py \
--extra-files ./index_to_name.json,./handler.py \
--export-path model-store -f

torchserve --start \
--ncs \
--model-store model-store \
--models yolov8n.mar
```

## Docker
```
docker run --rm -it \
-p 8080:8080 -p 8081:8081 \
-v ./model-store:/home/model-server/model-store pytorch/torchserve:0.1-cpu \
torchserve --start --model-store model-store --models yolov8n.mar
```

## References
- [basic](https://ichi.pro/ko/torchserve-eseo-model-baepo-mich-sayongja-jijeong-haendeulleo-saengseong-277270280257169)  
- [Custom Handler - jarutis](https://gist.github.com/jarutis/f57a3db7b4c37b59163a2ff5d8c8d54e)  
- [Example yolov5 - IvanGarcia7](https://github.com/IvanGarcia7/TORCHSERVER)  
- [Example yolov8 export](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python)