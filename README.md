# ProPedestrian

Privacy protection based on specific pedestrians

This project can identify specific pedestrians in images or videos, remove them from the images, and restore them. Thus achieving anonymity of specific pedestrians

<video src="example/demo-3.mp4"></video>



## Download pre-trained model

1. yolov7:
   - Download pre-trained model：https://github.com/WongKinYiu/yolov7
   - Place it in the yolov7/weights folder
2. lama
   - Download pre-trained model: https://github.com/advimman/lama
   - Place it in the lama/big-lama folder
3. HiNet
   - Download pre-trained model: https://github.com/TomTomTommi/HiNet
   - Place it in the HiNet/model folder
4. Fast-reid
   - Download pre-trained model: https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md
   - Place it in the yolov7/fast_reid_master/model folder

## test

### Profile Path

In the files yolo_fastreid_lama.py、 anony_recory.py and yolov7/yolo_fastreid.py, the.../ replace with one's own actual work path

### Encryption process

1. Put the pedestrian images to be protected into /yolov7/fast-reid_master/datasets/query
2. You can place the image in example/input image and modify the corresponding path in yolo_fastreid_lama.py. Then run yolo_fastreid_lama.py

```python
python yolo_fastreid_lama.py
```


https://github.com/user-attachments/assets/e3bb611e-1224-40e8-b1a9-92bb2499e976


### Decryption process

```python
python anony_recory.py
```



