# ProPedestrian

Privacy protection based on specific pedestrians

This project can identify specific pedestrians in images or videos, remove them from the images, and restore them. Thus achieving anonymity of specific pedestrians

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

### Encryption process

You can place the image in example/input image and modify the corresponding path in yolo_fastreid_lama.py. Then run yolo_fastreid_lama.py

```python
python yolo_fastreid_lama.py
```

### Decryption process

```python
python anony_recory.py
```


