![propic](https://github.com/user-attachments/assets/81bb2df3-86a9-4a61-82a2-e0fb3547c278)#ProPedestrian

Privacy protection based on specific pedestrians

This project can identify specific pedestrians in images or videos, remove them from the images, and restore them. Thus achieving anonymity of specific pedestrians

With the rapid development of machine learning and the expansion of personal data, various intelligent applications continue to emerge, providing significant value to individuals and society. However, sensitive personal information has raised increasingly serious privacy concerns. Ubiquitous surveillance systems capture a large volume of raw pedestrian images and videos. On one hand, this is useful for legitimate users in many scenarios, such as criminal investigations. On the other hand, these images and videos, stored locally or uploaded to cloud servers, are vulnerable to hackers, posing severe privacy risks to individuals and public safety. Raw images or videos contain sensitive information about pedestrians, such as the true identities of specific individuals or communities. Without careful protection, highly sensitive information could be leaked or misused by malicious parties.

This project focuses on a deep learning-based system for privacy protection of specific pedestrians in surveillance videos. The system can identify individuals in the video and detect those who require privacy protection. The specific individuals can either be extracted from the video or uploaded separately. Once identified, the system removes the individuals from the video and seamlessly fills in the missing areas. It also implements a restoration function that can recover the removed individuals to their original positions in the video. The project enables both pedestrian anonymization and the recovery of anonymized individuals.

![图片3](https://github.com/user-attachments/assets/5915495d-1ffa-4e29-b152-df02da4ff383)

## Download pre-trained model

1. yolov7:
   - The purpose of using YOLOv7 is to detect pedestrians in complex scenes within images or videos
   - Implementation of paper：[[2207.02696\] YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
   - Download pre-trained model：https://github.com/WongKinYiu/yolov7
   - Place it in the yolov7/weights folder
2. lama
   - The purpose of using lama is to erase pedestrians from images or videos and then restore the images or videos
   - Implementation of paper：[[2109.07161\] Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
   - Download pre-trained model: https://github.com/advimman/lama
   - Place it in the lama/big-lama folder
3. HiNet
   - The purpose of using HiNet is to conceal detected individuals in images, allowing for the removal of individuals while still enabling restoration
   - Implementation of paper：[ICCV 2021 Open Access Repository](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_HiNet_Deep_Image_Hiding_by_Invertible_Network_ICCV_2021_paper.html)
   -  Download pre-trained model: https://github.com/TomTomTommi/HiNet
   - Place it in the HiNet/model folder
4. Fast-reid
   - The purpose of using Fast-reid is to match detected individuals with specific individuals that need protection, in order to identify those requiring protection in images or videos
   - Implementation of paper：[[2006.02631\] FastReID: A Pytorch Toolbox for General Instance Re-identification](https://arxiv.org/abs/2006.02631)
   - Download pre-trained model: https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md
   - Place it in the yolov7/fast_reid_master/model folder

## test

### Profile Path

In the files yolo_fastreid_lama.py、 anony_recory.py and yolov7/yolo_fastreid.py, the.../ replace with one's own actual work path

### Select protected object

Run the image_cut.py file, select the pedestrians to be protected from the original video or image, and capture them. The captured images will be saved to the query folder of the pedestrian re identification module. You can select one or more pedestrians from the picture to capture. After selecting the clipping area, press the ENTER key to save. To exit, do not select any area. Press the ENTER key to exit.

```python
python image_cut.py
```

![image](https://github.com/user-attachments/assets/4216e7ee-a7e1-4609-ab99-0840db1f721c)


### Encryption process

1. Put the pedestrian images to be protected into /yolov7/fast-reid_master/datasets/query
2. You can place the image in example/input image and modify the corresponding path in yolo_fastreid_lama.py. Then run yolo_fastreid_lama.py

```python
python yolo_fastreid_lama.py
```
#### Single person scene

https://github.com/user-attachments/assets/8519d0e8-588b-46c9-950d-fe0a5f112266

#### Multiplayer scene

https://github.com/user-attachments/assets/761db510-e56a-4527-887c-7cfde7ad3f4f


#### Personnel intensive scene

https://github.com/user-attachments/assets/d9b4977e-f65f-43fe-96d2-ef0fd4cae5db

### Decryption process

```python
python anony_recory.py
```

#### Single person scene

https://github.com/user-attachments/assets/4690694c-b1cd-41ed-a4c9-f16e92495954

#### Multiplayer scene
The feature is expected to be updated next time.

#### Personnel intensive scene

https://github.com/user-attachments/assets/9cc61858-1049-4d87-97eb-0606a252112a



