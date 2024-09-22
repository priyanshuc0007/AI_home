import cv2
import pixellib
import pixellib.instance import instance_segmentation
image_segmet=instance_segmentation("mask_rcnn_coco.h5")
image_segmet.load_model()
camera=cv2.VideoCapture()

while camera.isOpened():
    res,frame=camera.read()
    result=image_segmet.segment_image(frame,show_boxes=True)
    cv2.imshow('frame',result)

    if cv2.waitKey(10) & 0xff==ord("q"):
        break


camera.release()
cv2.destroyAllWindows
