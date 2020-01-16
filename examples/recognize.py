import cv2 as cv

from facesdk import FaceSDK
from utils import draw_bboxes_det, draw_bboxes_reg

if __name__ == "__main__":
    # Face detection
    img_raw = cv.imread('images/aqgy_0.jpg')
    img = img_raw.copy()

    facesdk = FaceSDK()
    bboxes, landmarks = facesdk.detect_faces(img)

    img = draw_bboxes_det(img, bboxes, landmarks)
    cv.imshow('', img)
    cv.waitKey(0)

    # Face recognition
    facesdk.register('images/chenmeijia.jpg', 'chenmeijia')
    facesdk.register('images/guangu.jpg', 'guangu')
    facesdk.register('images/huyifei.jpg', 'huyifei')
    facesdk.register('images/linwanyu.jpg', 'linwanyu')
    facesdk.register('images/luzhanbo.jpg', 'luzhanbo')
    facesdk.register('images/lvziqiao.jpg', 'lvziqiao')
    facesdk.register('images/tangyouyou.jpg', 'tangyouyou')
    facesdk.register('images/zengxiaoxian.jpg', 'zengxiaoxian')
    facesdk.register('images/zhangwei.jpg', 'zhangwei')
    face_ids = facesdk.recognize(img_raw)
    print(face_ids)

    img = draw_bboxes_reg(img_raw, bboxes, face_ids)
    cv.imshow('', img)
    cv.waitKey(0)
