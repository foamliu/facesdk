import cv2 as cv

from facesdk import FaceSDK


def draw_bboxes(img, bboxes, face_ids):
    num_faces = bboxes.shape[0]

    # show image
    for i in range(num_faces):
        b = bboxes[i]
        text = face_ids[i]
        b = list(map(int, b))
        cv.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv.putText(img, text, (cx, cy),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    return img


if __name__ == "__main__":
    img = cv.imread('examples/data/aqgy.jpg')

    facesdk = FaceSDK()

    facesdk.register('examples/data/chenmeijia.jpg', 'chenmeijia')
    facesdk.register('examples/data/guangu.jpg', 'guangu')
    facesdk.register('examples/data/huyifei.jpg', 'huyifei')
    facesdk.register('examples/data/linwanyu.jpg', 'linwanyu')
    facesdk.register('examples/data/luzhanbo.jpg', 'luzhanbo')
    facesdk.register('examples/data/lvziqiao.jpg', 'lvziqiao')
    facesdk.register('examples/data/tangyouyou.jpg', 'tangyouyou')
    facesdk.register('examples/data/zengxiaoxian.jpg', 'zengxiaoxian')
    facesdk.register('examples/data/zhangwei.jpg', 'zhangwei')

    bboxes, face_ids = facesdk.recognize(img)

    img = draw_bboxes(img, bboxes, face_ids)
    cv.imshow('face recognition', img)
    cv.waitKey(0)
