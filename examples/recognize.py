import cv2 as cv

from facesdk.core import FaceSDK


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


def register(facesdk, name):
    print('Face registration for {}...'.format(name))
    facesdk.register('examples/data/{0}.jpg'.format(name), name)


if __name__ == "__main__":
    filename = 'examples/data/aqgy.jpg'
    img = cv.imread(filename)
    persons = ['chenmeijia', 'guangu', 'huyifei', 'linwanyu', 'luzhanbo', 'lvziqiao', 'tangyouyou', 'zengxiaoxian',
               'zhangwei']

    facesdk = FaceSDK()
    for name in persons:
        register(facesdk, name)

    print('Face recognition for {}...'.format(filename))
    bboxes, face_ids = facesdk.recognize(img)

    img = draw_bboxes(img, bboxes, face_ids)
    cv.imshow('Face recognition', img)
    cv.waitKey(0)

    cv.imwrite('examples/output/face_recognition.jpg', img)
