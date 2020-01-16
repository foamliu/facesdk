import cv2 as cv

from facesdk.facesdk import FaceSDK


def draw_bboxes(img, bboxes, landmarks):
    num_faces = bboxes.shape[0]

    # show image
    for i in range(num_faces):
        b = bboxes[i]
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv.putText(img, text, (cx, cy),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        landms = landmarks[i]
        cv.circle(img, (landms[0], landms[5]), 1, (0, 0, 255), 4)
        cv.circle(img, (landms[1], landms[6]), 1, (0, 255, 255), 4)
        cv.circle(img, (landms[2], landms[7]), 1, (255, 0, 255), 4)
        cv.circle(img, (landms[3], landms[8]), 1, (0, 255, 0), 4)
        cv.circle(img, (landms[4], landms[9]), 1, (255, 0, 0), 4)

    return img


if __name__ == "__main__":
    # Face detection
    img_raw = cv.imread('data/aqgy.jpg')
    img = img_raw.copy()

    facesdk = FaceSDK()
    bboxes, landmarks = facesdk.detect_faces(img)

    img = draw_bboxes(img, bboxes, landmarks)
    cv.imshow('', img)
    cv.waitKey(0)
