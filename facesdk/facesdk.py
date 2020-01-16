import math

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from facesdk.align_faces import get_reference_facial_points, warp_and_crop_face
from facesdk.mobilefacenet import MobileFaceNet
from facesdk.retinaface.detector import detect_faces

im_size = 112
threshold = 74.27610703463543

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def align_face(img, facial5points):
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (im_size, im_size)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (im_size, im_size)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    dst_img = warp_and_crop_face(img, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


class FaceSDK(object):
    def __init__(self):
        self.model = MobileFaceNet()
        self.model.load_state_dict(torch.load('model.pt'))
        self.model.eval()
        self.features = []

    def detect_faces(self, img):
        return detect_faces(img)

    def get_feature(self, img, landmarks):
        img = align_face(img, landmarks)
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        imgs = img.unsqueeze(dim=0)
        with torch.no_grad():
            output = self.model(imgs)
        feature = output[0].cpu().numpy()
        return feature / np.linalg.norm(feature)

    def get_feature_by_filename(self, filename):
        img = cv.imread(filename)
        _, landmarks = detect_faces(img)
        return self.get_feature(img, landmarks)

    def register(self, filename, face_id):
        feature = self.get_feature_by_filename(filename)
        self.features.append({'feature': feature, 'face_id': face_id})

    def verify(self, x0, x1):
        cosine = np.dot(x0, x1)
        cosine = np.clip(cosine, -1.0, 1.0)
        theta = math.acos(cosine)
        theta = theta * 180 / math.pi
        return theta <= threshold

    def recognize(self, img):
        bboxes, landmarks = detect_faces(img)
        face_ids = []
        for i in range(len(bboxes)):
            x0 = self.get_feature(img, landmarks[i])
            face_id = 'unknown'
            for data in self.features:
                x1 = data['feature']
                if self.verify(x0, x1):
                    face_id = data['face_id']
                    break
            face_ids.append(face_id)

        return face_ids
