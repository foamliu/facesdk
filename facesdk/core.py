import math
import os
import pathlib

import cv2 as cv
import numpy as np
import torch
from facesdk.align_faces import get_reference_facial_points, warp_and_crop_face
from facesdk.mobilefacenet import MobileFaceNet
from facesdk.retinaface.detector import detect_faces
from torchvision import transforms

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
        facesdk_folder = pathlib.Path(__file__).parent.absolute()
        model_path = os.path.join(facesdk_folder, 'model.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.features = dict()

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
        self.features[face_id] = feature

    def recognize(self, img):
        bboxes, landmarks = detect_faces(img)
        face_ids = []
        for i in range(len(bboxes)):
            x0 = self.get_feature(img, landmarks[i])
            features = np.array(list(self.features.values()))
            cosines = np.dot(features, x0)
            idx = int(np.argmax(cosines))
            cosine = cosines[idx]
            cosine = np.clip(cosine, -1.0, 1.0)
            theta = math.acos(cosine)
            theta = theta * 180 / math.pi
            if theta < threshold:
                face_id = list(self.features.keys())[idx]
            else:
                face_id = 'unknown'
            # print('face_id: ' + str(face_id))
            face_ids.append(face_id)

        return bboxes, face_ids
