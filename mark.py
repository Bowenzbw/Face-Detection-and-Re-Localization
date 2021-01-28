import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
from PIL import Image

from matplotlib.patches import Circle
import os
import sys
sys.path.append(os.getcwd())


def mark_face(im_array, dets, landmarks, save_name, mark):
    import matplotlib.pyplot as plt
    import random
    import pylab

    figure = pylab.figure(dpi=600)
    pylab.imshow(im_array)

    for i in range(dets.shape[0]):
        if i not in mark[0:4]:
            continue
        bbox = dets[i, :4]

        if i == mark[0]:
            color = 'white'
        elif i == mark[1]:
            color = 'red'
        elif i == mark[2]:
            color = 'yellow'
        elif i == mark[3]:
            color = 'blue'

        rect = pylab.Rectangle((bbox[0], bbox[1]),
                               bbox[2] - bbox[0],
                               bbox[3] - bbox[1],
                               fill=False,
                               edgecolor=color,
                               linewidth=1.5)
        pylab.gca().add_patch(rect)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            if i not in mark:
                continue
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):
                # pylab.scatter(landmarks_one[j, 0], landmarks_one[j, 1], c='yellow', linewidths=0.1, marker='x', s=5)

                cir1 = Circle(xy=(landmarks_one[j, 0], landmarks_one[j, 1]),
                              radius=2,
                              alpha=0.4,
                              color="red")
                pylab.gca().add_patch(cir1)
        pylab.axis('off')
        pylab.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=72)
        # pylab.show()


if __name__ == "__main__":

    pnet, rnet, onet = create_mtcnn_net(
        p_model_path="./original_model/pnet_epoch.pt",
        r_model_path="./original_model/rnet_epoch.pt",
        o_model_path="./original_model/onet_epoch.pt",
        use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet,
                                   rnet=rnet,
                                   onet=onet,
                                   min_face_size=24)

    origin = []
    target = []
    thresh = 0.87
    f = open("./similarity(1).txt")
    for line in f:
        s = line.split('_')
        o = s[2].split('.')[0]
        t = s[-1].split('.')[0]
        similarity = s[4].split(' ')[-1][:-1]
        similarity = float(similarity)
        if similarity < thresh:
            origin.append(int(o))
            target.append(int(t))
    assert (len(origin) == len(target))

    img = cv2.imread("./o.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    obboxs, olandmarks = mtcnn_detector.detect_face(img)
    save_name = 'o_m_limit.png'
    mark_face(img_bg, obboxs, olandmarks, save_name, origin)

    img = cv2.imread("./t.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tbboxs, tlandmarks = mtcnn_detector.detect_face(img)
    save_name = 't_m_limit.png'
    mark_face(img_bg, tbboxs, tlandmarks, save_name, target)
