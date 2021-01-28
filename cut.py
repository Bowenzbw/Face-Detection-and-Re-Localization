import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
from PIL import Image


if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./target.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align
    save_name = 't_m.jpg'
    vis_face(img_bg, bboxs, landmarks, save_name)
    
    im = Image.open("./target.jpg")
    num_faces = bboxs.shape[0]
    for i in range(num_faces):
        bbox = bboxs[i, :4]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        region = im.crop((bbox[0], bbox[1],
                          bbox[2],
                          bbox[3]))
        region.save("./target/face_target_" + str(i) + ".jpg")
    
    img = cv2.imread("./origin.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    # print box_align
    save_name = 'o_m.jpg'
    vis_face(img_bg, bboxs, landmarks, save_name)
    
    im = Image.open("./origin.jpg")
    num_faces = bboxs.shape[0]
    for i in range(num_faces):
        bbox = bboxs[i, :4]
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        region = im.crop((bbox[0], bbox[1],
                          bbox[2],
                          bbox[3]))
        region.save("./origin/face_origin_" + str(i) + ".jpg")