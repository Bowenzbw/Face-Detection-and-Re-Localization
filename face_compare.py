# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet.src.facenet as facenet
import os
import matplotlib.pyplot as plt


modeldir = './pretrained_model/20180402-114759.pb'  # change to your model dir
similar_rate = 0.9


def img_similarity(img_list1, img_list2):
    # input: two images as np array
    # output: similarity, if the similarity under 1.1, we said two images are same.
    image_size = 160

    print('build facenet embedding model')
    tf.Graph().as_default()
    sess = tf.Session()
    # load pre-trained facenet model
    facenet.load_model(modeldir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    print('facenet embedding built')

    # compare each pair of face images
    similar_img_list = {}
    for i in range(len(img_list1)):
        min_value = 20
        min_index_j = False
        for j in range(len(img_list2)):
            print('{} and {}\n'.format(i, j))
            scaled_reshape = []
            img1 = img_list1[i]
            img2 = img_list2[j]

            image1 = cv2.resize(img1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            image1 = facenet.prewhiten(image1)
            scaled_reshape.append(image1.reshape(-1, image_size, image_size, 3))
            emb_array1 = np.zeros((1, embedding_size))
            emb_array1[0, :] = \
            sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False})[0]

            image2 = cv2.resize(img2, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            image2 = facenet.prewhiten(image2)
            scaled_reshape.append(image2.reshape(-1, image_size, image_size, 3))
            emb_array2 = np.zeros((1, embedding_size))
            emb_array2[0, :] = \
            sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False})[0]

            dist = np.sqrt(np.sum(np.square(emb_array1[0] - emb_array2[0])))
            if dist < min_value:
                min_value = dist
                min_index_j = j
        print(min_value)
        print(min_index_j)
        if min_index_j == 0:
            similar_img_list[min_index_j] = (i, min_index_j, min_value)
        if min_value <= similar_rate and min_index_j:
            if min_index_j in similar_img_list.keys() and min_value < similar_img_list[min_index_j][2]:
                similar_img_list[min_index_j] = (i, min_index_j, min_value)
            elif min_index_j not in similar_img_list.keys():
                similar_img_list[min_index_j] = (i, min_index_j, min_value)
        print(similar_img_list)
    return similar_img_list


def load_img(path, face_list):
    file_name_list = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        file_name_list.append(file)
        img = scipy.misc.imread(img_path, mode='RGB')
        face_list.append(img)
    return face_list, file_name_list


if __name__ == '__main__':
    origin_face_path = './normalize/origin'
    target_face_path = './normalize/target'
    face_list_origin = []
    face_list_target = []
    face_list_origin, origin_file = load_img(origin_face_path, face_list_origin)
    face_list_target, target_file = load_img(target_face_path, face_list_target)
    # obtain the similarity of these two dataset.
    similar_list = img_similarity(face_list_origin, face_list_target)
    print(similar_list)
    # store the result
    with open('similarity.txt', 'w') as f:
        for i in similar_list.values():
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(face_list_origin[i[0]])
            # plt.subplot(1, 2, 2)
            # plt.imshow(face_list_target[i[1]])
            # plt.show()
            f.write('{} {} {}\n'.format(origin_file[i[0]], target_file[i[1]], i[2]))



