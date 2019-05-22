import numpy as np
import tensorflow as tf

def central_scale_images(image, depth_image):
    # image shape: (640, 480, 3)
    # depth image: (640, 480, 1)
    scale = np.random.uniform(0.5, 1)
    boxes = np.zeros((1, 4), dtype=np.float32)
    box_ind = np.zeros((1), dtype=np.int32)

    x1 = y1 = 0.5 - 0.5 * scale
    x2 = y2 = 0.5 + 0.5 * scale
    boxes[0] = np.array([y1, x1, y2, x2], dtype=np.float32)
    crop_size = np.array([image.shape[0], image.shape[1]], dtype=np.int32)

    tf.reset_default_graph()
    new_image = tf.placeholder(tf.float32, shape = (1, image.shape[0], image.shape[1], 3))
    new_depth_image = tf.placeholder(tf.float32, shape=(1, image.shape[0], image.shape[1], 1))

    tf_new_image = tf.image.crop_and_resize(new_image, boxes, box_ind, crop_size)
    tf_new_depth_image = tf.image.crop_and_resize(new_depth_image, boxes, box_ind, crop_size)

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        batch_img = np.expand_dims(image, axis = 0)
        scaled_img = sess.run(tf_new_image, feed_dict = {new_image: batch_img})
        batch_depth_img = np.expand_dims(depth_image, axis=0)
        scaled_depth_img = sess.run(tf_new_depth_image, feed_dict={new_depth_image: batch_depth_img})
    
    return scaled_img[0], scaled_depth_img[0] * scale


'''
ORIGINAL CODE

def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data
    
# Produce each image at scaling of 90%, 75% and 60% of original image.
scaled_imgs = central_scale_images(X_imgs, [0.90, 0.75, 0.60])

'''