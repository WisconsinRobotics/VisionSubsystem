import tensorflow as tf
import numpy as np
import os
import cv2

# use this OR TensorFlow Serving, not both
# see:
#   - https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
#   - https://stackoverflow.com/questions/45640951/tensorflow-classifier-export-savedmodel-beginner/48329456#48329456
def main():
    # preprocess images
    # images to be predicted should be placed in: ./tb_test_images
    # NOTE: change the first index to the number of images in the directory
    predict_data = np.zeros(shape=(10, 96, 96, 3))
    predict_directory = "./tb_test_images"
    for idx, img in enumerate(os.listdir(predict_directory)):
        loaded_img = cv2.imread(predict_directory + '/' + img)
        resized_img = cv2.resize(loaded_img, (96, 96))
        resized_img = (resized_img / (np.max(resized_img)/2)) - 1
        predict_data[idx] = resized_img
    predict_data = predict_data.astype(np.float32)

    assert not np.any(np.isnan(predict_data))

    print(type(predict_data))
    print(predict_data.shape)
    predict_data = predict_data.tolist()
    print(type(predict_data))

    # DEBUG
    #exit()

    # predict
    full_model_dir = "./tb_cnn_model_serve/1524374350"
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], full_model_dir)
        predictor = tf.contrib.predictor.from_saved_model(full_model_dir)
        print("before test")
        test = tf.train.FloatList(value=predict_data)
        print("after test")
        model_input = tf.train.Example(features=tf.train.Features(features={"x": tf.train.Feature(float_list=tf.train.FloatList(value=predict_data))}))
        model_input = model_input.SerializeToString()
        output_dict = predictor({"predictor_inputs":[model_input]})
        y_predicted = output_dict["pred_output_classes"][0]
        print(y_predicted)

if __name__ == "__main__":
  main()
