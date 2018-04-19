import os
import numpy as np
import tensorflow as tf
import cv2
import csv

def cnn_model_fn(features, labels, mode):
  """
  Model function for CNN.
  """

  # Input Layer
  #   - 96x96
  # [batch_size, width, height, channels]
#  input_layer = tf.reshape(features["x"], [-1, 96, 96, 3])

  # Feature Extractor:
  # ---------------------------------------------------------------------------------------------------------
  # Convolutional Layer #1
  # 15x15 kernel, 50 filters
  # Input: [batch_size, 96, 96, 3]
  # Output: [batch_size, 82, 82, 50]
  # https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
  conv1 = tf.layers.conv2d(
      inputs=features["x"],
      filters=50,
      kernel_size=[15, 15],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # 6x6 pool, stride size of 2
  # Input: [batch_size, 82, 82, 30]
  # Output: [batch_size, 39, 39, 30]
  # *see above ^
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[6, 6], strides=2)

  # Convolutional Layer #2
  # 5x5 kernel, 100 filters
  # Input: [batch_size, 39, 39, 30]
  # Output: [batch_size, 35, 35, 100]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=100,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # 3x3 pool, stride size of 2
  # Input: [batch_size, 35, 35, 100]
  # Output: [batch_size, 17, 17, 100]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

  # Determination:
  # ---------------------------------------------------------------------------------------------------------
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 17, 17, 100]
  # Output Tensor Shape: [batch_size, 17 * 17 * 100]
  pool2_flat = tf.reshape(pool2, [-1, 17 * 17 * 100])

  # Dense Layer
  # Input Tensor Shape: [batch_size, 17 * 17 * 150]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2)

  # Results:
  # ---------------------------------------------------------------------------------------------------------
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, depth=2), logits=logits))

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    print("logits layer: ")
    print(type(logits))
    print(logits.get_shape().as_list())
    print("loss layer: ")
    print(type(loss))
    print(loss.get_shape().as_list())
    loss_val = tf.losses.compute_weighted_loss(losses=loss)
    print("loss value: ")
    print(type(loss_val))
    print(loss_val)

    # DEBUG
    #exit()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss_val, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and testing data
  train_data = np.zeros(shape=(200, 96, 96, 3))
  train_labels = np.zeros(shape=(200))
  test_data = np.zeros(shape=(50, 96, 96, 3))
  test_labels = np.zeros(shape=(50))
  train_directory = "./temp_data/alex-training-data"
  train_labels_file = "./temp_data/train-labels.csv"
  test_directory = "./temp_data/alex-testing-data"
  test_labels_file = "./temp_data/test-labels.csv"
  for idx, img in enumerate(os.listdir(train_directory)):
      loaded_img = cv2.imread(train_directory + '/' + img)
      resized_img = cv2.resize(loaded_img, (96, 96))
      resized_img = (resized_img / (np.max(resized_img)/2)) - 1
      train_data[idx] = resized_img
  with open(train_labels_file, newline='') as csvfile:
      csvrdr = csv.reader(csvfile, delimiter=' ')
      for idx, r in enumerate(csvrdr):
          train_labels[idx] = int(r[0])

  for idx, img in enumerate(os.listdir(test_directory)):
      loaded_img = cv2.imread(test_directory + '/' + img)
      resized_img = cv2.resize(loaded_img, (96, 96))
      resized_img = (resized_img / (np.max(resized_img)/2)) - 1
      test_data[idx] = resized_img
  with open(test_labels_file, newline='') as csvfile:
      csvrdr = csv.reader(csvfile, delimiter=' ')
      for idx, r in enumerate(csvrdr):
          test_labels[idx] = int(r[0])
  train_data = train_data.astype(np.float32)
  train_labels = train_labels.astype(np.int32)
  test_data = test_data.astype(np.float32)
  test_labels = test_labels.astype(np.int32)

  assert not np.any(np.isnan(train_data))
  assert not np.any(np.isnan(train_labels))
  assert not np.any(np.isnan(test_data))
  assert not np.any(np.isnan(test_labels))

  # DEBUG
  #exit()

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./tb_cnn_model")

  # Set up logging for predictions, specifically "probabilities" from "softmax" tensor
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=10,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=10000,
      hooks=[logging_hook])

  # Test
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  # Results
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
