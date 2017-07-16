---
title: "Data loading pipeline in Tensorflow"
date: 2017-07-15T10:58:00-05:00
---
Data is the most critical part for deep learning. The very popularity of deep
neural nets roots in the huge amount of data we possess now. However, dealing
with large dataset is always tricky, since if not properly handled, it might
significantly slow down the training pipeline or bias the model we trained.

Tensorflow, the most popular deep learning framework of the time, supports
the efficient loading of data as well. In this article we will look into how to
make use of the queues and threads in Tensorflow to seamlessly stream a huge
dataset.

# **Requirements**

* Tensorflow: The version tested is 1.2.1. To check the version of your
Tensorflow, run the following command:

    ```bash
    python -c 'import tensorflow as tf; print(tf.__version__)'
    ```
* Dataset: Techniquely, you can use any dataset you want. In this article, we
  will use the dataset of a Kaggle competition: [Planet: Understanding the
  Amazon from
  Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data).
  Download the train-jpg.tar.7z and train_v2.csv.zip.

* Pandas: We will need it to parse the csv file containing lables.

# **A high level overview**

We will make use of the queueing system in Tensorflow. First of all, we create a
file name queue, which will randomly shuffle the order of the file names
dequeued, since we need to randomly shuffle the batch. Then, we create a second
queue. This time the queue is used to hold data items waiting to be feeded into
the training algorithm. In the middle of the file name queue and the data queue,
are the data loader threads. We employ multiple threads of loader to parallelize
the loading, and therefore increase the throughput. Fortunately, the queue in
Tensorflow is already thread safe, so in each loader thread, we can simply read
a file name from file name queue, and push the loaded data items into the data
queue without no worry about synchronization. The GIF from Tensorflow official
doc illustrates this process nicely.

![data loading](https://www.tensorflow.org/images/AnimatedFileQueues.gif)



