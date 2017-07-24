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

# **Dirty our hand a little bit**

Now it's the time to write some code. We are going to use an OOP approach which
is using an object to encapsulate the whole loading pipeline. First of all, we
need a class skeleton. In this case, it's pretty simple, since we only need a
constructor, a get batch function and and some helper "private" functions. The
code looks like this:

```python
class MultiThrdLoader(object):

    def __init__(self, sess, data_root, img_dir, label_csv, file_idxs, \
                    data_shape=(256, 256, 3), label_shape=(17), num_thrds=10):
        pass

    def _process_labels(self):
        pass

    def _process_labels(self):
        pass

    def get_batch_op(self, n):
        pass

    def stop(self):
        pass

```

# **Dig deeper**

Next we need to fill in the body of the functions. We will start with the
constructor

Constructor assumes the responsibility of creating queues and starting the
threads. We begin with saving the parameters, which has little excitement at all.

```python
self.data_root = data_root
self.label_csv = os.path.join(data_root, label_csv)
self.img_dir = os.path.join(data_root, img_dir)
self.data_shape = data_shape
self.label_shape = label_shape
self.sess = sess
```

Then comes the fun parts of creating the queues. But before we create any queue, we
want to declare a coordinator with [`tf.train.Coordinator()`](https://www.tensorflow.org/api_docs/python/tf/train/Coordinator).
According to the Tensorflow official doc, the coordinator implements a simple mechanism to
coordinate the termination of a set of threads. So, basically a coordinator is just
a shared (global) thread-safe flag between threads denotes whether it's time to terminate.

```python
self.coord = tf.train.Coordinator()
```


