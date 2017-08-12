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

    def _load_worker(self, sess):
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
want to define a coordinator with [`tf.train.Coordinator()`](https://www.tensorflow.org/api_docs/python/tf/train/Coordinator).
According to the Tensorflow official doc, the coordinator implements a simple mechanism to
coordinate the termination of a set of threads. You can think it as a global shared flag variable passing information from the main thread to the spawned threads.

```python
self.coord = tf.train.Coordinator()
```

For our training task, we would like the batches to be randomly created to meet the stochastic assumption of the optimization algorithm. There are two options, either randomly suffle the file identifiers or suffle the loaded images. We are going for the first one.

## Create a randomly shffled file id queue

```python
self.y, self.file_ids = self._process_labels()

self.file_idx_q = tf.RandomShuffleQueue(10000, 0, dtypes=[tf.int32], shapes=[()])
self.eq_fn_op = self.file_idx_q.enqueue_many([file_idxs])

self.fn_qr = tf.train.QueueRunner(self.file_idx_q, [self.eq_fn_op])
fn_thrds = self.fn_qr.create_threads(sess, coord=self.coord, start=True)
```

First, we generate the one-hot representation of the labels and the integer id for each image file (since we can easily map them back to the actual file names). Then we create a [`tf.RandomShuffleQueue`](https://www.tensorflow.org/api_docs/python/tf/RandomShuffleQueue) instance with capacity of 10000 (this value doesn't really matter since enqueuing file ids doesn't take too much time). To enqueue values as a batch, we use the [`enqueue_many`](https://www.tensorflow.org/api_docs/python/tf/RandomShuffleQueue#enqueue_many) function, which returns a tensor. Note that simply calling `enqueue_many` won't actually do the enqueue operation. Only when you evaluate the returned tensor will get those values into the queue. Lastly, the [`tf.train.QueueRunner`](https://www.tensorflow.org/api_docs/python/tf/train/QueueRunner) is a thread runs the enqueue tensor and [`create_threads`](https://www.tensorflow.org/api_docs/python/tf/train/QueueRunner#create_threads) starts the threads. Since here we are enqueuing the same values all the time, the generic runner is fine.

## Create a FIFO data queue

```python
self.data_q = tf.FIFOQueue(500, dtypes=[tf.float32, tf.float32, tf.int32], shapes=[data_shape, label_shape, ()])
self.X_holder = tf.placeholder(tf.float32)
self.y_holder = tf.placeholder(tf.float32)
self.idx_holder = tf.placeholder(tf.int32)
self.enqueue = self.data_q.enqueue([self.X_holder, self.y_holder, self.idx_holder])
self.fdx_dequeue = self.file_idx_q.dequeue()
self.threads = [threading.Thread(target=self._load_worker, args=(sess,)) for i in range(num_thrds)]
```

Same idea as the file id queue, we create an enqueue operation tensor. The different is, instead of using a fixed value, we use placeholders, which we can populate later. This gives a lot of convenience for the loader worker threads. Another difference is the threads. Here we are manually creating threads with the loader function. Also we get an instance of dequeue operation for the loader function.

## The loader function

```python
def _load_worker(self, sess):
    while not self.coord.should_stop():
        # dequeue one filename from the file name queue
        idx = sess.run(self.fdx_dequeue)
        # load the image
        X = np.ndarray(self.data_shape)
        y = np.ndarray(self.label_shape)
        file_path = os.path.join(self.img_dir, self.file_ids[idx] + ".jpg")
        X = io.imread(file_path)
        y = self.y[idx]
        try:
            sess.run(self.enqueue, feed_dict={
                self.X_holder: X,
                self.y_holder: y,
                self.idx_holder: idx
                })
        except tf.errors.CancelledError:
            return
```

In the dequeue function, several things are interesting. First,

```python
idx = sess.run(self.fdx_dequeue)
```

will perform dequeue action on the file id queue. Second is the way we are enqueuing the images. We wrap the whole enqueue operation in a `try except` block since evaluating enqueue tensor is blocking, therefore we can't terminate the thread if it's blocked on enqueue, unless we set `cancel_pending_enqueues` to be true when closing the queue. We'll get the `tf.errors.CancelledError` exception if enqueue has been cancelled.

## The stop function

```python
def stop(self):
    self.coord.request_stop()
    d_q_clos = self.data_q.close(cancel_pending_enqueues=True)
    self.sess.run(d_q_clos)
    self.coord.join(self.threads)
```
This function is for clean exit. Since we have a lot of threads, we need a way to stop them before exit. With the help of the coordinator, this will be easy, just don't forget to evaluate the returned tensor for closing queue.

This is all we need to do in order to load large dataset in Tensorflow. Also, checkout the full code on [Github](https://github.com/Zonglin-Li6565/Kaggle-Amazon). The [multithrdloader.py](https://github.com/Zonglin-Li6565/Kaggle-Amazon/blob/master/multithrdloader.py) contains the full loader and [train.py](https://github.com/Zonglin-Li6565/Kaggle-Amazon/blob/master/train.py) contains the actual training code.
