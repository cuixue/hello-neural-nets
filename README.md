# Tensorflow and Theano examples

This is a series of simple examples to serve as a bridge between Hello World and MNIST digit recognition.

The first is a hello world showing addition.
Then addition plus some saved state.
Then a dot product.
Then a silly learning task: learning to add two numbers (zero through 4) with logistic regression.

# Common Mistakes
## Uninitialized value Variable

Error message: `tensorflow.python.framework.errors.FailedPreconditionError: Attempting to use uninitialized value Variable`

One tricky stumbling block is the concept of variable initialization.
You need to make sure that you build the graph first, then call `tf.initialize_all_variables()` afterwards.

The following code works:

```
import tensorflow as tf

with tf.Session() as sess:
    count = tf.Variable(0)
    updater = tf.assign(count, count + 1)
    sess.run(tf.initialize_all_variables())
    sess.run(updater)
    print("The count is {}".format(sess.run(count)))
```

The following code fails with `tensorflow.python.framework.errors.FailedPreconditionError: Attempting to use uninitialized value Variable`
```
import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    count = tf.Variable(0)
    updater = tf.assign(count, count + 1)
    sess.run(updater)
    print("The count is {}".format(sess.run(count)))
```
