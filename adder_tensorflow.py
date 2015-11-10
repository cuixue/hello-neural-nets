import tensorflow as tf


def build_adder(x, y):
    return x + y


x = tf.placeholder('float')
y = tf.placeholder('float')
adder = build_adder(x, y)
with tf.Session() as sess:
    four = adder.eval(feed_dict={x: 2, y: 2})
    print "Two plus two is {}".format(four)
