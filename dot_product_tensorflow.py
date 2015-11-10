import tensorflow as tf


def build_dotter(x, y):
    return tf.mul(x, y)


x = tf.placeholder('float')
y = tf.placeholder('float')
adder = build_dotter(x, y)
a = [1,2,3]
b = [4,5,6]
with tf.Session() as sess:
    result = adder.eval(feed_dict={x: a, y: b})
    print('The dot product of {} and {} is {}'.format(a, b, result))
