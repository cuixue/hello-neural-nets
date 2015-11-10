import tensorflow as tf


def build_adder(a, b):
    x = tf.constant(a)
    y = tf.constant(b)
    return x + y


if __name__ == '__main__':
    z = build_adder(2, 2)
    with tf.Session():
        print("Two plus two is {}".format(z.eval()))
