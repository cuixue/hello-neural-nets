import tensorflow as tf


def build_adder(x, y):
    return x + y


def build_counter():
    count = tf.Variable(0)
    update_count = tf.assign(count, count + 1)
    return count, update_count


x = tf.placeholder('float')
y = tf.placeholder('float')
adder = build_adder(x, y)
get_count, update_count = build_counter()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(5):
        a, b = i, i*2
        value = adder.eval(feed_dict={x: a, y: b})
        update_count.eval()
        count = get_count.eval()
        print "The sum of {} and {} is {}. The function has been run {} times".format(a, b, value, count)
