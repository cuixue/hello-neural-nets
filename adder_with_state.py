import theano


def build_adder():
    x = theano.tensor.scalar()
    y = theano.tensor.scalar()
    z = x + y
    run_count = theano.shared(value=0)
    update_run_count = (run_count, run_count + 1)
    return theano.function(
        inputs=[x, y],
        updates=[update_run_count],
        outputs=[z, run_count])


adder = build_adder()
for i in range(5):
    a, b = i, 2
    value, count = adder(a, b)
    print('The sum of {} and {} is {}. You have run this function {} times.'.format(a, b, value, count))
