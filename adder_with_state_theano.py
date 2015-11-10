import theano


def build_adder():
    x = theano.tensor.scalar()
    y = theano.tensor.scalar()
    z = x + y
    run_count = theano.shared(value=0)
    update_run_count = (run_count, run_count + 1)
    get_count = theano.function(inputs=[], outputs=run_count)
    add = theano.function(
        inputs=[x, y],
        updates=[update_run_count],
        outputs=[z])
    return add, get_count


adder, get_count = build_adder()
for i in range(5):
    a, b = i, i*2
    value, = adder(a, b)
    count = get_count()
    print('The sum of {} and {} is {}. You have run this function {} times.'.format(a, b, value, count))
