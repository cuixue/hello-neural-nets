import theano


def build_adder():
    x = theano.tensor.scalar()
    y = theano.tensor.scalar()
    z = x + y
    return theano.function(
        inputs=[x, y],
        outputs=z)


adder = build_adder()
print('Two plus two is {}'.format(adder(2, 2)))
