import theano


def build_adder():
    x = theano.tensor.scalar()
    y = theano.tensor.scalar()
    z = x + y
    return theano.function(
        inputs=[x, y],
        outputs=z)


if __name__ == '__main__':
    adder = build_adder()
    print('Two plus two is {}'.format(adder(2, 2)))
