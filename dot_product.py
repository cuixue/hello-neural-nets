import theano


def build_dotter():
    x = theano.tensor.vector()
    y = theano.tensor.vector()
    z = theano.tensor.dot(x, y)
    return theano.function(
        inputs=[x, y],
        outputs=z)


dotter = build_dotter()
a = [1,2,3]
b = [4,5,6]
print('The dot product of {} and {} is {}'.format(a, b, dotter(a, b)))
