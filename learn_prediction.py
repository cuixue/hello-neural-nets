import theano
import numpy
import random


def build_classifier():
    # Input is a vector of two numbers, each one between 0 and 5
    # input_values is a vector of 2 floats
    input_values = theano.tensor.vector()

    # Append 1.0 to the inputs (this has the same effect as adding a 'b' vector)
    # x is a vector of 3 floats
    x = theano.tensor.join(0, [1.0], input_values)

    # Initialize a matrix of weights which will be updated by training
    # W is a matrix with 3 rows and 11 columns
    initial_W0 = numpy.random.rand(3,11)
    W = theano.shared(value=initial_W0, name='W', borrow=True)

    # This computes a probability distribution over the 11 output classes
    # Type: p_y_given_x is a vector of 11 floats
    p_y_given_x = theano.tensor.nnet.softmax(theano.tensor.dot(x, W))[0]

    # This chooses the integer index of the class with max probability
    # Type: y_pred is a vector of 1 integer
    y_pred = theano.tensor.argmax(p_y_given_x)

    # This function will be wrong until we train it by updating W
    # input is a vector size (2,) and output is an integer
    predict = theano.function(
        inputs=[input_values],
        outputs=y_pred)

    # The expected output label is a number between 0 and 10. Type: scalar integer
    input_label = theano.tensor.iscalar()

    # This is the negative log likelihood. Type: scalar float
    cost = -theano.tensor.log(p_y_given_x)[input_label]

    # Theano computes the gradient via automatic differentiation
    # g_W is a matrix of floats with 3 rows and 11 columns
    g_W = theano.tensor.grad(cost=cost, wrt=W)

    # Each update is a (before, after) double. We have one update.
    # Update W by the gradient (scaled by a learning rate of 0.01)
    updates = [(W, W - 0.01 * g_W)]

    # Each time this function is called, the updates will run once
    train = theano.function(
        inputs=[input_values, input_label],
        updates=updates
    )
    return train, predict


def demonstrate(predict):
    for i in range(4):
        for j in range(4):
            output = predict([i,j])
            print("I think {} + {} = {}".format(i, j, output))


def gradient_descent(train, iterations):
    print("Training for {} iterations...".format(iterations))
    for i in range(iterations):
        in_x = random.randint(0, 4)
        in_y = random.randint(0, 4)
        train([in_x, in_y], in_x + in_y)


train, predict = build_classifier()
demonstrate(predict)
gradient_descent(train, 1000)
demonstrate(predict)
gradient_descent(train, 100000)
demonstrate(predict)
