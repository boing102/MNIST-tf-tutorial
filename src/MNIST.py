import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# Placeholders for input images and output classes
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weights and bias, represented by tf variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Assigns initial values to variables
sess.run(tf.initialize_all_variables())

# multiplying input image by weights and adding bias
y = tf.matmul(x,W) + b

# loss function is the cross-entropy between target and activation functon
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# minimizing cross-entropy with a step length of 0.5 using gradient descent
# train step is operation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# running the operation for a batch from the training set
for i in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# getting correct and predicted values and comparing them
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# casting to float and getting mean to get accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

