import input_data
mnist = input_data.read_data_sets(".\\data\\", one_hot=True)

import tensorflow as tf

learning_rate = 0.01
training_iteration = 100
batch_size = 200
display_step = 1

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

W = tf.Variable(tf.zeros([784, 10]), name="Weights")
b = tf.Variable(tf.zeros([10]), name="Biases")

with tf.name_scope('xW_b') as scope:
    model = tf.nn.softmax(tf.matmul(x, W) + b)

W_hist = tf.summary.histogram("Weights", W)
b_hist = tf.summary.histogram("Biases", b)

with tf.name_scope('cost_function') as scope:
    cost_function = -tf.reduce_sum(y * tf.log(model)) # -sum( q(x) * log(p(x)) )
    tf.summary.scalar('cost_function', cost_function)

with tf.name_scope('train') as scope:
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost_function)
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.summary.FileWriter(".\\data\\logs", graph=sess.graph)

    for iteration in range(training_iteration):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
            avg_cost += sess.run(cost_function, feed_dict={x:batch_xs, y:batch_ys}) / total_batch

            summary_str = sess.run(merged_summary_op, feed_dict={x:batch_xs, y:batch_ys})
            summary_writer.add_summary(summary_str, iteration * total_batch + i)

        if iteration % display_step == 0:
            print("Iteration :", "{:04d}".format(iteration + 1), "cost =", "{:.9f}".format(avg_cost))

    print("Tuning Completed!")

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuarcy :", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
