%tensorflow_version 1.x
import tensorflow as tf
import numpy as np

# Import the mLSTM babbler model
from unirep import babbler1900 as babbler


# Set seeds
tf.set_random_seed(42)
np.random.seed(42)

# Where model weights are stored.
MODEL_WEIGHT_PATH = "blac_unirep_global_init_1"

# We next need to define a function that allows us to calculate the length each sequence in the batch,
# so that we know what index to use to obtain the right "final" hidden state
def nonpad_len(batch):
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=1)
    return lengths

# Initialize UniRep, also referred to as the "babbler" in our code.
# You need to provide the batch size you will use and the path to the weight directory.
batch_size = 12

b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)

"""Before you can train your model, we need to save sequences in the correct format. This is your ~21 training sequences from GRAPE paper.

"MAXIMUM SEQUENCE LENGTH WE CAN SUPPORT IS 275 AS BEYOND THAT IT GETS UNWIELDY." Our input is 290 AA's long so modifying it to have 300 as the cutoff.

By default format_seq does not include the stop symbol (25) at the end of the sequence. This is the correct behavior if you are trying to train a top model, but not if you are training UniRep representations.

### THIS STEP HAS ALREADY BEEN DONE IN "format_input_seqs.ipynb"

Now we can use a custom function to bucket, batch and pad sequences. The bucketing occurs in the graph.
What is bucketing? Specify a lower and upper bound, and interval. All sequences less than lower or greater than upper will be batched together. The interval defines the "sides" of buckets between these bounds. Don't pick a small interval for a small dataset, because the function will just repeat a sequence if there are not enough to fill a batch.

All batches are the size you passed when initializing the babbler. This also shuffles sequences randomly by sampling from a 10000 sequence buffer automatically pads sequences with 0's so the returned batch is a perfect rectangle automatically repeating the dataset.

Inconveniently, this does not make it easy for a value to be associated with each sequence and not lost during shuffling. You can get around this by just pre-pending every integer sequence with the sequence label (eg, every sequence would be saved to the file as "{brightness value}, 24, 1, 5,..." and then you could just index out the first column after calling the bucket_op.
###  I DID THE ABOVE IN "format_input_seqs.ipynb"
"""

bucket_op = b.bucket_batch_pad("wt_mutants_formatted.txt", interval=1000) # Large interval

# Now that we have a bucket_op we can simply ses.run() it to get a correctly formatted batch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fit_and_batch = sess.run(bucket_op)
    
print(fit_and_batch)
print(fit_and_batch.shape)

# Slice off fitness scores and store them in an array in the same order as the sequences 
# also reshape fitness to fit the input required for training (see where we set y = fitness below)
fitness = np.reshape(fit_and_batch[:,0], (12,1))
batch = fit_and_batch[:,1:]
print(fitness)
print(fitness.shape)
print(batch)
print(batch.shape)

# First, obtain all of the ops needed to output a representation
# final_hidden should be a batch_size x rep_dim matrix.
final_hidden, x_placeholder, batch_size_placeholder, seq_length_placeholder, initial_state_placeholder = (
    b.get_rep_ops())

# to train a basic FF network as the top model doing regression with MSE loss, and the Adam optimizer.
# 1.) define loss function
# 2.) define an optimizer thats only optimizing variables in the top model
# 3.) minimize the loss inside of a TF session
y_placeholder = tf.placeholder(tf.float32, shape=[None,1], name="y")
initializer = tf.contrib.layers.xavier_initializer(uniform=False)

with tf.variable_scope("top"):
    prediction = tf.contrib.layers.fully_connected(
        final_hidden, 1, activation_fn=None, 
        weights_initializer=initializer,
        biases_initializer=tf.zeros_initializer()
    )

loss = tf.losses.mean_squared_error(y_placeholder, prediction)

# You can specifically train the top model first by isolating variables of the "top" scope,
# and forcing the optimizer to only optimize these.
learning_rate=.001
top_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="top")
optimizer = tf.train.AdamOptimizer(learning_rate)
top_only_step_op = optimizer.minimize(loss, var_list=top_variables)
all_step_op = optimizer.minimize(loss)

"""We are ready to train."""

# here we optimize just the top model
y = fitness
num_iters = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        batch = sess.run(bucket_op)
        length = nonpad_len(batch)
        loss_, __, = sess.run([loss, top_only_step_op],
                feed_dict={
                     x_placeholder: batch,
                     y_placeholder: y,
                     batch_size_placeholder: batch_size,
                     seq_length_placeholder:length,
                     initial_state_placeholder:b._zero_state
                }
        )
                  
        print("Iteration {0}: {1}".format(i, loss_))

"""## TODO: SOMEHOW TRANSFER The results from cell below TO DIRECTED EVOLUTION INPUT."""

# here we jointly train the top model and the mLSTM.
# note the model requires a GPU with at least 16GB RAM.
y = fitness
num_iters = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        batch = sess.run(bucket_op)
        length = nonpad_len(batch)
        loss_, __, = sess.run([loss, all_step_op],
                feed_dict={
                     x_placeholder: batch,
                     y_placeholder: y,
                     batch_size_placeholder: batch_size,
                     seq_length_placeholder:length,
                     initial_state_placeholder:b._zero_state
                }
        )
        
        print("Iteration {0}: {1}".format(i,loss_))

