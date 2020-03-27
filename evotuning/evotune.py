import argparse
import numpy as np
import tensorflow as tf
from unirep import babbler1900 as babbler
from Bio import SeqIO


def nonpad_len(batch):
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=1)
    return lengths
            

def dump_weights(self,sess,dir_name="./1900_weights"):
    """
    Saves the weights of the model in dir_name in the format required 
    for loading in this module. Must be called within a tf.Session
    For which the weights are already initialized.
    """
    vs = tf.trainable_variables()
    for v in vs:
        name = v.name
        value = sess.run(v)
        print(name)
        print(value)
        np.save(os.path.join(dir_name,name.replace('/', '_') + ".npy"), np.array(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path')
    parser.add_argument('--data_file')
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--iters', default=20000, type=int)
    parser.add_argument('--steps_per_print', default=200, type=int)
    args = parser.parse_args()

    with tf.Session() as sess:
        model = babbler(batch_size=args.batch_size, model_path=args.weight_path)

        (
            logits, 
            loss, 
            x_placeholder, 
            y_placeholder, 
            batch_size_placeholder, 
            initial_state_placeholder
        ) = model.get_babbler_ops()

        optimizer = tf.train.AdamOptimizer(args.learning_rate)

        optimizer_op = optimizer.minimize(loss)
        bucket_op = model.bucket_batch_pad(args.data_file, interval=1000)

        for i in range(args.iters):
            batch = sess.run(bucket_op)
            length = nonpad_len(batch)
            loss_, __ = sess.run(
                [loss, optimizer_op],
                feed_dict={
                     x_placeholder: batch,
                     y_placeholder: batch,
                     batch_size_placeholder: args.batch_size,
                     initial_state_placeholder: model._zero_state
                }
            )

            if (i + 1) % args.steps_per_print == 0:
                print("Iteration {0}: {1}".format(i,loss_))
