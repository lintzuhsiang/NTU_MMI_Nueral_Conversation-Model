

"""Most of the code comes from seq2seq tutorial. Binary for training conversation models and decoding from them.

Running this program without --decode will  tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint performs

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""


import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from  data_utils import *
from  seq2seq_model import *
import codecs
from tensorflow.core.protobuf import saver_pb2

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_string("train_dir", "./train/", "Training directory.")
tf.app.flags.DEFINE_string("vocab_path", "./train/", "Data directory")


tf.app.flags.DEFINE_string("data_path_ST", "./mydata/", "Training directory_ST.")
tf.app.flags.DEFINE_string("data_path_TS","./mydata/","Training directory_TS.")


#tf.app.flags.DEFINE_string("dev_data", "./tmp/", "Data directory")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("beam_size", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("beam_search", False,
                            "Set to True for beam_search.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("attention", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_float("MMIweight", 0.5, "the parameter of forward and backword models' weight")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(10, 10), (15, 15), (25, 25), (50, 50)]




def read_chat_data(data_path,vocabulary_path, max_size=None):
    counter = 0
    vocab, _ = initialize_vocabulary(vocabulary_path)
    print len(vocab)
    print max_size
    data_set = [[] for _ in _buckets]
    with codecs.open(data_path, "rb") as fi:
        for line in fi.readlines():
            counter += 1
            if max_size!=0 and counter > max_size:
                break
            if counter % 10000 == 0:
              print("  reading data line %d" % counter)
              sys.stdout.flush()
            entities = line.lower().split("\t")
            # print entities
            if len(entities) == 2:
                source = entities[0]
                target = entities[1]
                source_ids = [int(x) for x in sentence_to_token_ids(source,vocab)]
                target_ids = [int(x) for x in sentence_to_token_ids(target,vocab)]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                  if len(source_ids) < source_size and len(target_ids) < target_size + 1:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
    return data_set

def create_model(session, forward_only, beam_search, beam_size = 10, attention = True):
  """Create translation model and initialize or load parameters in session."""
  model = Seq2SeqModel(
      FLAGS.en_vocab_size, FLAGS.en_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.MMIweight,
      forward_only=forward_only, beam_search=beam_search, beam_size=beam_size, attention=attention)
  print FLAGS.train_dir
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

  # ckpt.model_checkpoint_path ="./train/chat_bot.ckpt-1201.data-00000-of-00001"
  # print ckpt.model_checkpoint_path
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def create_models(path, en_vocab_size, session, forward_only, beam_search, beam_size = 10, attention = True):
  """Create translation model and initialize or load parameters in session."""
  model = Seq2SeqModel(
      en_vocab_size, en_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only, beam_search=beam_search, beam_size=beam_size, attention=attention)
  print FLAGS.train_dir
  ckpt = tf.train.get_checkpoint_state(path)

  # ckpt.model_checkpoint_path ="./big_models/chat_bot.ckpt-183600"
  # print ckpt.model_checkpoint_path
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def train():

  data_path_ST =FLAGS.data_path_ST
  data_path_TS =FLAGS.data_path_TS
  #dev_data = FLAGS.dev_data
  vocab_path =FLAGS.vocab_path
  # Beam search is false during training operation and usedat inference .
  beam_search = False
  beam_size =10
  attention = FLAGS.attention

  normalize_digits=True
  #TODO
  create_vocabulary(vocab_path, data_path_ST, FLAGS.en_vocab_size )
  create_vocabulary(vocab_path, data_path_TS, FLAGS.en_vocab_size )


  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False,beam_search=beam_search, beam_size=beam_size, attention=attention)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    train_set_ST =read_chat_data(data_path_ST,vocab_path, FLAGS.max_train_data_size)
    train_set_TS =read_chat_data(data_path_TS,vocab_path, FLAGS.max_train_data_size)
    print("------------------------------")
    for x in train_set_ST:
        print(len(x))
    print("------------------------------")
    for x in train_set_TS:
        print(len(x))
    print("------------------------------")
    #dev_set =read_chat_data(dev_data,vocab_path, FLAGS.max_train_data_size)

    train_bucket_sizes_ST = [len(train_set_ST[b]) for b in xrange(len(_buckets))]
    train_total_size_ST = float(sum(train_bucket_sizes_ST))

    train_bucket_sizes_TS = [len(train_set_TS[b]) for b in xrange(len(_buckets))]
    train_total_size_TS = float(sum(train_bucket_sizes_TS))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = max([sum(train_bucket_sizes_ST[:i + 1]) / train_total_size_ST
                           for i in xrange(len(train_bucket_sizes_ST))],
                           [sum(train_bucket_sizes_TS[:i + 1]) / train_total_size_TS
                           for i in xrange(len(train_bucket_sizes_TS))])



    # This is the training loop.
    step_time, loss_ST, loss_TS , loss_MMI = 0.0, 0.0, 0.0, 0.0
    current_step = 0
    #TODO
    previous_losses = []
    ## TODO: properly use model.step_ST and model.stepTS here.  
    for i in range(401):
      t1 = time.time()
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      # print "Started"
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs_ST, decoder_inputs_ST, target_weights_ST = model.get_batch(
          train_set_ST, bucket_id)

      encoder_inputs_TS, decoder_inputs_TS, target_weights_TS = model.get_batch(
          train_set_TS, bucket_id)

      _, step_loss_ST, _ = model.step_ST(sess, encoder_inputs_ST, decoder_inputs_ST,
                                    target_weights_ST, bucket_id, False, beam_search)

      _, step_loss_TS, _ = model.step_TS(sess, encoder_inputs_TS, decoder_inputs_TS,
                                    target_weights_TS, bucket_id, False, beam_search)



      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss_ST += step_loss_ST / FLAGS.steps_per_checkpoint
      loss_TS += step_loss_TS / FLAGS.steps_per_checkpoint
      current_step += 1

      perplexity_ST = math.exp(loss_ST) if loss_ST < 300 else float('inf')
      perplexity_TS = math.exp(loss_TS) if loss_TS < 300 else float('inf')
      print ("global step %d learning rate %.4f step-time %.2f perplexity_ST %.2f perplexity_TS "
                "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity_ST, perplexity_TS))

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        print "Running epochs"
        perplexity_ST = math.exp(loss_ST) if loss_ST < 300 else float('inf')
        perplexity_TS = math.exp(loss_TS) if loss_TS < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity_ST %.2f perplexity_TS "
                "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity_ST, perplexity_TS))
        # # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and max(loss_ST,loss_TS) > max(previous_losses[-3:]):
           sess.run(model.learning_rate_decay_op)
        previous_losses.append(max(loss_ST,loss_TS))
        # # Save checkpoint and zero timer and loss.
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
	saver.save(sess, FLAGS.train_dir+ 'model.ckpt', global_step = current_step)
	#checkpoint_path = os.path.join(FLAGS.train_dir, "chat_bot.ckpt")
        #model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss_ST,loss_TS = 0.0, 0.0, 0.0
        # # Run evals on development set and print their perplexity.
        '''
        for bucket_id in xrange(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs_ST, decoder_inputs_ST, target_weights_ST = model.get_batch(
                train_set_TS, bucket_id)

            encoder_inputs_TS, decoder_inputs_TS, target_weights_TS = model.get_batch(
                train_set_TS, bucket_id)

            _, eval_loss_ST, _ = model.step_ST(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True, beam_search)

            _, eval_loss_TS, _ = model.step_TS(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True, beam_search)
            eval_ppx = math.exp(max(eval_loss_TS,eval_loss_ST)) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        '''
        sys.stdout.flush()
      print("time consumed :", time.time() - t1)
     


    # properly use model.step3 here. the code above may help.
    print("Training model MMI.")
    for i in range(401):
      t1 = time.time()
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      # print "Started"
      
      random_number_02 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_02])

      # Get a batch and make a step.
      start_time = time.time()

      encoder_inputs_ST, decoder_inputs_ST, target_weights_ST,encoder_inputs_TS, decoder_inputs_TS, target_weights_TS = model.get_batch_MMI(train_set_ST, train_set_TS, bucket_id)

      _, step_loss_MMI, _ = model.step_MMI(sess,encoder_inputs_ST, decoder_inputs_ST, target_weights_ST,
                                                encoder_inputs_TS, decoder_inputs_TS, target_weights_TS,
                                                bucket_id, False, beam_search)

      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss_MMI += step_loss_MMI / FLAGS.steps_per_checkpoint
      current_step += 1

      perplexity_MMI = math.exp(loss_MMI) if loss_MMI < 300 else float('inf')
      print ("global step %d learning rate %.4f step-time %.2f perplexity_MMI"
                "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity_MMI))
      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        print "Running epochs"
        perplexity_MMI = math.exp(loss_MMI) if loss_MMI < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity_MMI"
                "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity_MMI))
        # # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss_MMI > max(previous_losses[-3:]):
           sess.run(model.learning_rate_decay_op)
        print ("loss_MMI: %.2f" % loss_MMI)
        previous_losses.append(loss_MMI)
        # # Save checkpoint and zero timer and loss.
        #TODO
        print ("loss_MMI: %.2f" % loss_MMI)
        saver = tf.train.Saver(write_version = tf.train.SaverDef.V2)
	saver.save(sess, FLAGS.train_dir + 'MMI_model.ckpt', global_step = current_step)
        #checkpoint_path = os.path.join(FLAGS.train_dir, "chat_bot.ckpt")
        #model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss_MMI = 0.0, 0.0
        # # Run evals on development set and print their perplexity.
        '''
        for bucket_id in xrange(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                dev_set, bucket_id)
            _, eval_loss__MMI, _ = model.step_MMI(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True, beam_search)
            eval_ppx = math.exp(eval_loss_MMI) if eval_loss < 300 else float('inf')
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()
        '''
      print("time consumed :", time.time() - t1)


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    beam_size = FLAGS.beam_size
    beam_search = FLAGS.beam_search
    attention = FLAGS.attention
    model = create_model(sess, True, beam_search=beam_search, beam_size=beam_size, attention=attention)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    vocab_path = FLAGS.vocab_path
    vocab, rev_vocab = initialize_vocabulary(vocab_path)

    # Decode from standard input.
    if beam_search:
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
          # Get token-ids for the input sentence.
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
          # Which bucket does it belong to?
          bucket_id = min([b for b in xrange(len(_buckets))
                           if _buckets[b][0] > len(token_ids)])
          # Get a 1-element batch to feed the sentence to the model.
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
          # Get output logits for the sentence.
          # print bucket_id
          path, symbol , output_logits = model.step_ST(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True,beam_search )

          k = output_logits[0]
          paths = []
          for kk in range(beam_size):
              paths.append([])
          curr = range(beam_size)
          num_steps = len(path)
          for i in range(num_steps-1, -1, -1):
              for kk in range(beam_size):
                paths[kk].append(symbol[i][curr[kk]])
                curr[kk] = path[i][curr[kk]]
          recos = set()
          print "Replies --------------------------------------->"
          for kk in range(beam_size):
              foutputs = [int(logit)  for logit in paths[kk][::-1]]

          # If there is an EOS symbol in outputs, cut them at that point.
              if EOS_ID in foutputs:
          #         # print outputs
                   foutputs = foutputs[:foutputs.index(EOS_ID)]
              rec = " ".join([tf.compat.as_str(rev_vocab[output]) for output in foutputs])
              if rec not in recos:
                      recos.add(rec)
                      print rec

          print("> ", "")
          sys.stdout.flush()
          sentence = sys.stdin.readline()
    else:
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
              # Get token-ids for the input sentence.
              token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
              # Which bucket does it belong to?
              bucket_id = min([b for b in xrange(len(_buckets))
                               if _buckets[b][0] > len(token_ids)])
              # for loc in locs:
                  # Get a 1-element batch to feed the sentence to the model.
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                      {bucket_id: [(token_ids, [],)]}, bucket_id)

              _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                   target_weights, bucket_id, True,beam_search)
              # This is a greedy decoder - outputs are just argmaxes of output_logits.

              outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
              # If there is an EOS symbol in outputs, cut them at that point.
              if EOS_ID in outputs:
                  # print outputs
                  outputs = outputs[:outputs.index(EOS_ID)]

              print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
              print("> ", "")
              sys.stdout.flush()
              sentence = sys.stdin.readline()



def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
