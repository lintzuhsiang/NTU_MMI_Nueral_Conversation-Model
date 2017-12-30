
"""Sequence-to-sequence model with an attention mechanism."""


import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from  data_utils import *
from my_seq2seq import *

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, MMIweight, use_lstm=False,
               num_samples=1024, forward_only=False, beam_search = True, beam_size=10, attention=True):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    self.MMIweight = MMIweight
    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      with tf.device("/cpu:0"):
        w = tf.get_variable("proj_w", [size, self.target_vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                            self.target_vocab_size)
      softmax_loss_function = sampled_loss
    # TODO there should be cellST and cellTS
    # Create the internal multi-layer cell for our RNN.
    single_cellST = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      single_cellST = tf.nn.rnn_cell.BasicLSTMCell(size)
    cellST = single_cellST
    if num_layers > 1:
      cellST = tf.nn.rnn_cell.MultiRNNCell([single_cellST] * num_layers)

    single_cellTS = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      single_cellTS = tf.nn.rnn_cell.BasicLSTMCell(size)
    cellTS = single_cellTS
    if num_layers > 1:
      cellTS = tf.nn.rnn_cell.MultiRNNCell([single_cellTS] * num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    # TODO define source2target and target2source
    def seq2seq_ST(encoder_inputs, decoder_inputs, do_decode):
        if attention:
            print "seq2seq_ST Attention Model"
            return embedding_attention_seq2seq(
               encoder_inputs, decoder_inputs, cellST,
               num_encoder_symbols=source_vocab_size,
               num_decoder_symbols=target_vocab_size,
               embedding_size=size,
               output_projection=output_projection,
               feed_previous=do_decode,scope="embedding_attention_seq2seq_ST",
               beam_search=beam_search,
               beam_size=beam_size )
        else:
            print "seq2seq_ST Simple Model"
            return embedding_rnn_seq2seq(
              encoder_inputs, decoder_inputs, cellST,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode,
              beam_search=beam_search,
              beam_size=beam_size )
    # TODO
    def seq2seq_TS(encoder_inputs,decoder_inputs,do_decode):
        if attention:
            print "seq2seq_TS Attention Model"
            return embedding_attention_seq2seq(
               encoder_inputs, decoder_inputs, cellTS,
               num_encoder_symbols=source_vocab_size,
               num_decoder_symbols=target_vocab_size,
               embedding_size=size,
               output_projection=output_projection,
               feed_previous=do_decode,scope="embedding_attention_seq2seq_TS",
               beam_search=beam_search,
               beam_size=beam_size )
        else:
            print "seq2seq_TS Simple Model"
            return embedding_rnn_seq2seq(
              encoder_inputs, decoder_inputs, cellTS,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode,
              beam_search=beam_search,
              beam_size=beam_size )

    # Feeds for inputs.
    self.encoder_inputs_ST = []
    self.decoder_inputs_ST = []
    self.target_weights_ST = []

    self.encoder_inputs_TS = []
    self.decoder_inputs_TS = []
    self.target_weights_TS = []

    '''
    self.encoder_inputs_MMI = []
    self.decoder_inputs_MMI = []
    self.target_weights_MMI = []
    '''
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs_ST.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder_ST{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs_ST.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder_ST{0}".format(i)))
      self.target_weights_ST.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight_ST{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets_ST = [self.decoder_inputs_ST[i + 1]
               for i in xrange(len(self.decoder_inputs_ST) - 1)]

    
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs_TS.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder_TS{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs_TS.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder_TS{0}".format(i)))
      self.target_weights_TS.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight_TS{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets_TS = [self.decoder_inputs_TS[i + 1]
               for i in xrange(len(self.decoder_inputs_TS) - 1)]

    # Training outputs and losses.

    if forward_only:
        if beam_search:
              self.outputs_ST, self.beam_path, self.beam_symbol = decode_model_with_buckets(
                  self.encoder_inputs_ST, self.decoder_inputs_ST, targets_ST,
                  self.target_weights_ST, buckets, lambda x, y: seq2seq_ST(x, y, True),
                  softmax_loss_function=softmax_loss_function)
        else:
              # print self.decoder_inputs
              self.outputs_ST,self.outputs_TS, self.losses_ST,self.losses_TS,self.losses_MMI = model_with_buckets(
                  self.encoder_inputs_ST, self.decoder_inputs_ST,self.encoder_inputs_TS,self.decoder_inputs_TS, targets_ST,targets_TS,
                  self.target_weights_ST, self.target_weights_TS,buckets, lambda x, y: seq2seq_ST(x, y, True),
                  lambda w, z: seq2seq_TS(w, z,True),softmax_loss_function=None ,MMIparam=MMIweight)
              # If we use output projection, we need to project outputs for decoding.
              if output_projection is not None:
                    for b in xrange(len(buckets)):
                      self.outputs[b] = [
                          tf.matmul(output, output_projection[0]) + output_projection[1]
                          for output in self.outputs[b]
                      ]

    else:
        # TODO: losses part: change model_with_buckets's arguments and what it returns
              self.outputs_ST,self.outputs_TS, self.losses_ST,self.losses_TS,self.losses_MMI = model_with_buckets(self.encoder_inputs_ST, self.decoder_inputs_ST, self.encoder_inputs_TS, self.decoder_inputs_TS, targets_ST,targets_TS,
                  self.target_weights_ST, self.target_weights_TS, buckets, lambda x, y: seq2seq_ST(x, y, True),
                  lambda w, z: seq2seq_TS(w, z,True), softmax_loss_function=None, MMIparam=MMIweight)
              

    # TODO: gradient_norms and updates part 
    # Gradients and SGD update operation for training the model.
    params_ST = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="embedding_attention_seq2seq_ST")
    if not forward_only:
      self.gradient_norms_ST = []
      self.updates_ST = []
      #TODO
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients_ST = tf.gradients(self.losses_ST[b], params_ST)
        clipped_gradients_ST, norm_ST = tf.clip_by_global_norm(gradients_ST,
                                                         max_gradient_norm)
        self.gradient_norms_ST.append(norm_ST)
        self.updates_ST.append(opt.apply_gradients(
            zip(clipped_gradients_ST, params_ST), global_step=self.global_step))

    #gradient_norm_TS PART
    
    params_TS = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="embedding_attention_seq2seq_TS")
    if not forward_only:
      self.gradient_norms_TS = []
      self.updates_TS = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients_TS = tf.gradients(self.losses_TS[b], params_TS)
        clipped_gradients_TS, norm_TS = tf.clip_by_global_norm(gradients_TS,
                                                         max_gradient_norm)
        self.gradient_norms_TS.append(norm_TS)
        self.updates_TS.append(opt.apply_gradients(
            zip(clipped_gradients_TS, params_TS), global_step=self.global_step))

    #gradient_norm_MMI PART
    
    if not forward_only:
      self.gradient_norms_MMI = []
      self.updates_MMI = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients_MMI = tf.gradients(self.losses_MMI[b], params_ST)
        clipped_gradients_MMI, norm_MMI = tf.clip_by_global_norm(gradients_MMI,
                                                         max_gradient_norm)
        self.gradient_norms_MMI.append(norm_MMI)
        self.updates_MMI.append(opt.apply_gradients(
            zip(clipped_gradients_MMI, params_ST), global_step=self.global_step))
        
    self.saver = tf.train.Saver(tf.global_variables())
  
  # TODO :  def step_ST, def step_TS, def step_MMI
  def step_ST(self, session, encoder_inputs_ST, decoder_inputs_ST, target_weights_ST,
           bucket_id, forward_only, beam_search):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs_ST: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs_ST: list of numpy int vectors to feed as decoder inputs.
      target_weight_STs: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs_ST, decoder_inputs_ST, or
        target_weight_STs disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs_ST) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs_ST), encoder_size))
    if len(decoder_inputs_ST) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs_ST), decoder_size))
    if len(target_weights_ST) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights_ST), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weight_STs, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs_ST[l].name] = encoder_inputs_ST[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs_ST[l].name] = decoder_inputs_ST[l]
      input_feed[self.target_weights_ST[l].name] = target_weights_ST[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs_ST[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed_ST = [self.updates_ST[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms_ST[bucket_id],  # Gradient norm.
                     self.losses_ST[bucket_id]]  # Loss for this batch.
    else:
        if beam_search:
              output_feed_ST = [self.beam_path[bucket_id]]  # Loss for this batch.
              output_feed_ST.append(self.beam_symbol[bucket_id])
        else:
            output_feed_ST = [self.losses_ST[bucket_id]]

        for l in xrange(decoder_size):  # Output logits.
            output_feed_ST.append(self.outputs_ST[bucket_id][l])
    # print bucket_id
    outputs_ST = session.run(output_feed_ST, input_feed)
    if not forward_only:
      return outputs_ST[1], outputs_ST[2], None  # Gradient norm, loss, no outputs.
    else:
      if beam_search:
          return outputs_ST[0], outputs_ST[1], outputs_ST[2:]  # No gradient norm, loss, outputs.
      else:
          return None, outputs_ST[0], outputs_ST[1:]  # No gradient norm, loss, outputs.

     
  def step_TS(self, session, encoder_inputs_TS, decoder_inputs_TS, target_weights_TS,
           bucket_id, forward_only, beam_search):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs_TS: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs_TS: list of numpy int vectors to feed as decoder inputs.
      target_weights_TS: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs_TS, decoder_inputs_TS, or
        target_weights_TS disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs_TS) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs_TS), encoder_size))
    if len(decoder_inputs_TS) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs_TS), decoder_size))
    if len(target_weights_TS) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights_TS), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weight_TS_ST_TS_STs, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs_TS[l].name] = encoder_inputs_TS[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs_TS[l].name] = decoder_inputs_TS[l]
      input_feed[self.target_weights_TS[l].name] = target_weights_TS[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs_TS[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed_TS = [self.updates_TS[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms_TS[bucket_id],  # Gradient norm.
                     self.losses_TS[bucket_id]]  # Loss for this batch.
    else:
        if beam_search:
              output_feed_TS = [self.beam_path[bucket_id]]  # Loss for this batch.
              output_feed_TS.append(self.beam_symbol[bucket_id])
        else:
            output_feed_TS = [self.losses_TS[bucket_id]]

        for l in xrange(decoder_size):  # Output logits.
            output_feed_TS.append(self.outputs_TS[bucket_id][l])
    # print bucket_id
    outputs_TS = session.run(output_feed_TS, input_feed)
    if not forward_only:
      return outputs_TS[1], outputs_TS[2], None  # Gradient norm, loss, no outputs.
    else:
      if beam_search:
          return outputs_TS[0], outputs_TS[1], outputs_TS[2:]  # No gradient norm, loss, outputs.
      else:
          return None, outputs_TS[0], outputs_TS[1:]  # No gradient norm, loss, outputs.


  def step_MMI(self, session, encoder_inputs_ST, decoder_inputs_ST, target_weights_ST, 
               encoder_inputs_TS, decoder_inputs_TS, target_weights_TS, bucket_id, forward_only, beam_search):
  # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs_ST) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs_ST), encoder_size))
    if len(decoder_inputs_ST) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs_ST), decoder_size))
    if len(target_weights_ST) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights_ST), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs_ST[l].name] = encoder_inputs_ST[l]
      input_feed[self.encoder_inputs_TS[l].name] = encoder_inputs_TS[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs_ST[l].name] = decoder_inputs_ST[l]
      input_feed[self.target_weights_ST[l].name] = target_weights_ST[l]
      input_feed[self.decoder_inputs_TS[l].name] = decoder_inputs_TS[l]
      input_feed[self.target_weights_TS[l].name] = target_weights_TS[l]


    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs_ST[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
    last_target = self.decoder_inputs_TS[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed_MMI = [self.updates_MMI[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms_MMI[bucket_id],  # Gradient norm.
                     self.losses_MMI[bucket_id]]  # Loss for this batch.
    else:
        if beam_search:
              output_feed_MMI = [self.beam_path[bucket_id]]  # Loss for this batch.
              output_feed_MMI.append(self.beam_symbol[bucket_id])
        else:
            output_feed_MMI = [self.losses_MMI[bucket_id]]

        for l in xrange(decoder_size):  # Output logits.
            output_feed_MMI.append(self.outputs_ST[bucket_id][l])
    # print bucket_id
    outputs_MMI = session.run(output_feed_MMI, input_feed)
    if not forward_only:
      return outputs_MMI[1], outputs_MMI[2], None  # Gradient norm, loss, no outputs.
    else:
      if beam_search:
          return outputs_MMI[0], outputs_MMI[1], outputs_MMI[2:]  # No gradient norm, loss, outputs.
      else:
          return None, outputs_MMI[0], outputs_MMI[1:]  # No gradient norm, loss, outputs.

  

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([GO_ID] + decoder_input +
                            [PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

  def get_batch_MMI(self, data_ST, data_TS, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs_ST, decoder_inputs_ST = [], []
    encoder_inputs_TS, decoder_inputs_TS = [], []
    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      randidx = random.randint(0,len(data_ST[bucket_id]) - 1)
      encoder_input_ST, decoder_input_ST = data_ST[bucket_id][randidx]
      encoder_input_TS, decoder_input_TS = data_TS[bucket_id][randidx]
      # Encoder inputs are padded and then reversed.
      encoder_pad_ST = [PAD_ID] * (encoder_size - len(encoder_input_ST))
      encoder_inputs_ST.append(list(reversed(encoder_input_ST + encoder_pad_ST)))
      encoder_pad_TS = [PAD_ID] * (encoder_size - len(encoder_input_TS))
      encoder_inputs_TS.append(list(reversed(encoder_input_TS + encoder_pad_TS)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size_ST = decoder_size - len(decoder_input_ST) - 1
      decoder_inputs_ST.append([GO_ID] + decoder_input_ST +
                            [PAD_ID] * decoder_pad_size_ST)
      decoder_pad_size_TS = decoder_size - len(decoder_input_TS) - 1
      decoder_inputs_TS.append([GO_ID] + decoder_input_TS +
                            [PAD_ID] * decoder_pad_size_TS)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs_ST, batch_decoder_inputs_ST, batch_weights_ST = [], [], []
    batch_encoder_inputs_TS, batch_decoder_inputs_TS, batch_weights_TS = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs_ST.append(
          np.array([encoder_inputs_ST[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))
      batch_encoder_inputs_TS.append(
          np.array([encoder_inputs_TS[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs_ST.append(
          np.array([decoder_inputs_ST[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))
      batch_decoder_inputs_TS.append(
          np.array([decoder_inputs_TS[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight_ST = np.ones(self.batch_size, dtype=np.float32)
      batch_weight_TS = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target_ST = decoder_inputs_ST[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target_ST == PAD_ID:
          batch_weight_ST[batch_idx] = 0.0
        if length_idx < decoder_size - 1:
          target_TS = decoder_inputs_TS[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target_TS == PAD_ID:
          batch_weight_TS[batch_idx] = 0.0
      batch_weights_ST.append(batch_weight_ST)
      batch_weights_TS.append(batch_weight_TS)
    return batch_encoder_inputs_ST, batch_decoder_inputs_ST, batch_weights_ST, batch_encoder_inputs_TS, batch_decoder_inputs_TS, batch_weights_TS
