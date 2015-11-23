# coding: utf-8

import numpy as np
from igraph import Graph, summary, UniqueIdGenerator

from keras.preprocessing.sequence import skipgrams, make_sampling_table, pad_sequences
from keras.layers.embeddings import WordContextProduct
from keras.models import Sequential
from keras.utils import np_utils, generic_utils
from six.moves import cPickle


import logging
logging.basicConfig(format='%(asctime)s\t%(levelname)s:%(message)s', level=logging.INFO)

def load_adjlist(filename, directed=True):
    edgelist = []
    names = UniqueIdGenerator()
    for line in open(filename):
        parts = line.strip().split()
        u = names[parts.pop(0)]
        edgelist.extend([(u, names[v]) for v in parts])
    logging.debug("Edgelist for line %s : %s" % (parts, edgelist))
    g = Graph(edgelist, directed=directed)
    g.vs["name"] = names.values()
    return g


def train_batch(model, couples, labels):
    if len(couples) == 0:
        logging.warn("Input length is zero")
        return 0
    if len(couples) != len(labels):
        logging.warn("Length of input = %s and output =%s don't match" % (len(couples), len(labels)))
        return 0
    X = np.array(couples, dtype='int32')
    loss = model.train_on_batch(X,labels)
    return loss


def train_on_model(model, g, vocab_size, max_len = 10, epochs = 100, print_every=10, window_size=4, negative_sampling=1.0, sampling_table=None):
  losses, valid_sequences = 0.0, 0
  for epoch in xrange(epochs):
    sequences = pad_sequences([g.random_walk(k,max_len) for k in range(vocab_size)])
    X_couples = []
    y_labels = []
    for seq in sequences:
      couples, labels = skipgrams(seq, vocab_size, window_size=window_size, negative_samples=negative_sampling, sampling_table=sampling_table)
      X_couples.extend(couples)
      y_labels.extend(labels)
      if len(couples) == 0:
        continue
      valid_sequences += 1
    loss = train_batch(model, X_couples, y_labels)
    losses += loss
    if epoch % print_every == 0:
      logging.info("Mean loss in Epoch [%s] with %s valid sequences = %s" % (epoch, valid_sequences, losses / valid_sequences))
      losses, valid_sequences = 0.0, 0


if __name__ == "__main__":
  #g = Graph.Read_Edgelist("deepwalk/p2p-Gnutella08.edgelist")
  g = load_adjlist("deepwalk/karate.adjlist", directed=False)
  vocab_size = len(g.vs)
  max_len = 5
  save = True
  sampling_table = make_sampling_table(vocab_size)
  degrees = np.array(g.vs.degree())
  inv_sqrt_degree = 1/np.sqrt(degrees)
  sampling_table = inv_sqrt_degree/np.sum(inv_sqrt_degree)
  logging.info("Graph Summary: \n", summary(g))
  logging.info("Building Model")
  if save:
    model = cPickle.load(open("out/Karate.Model.3100.pkl"))
  else:
    model = cPickle.load("out/Karate.Model.3100.pkl")
    model = Sequential()
    model.add(WordContextProduct(vocab_size, proj_dim=300, init='uniform'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    #couples, labels = skipgrams(sequences[np.random.randint(vocab_size)], vocab_size, window_size=4, negative_samples=1.0, sampling_table=sampling_table)
    #train_on_model(model, g, vocab_size, print_every=1)
    #cPickle.dump(model, open("out/Karate.Model.3100.pkl", "wb"))
