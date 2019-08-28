import numpy as np
import random
import tensorflow as tf
from elmo.model import BidirectionalLanguageModel
from elmo.elmo import weight_layers
from elmo.data import TokenBatcher

random.seed(3)

# shared global variables to be imported from cls_model also
UNK = "[UNK]"


# NUM = "$NUM$"
# NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """

    def __init__(self, filename, elmo_model, elmo_dim, vocab_file,
                 generate_anchor=None, max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.generate_anchor = generate_anchor
        self.max_iter = max_iter
        self.length = None
        self.vocab = load_vocab(vocab_file)
        # self.elmo_filename = elmo_filename
        # self.elmo_option = elmo_option
        # self.elmo_token_embedding = elmo_token_embedding
        self.elmo_dim = elmo_dim
        self.elmo_model = elmo_model
        # self.elmo_info = get_elmo_model(vocab_file, elmo_option,elmo_filename, elmo_token_embedding)
        self.data = self.get_data()

    def __iter__(self):
        niter = 0
        anchors, anchor_labels, class_ids, sample_indexes = [], [], [], []
        with open(self.filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                arr = line.split('\t')

                sentence = arr[0]

                if len(sentence) != 0:
                    niter += 1
                    if self.max_iter is not None and niter > self.max_iter:
                        break

                    # generate anchors
                    if self.generate_anchor is not None:
                        anchors, anchor_labels, class_ids, sample_indexes = self.generate_anchor(line)

                    # get sentence for elmo

                    raw_sentence = list(sentence)
                    # sentence_embedding = sentence_embedding[0].reshape(-1, self.elmo_dim)
                    sentence = char_to_id(self.vocab, sentence)
                    yield sentence, raw_sentence, anchors, anchor_labels, class_ids

    def get_data(self):

        total_words = []
        total_raw_sentence = []
        total_anchors = []
        total_anchor_labels = []
        total_class_ids = []

        for (words, raw_sentence, anchors, anchor_labels, class_ids) in self:
            total_words += [words]
            total_raw_sentence += [raw_sentence]
            total_anchors += [anchors]
            total_anchor_labels += [anchor_labels]
            total_class_ids += [class_ids]

        total_elmo_feature = get_elmo_feature(total_raw_sentence, self.elmo_model)

        return [total_words, total_elmo_feature, total_anchors, total_anchor_labels, total_class_ids]

    def shuffle_data(self):
        data_len = len(self.data[0])
        data_idx = list(range(0, data_len))
        random.shuffle(data_idx)
        shuffled_data = []
        for i in range(len(self.data)):
            shuffled_data.append([self.data[i][x] for x in data_idx])
        self.data = shuffled_data

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, _, _, _, _ in dataset:
            vocab_words.update(words)
            # vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, _


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _, _, _, _ in dataset:
        for word in words:
            vocab_char.update(word)
    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def char_to_id(vocab, sentence):
    vocab_lst = []
    for char in sentence:
        if char in vocab.keys():
            vocab_lst.append(vocab[char])
        else:
            vocab_lst.append(vocab[UNK])

    return vocab_lst


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.random.random([len(vocab), dim])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_elmo_feature(context, elmo_model):
    batcher, context_token_ids, elmo_context_output = elmo_model

    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        # Create batches of data.
        context_ids = batcher.batch_sentences(context)
        nbatches = (len(context_ids) + 128 - 1) // 128
        total_elmo = []

        for batch in range(nbatches):
            # Compute ELMo representations (here for the input only, for simplicity).
            elmo_layer = sess.run([elmo_context_output['weighted_op']],
                                  feed_dict={context_token_ids: context_ids[batch * 128:(batch + 1) * 128]})

            elmo_layer = list(elmo_layer[0])
            total_elmo.extend(elmo_layer)

        elmo_feature = []
        for i, _ in enumerate(total_elmo):
            tmp = total_elmo[i][:np.where(context_ids[i] == 1)[0][0] - 1]
            elmo_feature.append(tmp)

    return elmo_feature


def get_elmo_model(vocab_file, options_file, weight_file, token_embedding_file):
    batcher = TokenBatcher(vocab_file)

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(
        options_file,
        weight_file,
        use_character_inputs=False,
        embedding_weight_file=token_embedding_file
    )

    # Input placeholders to the biLM.
    context_token_ids = tf.placeholder('int32', shape=(None, None))

    # Get ops to compute the LM embeddings.
    context_embeddings_op = bilm(context_token_ids)

    elmo_context_output = weight_layers(
        'output', context_embeddings_op, l2_coef=0.0)

    return batcher, context_token_ids, elmo_context_output


def get_char_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_char(vocab_chars=None, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """

    def f(sentence):
        # 0. get chars of words
        char_ids = []
        for char in sentence:
            # ignore chars out of vocabulary
            if char in vocab_chars:
                char_ids += [vocab_chars[char]]
            else:
                char_ids += vocab_chars[UNK]

            return char_ids

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    return sequence_padded, sequence_length


def pad_elmo_embedding(embedding, elmo_dim):
    """
    input:
        embedding: list of numpy array,
                   list len : batch_size
                   array shape: [3: layer_num, sentence_len, 1024]
    output:
        padded embedding(numpy array)
        shape: [batch_size, 3, max_sen_len, 1024]
    """
    max_len = max(map(lambda x: x.shape[0], embedding))

    padded_embedding = []
    sequence_length = []
    for emb in embedding:
        padded_len = max_len - emb.shape[0]
        empty_emb = np.zeros([padded_len, elmo_dim])
        padded_emb = np.append(emb, empty_emb, axis=0)
        padded_embedding.append(padded_emb)
        sequence_length.append(emb.shape[0])
    return np.array(padded_embedding), sequence_length


def minibatches(dataset, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch, sen_batch = [], [], []
    anchor_batch, anchor_label_batch, cls_batch = [], [], []
    for x, sen, anchor, anchor_label, class_id in zip(*dataset.data):
        if len(x_batch) == minibatch_size:
            yield x_batch, sen_batch, anchor_batch, anchor_label_batch, cls_batch
            x_batch, sen_batch = [], []
            anchor_batch, anchor_label_batch, cls_batch = [], [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        sen_batch += [sen]
        anchor_batch += [anchor]
        anchor_label_batch += [anchor_label]
        cls_batch += [class_id]

    if len(x_batch) != 0:
        yield x_batch, sen_batch, anchor_batch, anchor_label_batch, cls_batch
