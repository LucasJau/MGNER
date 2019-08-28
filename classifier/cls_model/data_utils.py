import numpy as np
import os
import random
random.seed(3)

# shared global variables to be imported from cls_model also
UNK = "[UNK]"

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
    def __init__(self, filename, max_iter=None):
        """
        Args:
            filename: path to the file
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.max_iter = max_iter
        self.length = None

        print("Read data start")
        self.roi_features = np.load(self.filename + 'roi_features.npy', allow_pickle=True)
        self.roi_elmo_features = np.load(self.filename + 'roi_elmo_features.npy', allow_pickle=True)
        self.roi_lens = np.load(self.filename + 'roi_lens.npy', allow_pickle=True)
        self.roi_labels = np.load(self.filename + 'roi_labels.npy', allow_pickle=True)
        self.roi_char_ids = np.load(self.filename + 'roi_char_ids.npy', allow_pickle=True)
        self.roi_word_lengths = np.load(self.filename + 'roi_word_lengths.npy', allow_pickle=True)
        self.sen_last_hidden = np.load(self.filename + 'sen_last_hidden.npy', allow_pickle=True)

        self.left_context_word_features = np.load(self.filename + 'left_context_word_feature.npy', allow_pickle=True)
        self.left_context_word_lens = np.load(self.filename + 'left_context_word_len.npy', allow_pickle=True)
        self.right_context_word_features = np.load(self.filename + 'right_context_word_feature.npy', allow_pickle=True)
        self.right_context_word_lens = np.load(self.filename + 'right_context_word_len.npy', allow_pickle=True)
        self.nsample = self.roi_lens.shape[0]
        print("Read data done")
        #print("********Loading self.filename", self.filename, "nsamples", self.nsample)
        self.data = self.get_data()

    def __iter__(self):
        niter = 0

        for i in range(self.nsample):
            niter += 1
            if self.max_iter is not None and niter > self.max_iter:
                break
            try:
                yield self.roi_features[i].tolist(), self.roi_elmo_features[i].tolist(), self.roi_lens[i].tolist(), self.roi_labels[i].tolist(), self.roi_char_ids[i], self.roi_word_lengths[i], self.sen_last_hidden[i].tolist(), self.left_context_word_features[i], self.left_context_word_lens[i], self.right_context_word_features[i], self.right_context_word_lens[i]
            except:
                print("except, i", i, "nsample", self.nsample)

    def get_data(self):
        print("Get data start")

        total_features = []
        total_elmo_features = []
        total_lens = []
        total_labels = []
        total_chars = []
        total_word_lens = []
        total_sen_last_hiddens = []
        total_left_context_word_features = []
        total_left_context_word_lens = []
        total_right_context_word_features = []
        total_right_context_word_lens = []

        for (features, elmo_features, lens, labels, char_ids, word_lens, last_hidden, lf, ll, rf, rl) in self:
            total_features += [features]
            total_elmo_features += [elmo_features]
            total_lens += [lens]
            total_labels += [labels]
            total_chars += [char_ids]
            total_word_lens += [word_lens]
            total_sen_last_hiddens += [last_hidden]
            total_left_context_word_features += [lf]
            total_left_context_word_lens += [ll]
            total_right_context_word_features += [rf]
            total_right_context_word_lens += [rl]
        print("Get data done")
        return [total_features, total_elmo_features, total_lens, total_labels, total_chars, total_word_lens, total_sen_last_hiddens, total_left_context_word_features, total_left_context_word_lens, total_right_context_word_features, total_right_context_word_lens]

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
        for words, tags, _, _ in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _, _, _ in dataset:
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
    with open(filename, "w") as f:
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


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


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
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        max_length = max(max_length, 1)
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(dataset, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, elmo_batch, len_batch, y_batch = [], [], [], []
    chars_batch, word_lens_batch, last_hidden_batch = [], [], []
    lc_batch, ll_batch, rc_batch, rl_batch = [],[], [], []
    for (x, e, l, y, c, wl, lh, lc, ll, rc, rl) in zip(*dataset.data):
        if len(x_batch) == minibatch_size:
            yield x_batch, elmo_batch, len_batch, y_batch, chars_batch, word_lens_batch, last_hidden_batch, lc_batch, ll_batch, rc_batch, rl_batch
            x_batch, elmo_batch, len_batch, y_batch = [], [], [], []
            chars_batch, word_lens_batch, last_hidden_batch = [], [], []
            lc_batch, ll_batch, rc_batch, rl_batch = [],[], [], []

        x_batch += [x]
        elmo_batch += [e]
        len_batch += [l]
        y_batch += [y]
        chars_batch += [c]
        word_lens_batch += [wl]
        last_hidden_batch += [lh]
        lc_batch += [lc]
        ll_batch += [ll]
        rc_batch += [rc]
        rl_batch += [rl]

    if len(x_batch) != 0:
        yield x_batch, elmo_batch, len_batch, y_batch, chars_batch, word_lens_batch, last_hidden_batch, lc_batch, ll_batch, rc_batch, rl_batch
