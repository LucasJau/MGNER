import os

from .rpn import generate_anchor, load_label_dict, load_cls_dict
from .general_utils import get_logger
from .data_utils import get_char_vectors, load_vocab


class Config:
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_chars = load_vocab(self.filename_chars)
        self.nchars = len(self.vocab_chars)

        # 3. get pre-trained embeddings
        self.embeddings = get_char_vectors(self.filename_trimmed_char_vec)

    # general config
    dir_output = "cls_results/test/"
    dir_model = dir_output + "cls_model.weights/"
    dir_ner_model = "../sequence_tagging/" + dir_output + "shared_var.ckpt/"
    data_pre = "../data/"
    path_log = dir_output + "log.txt"

    # save features
    dir_saved_roi = "../data/saved_roi/"

    # embeddings
    dim_word = 100
    dim_char = 100

    # anchor types for each word center
    anchor_types = 6
    rpn_topN = 20
    roi_types = 8

    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed_char_vec = data_pre + "trimmed_char_vec.npz"
    use_pretrained = True

    # dataset
    filename_dev = dir_saved_roi + "dev_word_ids/"
    filename_test = dir_saved_roi + "test_word_ids/"
    filename_train = dir_saved_roi + "train_word_ids/"

    max_iter = None  # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_chars = "../data/chars.txt"

    # training
    train_embeddings = True
    nepochs = 30  # 15
    dropout = 0.5
    batch_size = 32
    lr_method = "adam"
    lr = 0.005
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 5
    batch_sample = 128  # samples that use to calculate rpn loss
    decay_logic = True

    # cls_model hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm_1 = 500  # lstm on word embeddings
    hidden_size_lstm_2 = 500  # lstm on word embeddings
    input_feature_dim = 128  # 600 + 1024
    cls_hidden_size = 100

    elmo_dim = 200

    # evaluataion
    train_total_entity = 33992
    dev_total_entity = 3819
    test_total_entity = 3027

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = False  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU

    run_name = "size_" + str(hidden_size_lstm_1)

