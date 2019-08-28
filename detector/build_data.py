from detector.dtc_model.config import Config
from detector.dtc_model.data_utils import CoNLLDataset, \
    load_vocab, export_trimmed_glove_vectors


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)
    # processing_word = get_processing_char(lowercase=True)
    #
    # dev   = CoNLLDataset(config.filename_dev, config.elmofile_dev,
    #                      config.elmo_option_file, config.elmo_dim, processing_word)
    # test   = CoNLLDataset(config.filename_test, config.elmofile_test,
    #                       config.elmo_option_file, config.elmo_dim, processing_word)
    # train   = CoNLLDataset(config.filename_train, config.elmofile_train,
    #                        config.elmo_option_file, config.elmo_dim, processing_word)



    # Build and save char vocab
    vocab_chars = load_vocab(config.filename_chars)
    export_trimmed_glove_vectors(vocab_chars, config.filename_char_vec,
                                 config.filename_trimmed_char_vec, config.dim_word)

if __name__ == "__main__":
    main()
