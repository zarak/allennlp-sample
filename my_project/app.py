import streamlit as st
from predict import SentenceClassifierPredictor
from model import SimpleClassifier
from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp.models import Model
from dataset_reader import ClassificationTsvReader

from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder



def build_model(vocab):
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)


if __name__ == "__main__":
    # vocab = Vocabulary.from_files("results/vocabulary")
    # model = build_model(vocab)
    # Model.from_archive("results/model.tar.gz")
    model = Model.from_archive("/home/ubuntu/allennlp-sample/results/model.tar.gz")
    # model = Model.from_archive("../results/model.tar.gz")

    token_indexer = ELMoTokenCharactersIndexer()
    dataset_reader = ClassificationTsvReader(token_indexers={'elmo': token_indexer})
    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
     
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})


    # predictor = SentenceClassifierPredictor(model, dataset_reader)
    predictor = SentenceClassifierPredictor(model, dataset_reader)

    # output = predictor.predict('A good movie!')
    # print([(model.vocab.get_token_from_index(label_id, 'labels'), prob)
           # for label_id, prob in enumerate(output['probs'])])
    # output = predictor.predict('This was a monstrous waste of time.')
    # print([(model.vocab.get_token_from_index(label_id, 'labels'), prob)
           # for label_id, prob in enumerate(output['probs'])])

    st.title("Media Bias Demo")
    inp = st.text_area('Enter the text from a news article below')
    output = predictor.predict(inp)
    print("OUTPUT", output)
    st.write(f"The probability that this article is 'hyperpartisan' is {output['probs'][1]:.2f}.")
