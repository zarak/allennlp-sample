import streamlit as st
from predict import SentenceClassifierPredictor
from model import SimpleClassifier
from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from allennlp.models import Model
from dataset_reader import ClassificationTsvReader



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
    dataset_reader = ClassificationTsvReader()

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
