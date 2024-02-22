from sherlock.features.word_embeddings import initialise_word_embeddings
from sherlock.features.paragraph_vectors import (
    initialise_pretrained_model,
    initialise_nltk,
)
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)

print("init sherlock embeddings")
prepare_feature_extraction()
print("prepare_feature_extraction done")
initialise_word_embeddings()
print("initialise_word_embeddings done")
initialise_pretrained_model(400)
print("initialise_pretrained_model done")
initialise_nltk()
print("initialise_nltk done")