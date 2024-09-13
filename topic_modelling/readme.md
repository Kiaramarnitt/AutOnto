# BERTopicModel and CustomRepresentationModelV2

## BERTopicModel

The `BERTopicModel` class is designed to facilitate topic modeling using the BERTopic algorithm. It allows for easy training of the model, evaluation, and extraction of topics from a given dataset.

### Features:

- **Initialization**: Initialize the BERTopicModel with customizable hyperparameters such as `min_cluster_size`, `min_sample_size`, `embeddings`, `nr_topics`, `reduce_outliers`, and `messages`.

- **Training**: Train the BERTopic model on a dataset using the `train_model` method, which returns topics identified by the model, frequency of words in topics, topic dictionary, BERTopic topics, and the trained BERTopic model.

- **Evaluation**: Evaluate the trained model using the `evaluate_model` method, which computes NPMI (Normalized Pointwise Mutual Information) and Topic Diversity scores.

## CustomRepresentationModelV2

The `CustomRepresentationModelV2` class provides a custom representation model for generating topic descriptions based on a given set of representative documents. It interacts with the ChatGPT API to generate concise summaries or descriptions for each topic.

###### Insert your ChatGpt API key in this script

### Models.ipynb

Shows evaluation and training of the 3 models mentioned within the paper


