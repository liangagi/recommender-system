## source: this file is from https://gist.github.com/erap129/7049d6400e4baaa2c98c0a33bb397e6b#file-tfrs_movielens-py
## details in this blog: https://towardsdatascience.com/movielens-1m-deep-dive-part-ii-tensorflow-recommenders-4ca358cc886e
import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import re
from cachier import cachier
from imdb import Cinemagoer
pio.renderers.default = "browser"


import plotly.graph_objects as go

# Define the features used for movies and users
MOVIE_FEATURES = ['movie_title', 'movie_genres', 'movie_title_text', 'movie_length']
USER_FEATURES = ['user_id', 'timestamp', 'bucketized_user_age']


class UserModel(tf.keras.Model):
    """
    Model for representing user features.

    Args:
        unique_user_ids (numpy.ndarray): Array of unique user IDs.
        embedding_size (int): Size of the embedding vectors.
        additional_features (tuple): Tuple of additional features to be used in the model.
        additional_feature_info (dict): Dictionary containing additional feature information.

    Attributes:
        additional_embeddings (dict): Dictionary to store additional embeddings.
        user_embedding (tf.keras.Sequential): Sequential model for user embedding.
        user_age_normalizer (tf.keras.layers.Normalization): Normalization layer for user age.

    Methods:
        call: Forward pass of the model.

    """

    def __init__(self, unique_user_ids, embedding_size=32, additional_features=(), additional_feature_info=None):
        super().__init__()
        self.additional_embeddings = {}

        # User embedding
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_size)
        ])

        # Additional embeddings
        if 'timestamp' in additional_features:
            self.additional_embeddings['timestamp'] = tf.keras.Sequential([
                tf.keras.layers.Discretization(additional_feature_info['timestamp_buckets'].tolist()),
                tf.keras.layers.Embedding(len(additional_feature_info['timestamp_buckets']) + 1, embedding_size),
            ])

        if 'bucketized_user_age' in additional_features:
            self.user_age_normalizer = tf.keras.layers.Normalization(axis=None)
            self.user_age_normalizer.adapt(additional_feature_info['bucketized_user_age'])
            self.additional_embeddings['bucketized_user_age'] = tf.keras.Sequential([self.user_age_normalizer,
                                                                                     tf.keras.layers.Reshape([1])])

    def call(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (dict): Dictionary of input features.

        Returns:
            tf.Tensor: Concatenated embeddings.

        """
        return tf.concat([self.user_embedding(inputs['user_id'])] +
                         [self.additional_embeddings[k](inputs[k]) for k in self.additional_embeddings],
                         axis=1)


class MovieModel(tf.keras.Model):
    """
    Model for representing movie features.

    Args:
        unique_movie_titles (numpy.ndarray): Array of unique movie titles.
        additional_features (tuple): Tuple of additional features to be used in the model.
        additional_feature_info (dict): Dictionary containing additional feature information.
        embedding_size (int): Size of the embedding vectors.

    Attributes:
        additional_embeddings (dict): Dictionary to store additional embeddings.
        title_embedding (tf.keras.Sequential): Sequential model for movie title embedding.
        title_vectorizer (tf.keras.layers.TextVectorization): Text vectorization layer for movie titles.
        movie_length_normalizer (tf.keras.layers.Normalization): Normalization layer for movie length.

    Methods:
        call: Forward pass of the model.

    """

    def __init__(self, unique_movie_titles, additional_features, additional_feature_info, embedding_size=32):
        super().__init__()
        self.additional_embeddings = {}

        # Movie title embedding
        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_size)
        ])

        # Additional embeddings
        if 'movie_genres' in additional_features:
            self.additional_embeddings['movie_genres'] = tf.keras.Sequential([
                tf.keras.layers.Embedding(max(additional_feature_info['unique_movie_genres']) + 1, embedding_size),
                tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))
            ])

        if 'movie_title_text' in additional_features:
            max_tokens = 10_000
            self.title_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
            self.title_vectorizer.adapt(unique_movie_titles)
            self.additional_embeddings['movie_title_text'] = tf.keras.Sequential([
                self.title_vectorizer,
                tf.keras.layers.Embedding(max_tokens, embedding_size, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D(),
            ])

        if 'movie_length' in additional_features:
            self.movie_length_normalizer = tf.keras.layers.Normalization(axis=None)
            self.movie_length_normalizer.adapt(additional_feature_info['movie_length'])
            self.additional_embeddings['movie_length'] = tf.keras.Sequential([self.movie_length_normalizer,
                                                                              tf.keras.layers.Reshape([1])])

    def call(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (dict): Dictionary of input features.

        Returns:
            tf.Tensor: Concatenated embeddings.

        """
        return tf.concat([self.title_embedding(inputs['movie_title'])] +
                         [self.additional_embeddings[k](inputs[k]) for k in self.additional_embeddings],
                         axis=1)


class QueryCandidateModel(tf.keras.Model):
    """
    Model for representing query and candidate features.

    Args:
        layer_sizes (list): List of integers representing the sizes of the hidden layers.
        embedding_model (tf.keras.Model): Model for embedding features.

    Attributes:
        embedding_model (tf.keras.Model): Model for embedding features.
        dense_layers (tf.keras.Sequential): Sequential model for dense layers.

    Methods:
        call: Forward pass of the model.

    """

    def __init__(self, layer_sizes, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.dense_layers = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(layer_sizes[-1]))

    def call(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (dict): Dictionary of input features.

        Returns:
            tf.Tensor: Output of the dense layers.

        """
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class MovieLensModel(tfrs.models.Model):
    """
    Model for the MovieLens recommendation system.

    Args:
        layer_sizes (list): List of integers representing the sizes of the hidden layers.
        movies (tf.data.Dataset): Dataset containing movie data.
        unique_movie_titles (numpy.ndarray): Array of unique movie titles.
        n_unique_user_ids (numpy.ndarray): Array of unique user IDs.
        embedding_size (int): Size of the embedding vectors.
        additional_features (tuple): Tuple of additional features to be used in the model.
        additional_feature_info (dict): Dictionary containing additional feature information.

    Attributes:
        additional_features (tuple): Tuple of additional features to be used in the model.
        query_model (QueryCandidateModel): Model for query features.
        candidate_model (QueryCandidateModel): Model for candidate features.
        task (tfrs.tasks.Retrieval): Retrieval task.

    Methods:
        compute_loss: Computes the loss for the model.

    """

    def __init__(self, layer_sizes, movies, unique_movie_titles, n_unique_user_ids, embedding_size,
                 additional_features, additional_feature_info):
        super().__init__()
        self.additional_features = additional_features
        self.query_model = QueryCandidateModel(layer_sizes, UserModel(n_unique_user_ids,
                                                                      embedding_size=embedding_size,
                                                                      additional_features=self.additional_features,
                                                                      additional_feature_info=additional_feature_info))
        self.candidate_model = QueryCandidateModel(layer_sizes, MovieModel(unique_movie_titles,
                                                                           embedding_size=embedding_size,
                                                                           additional_features=self.additional_features,
                                                                           additional_feature_info=additional_feature_info))
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=(movies
                            .apply(tf.data.experimental.dense_to_ragged_batch(128))
                            .map(self.candidate_model)),
            ),
        )

    def compute_loss(self, features, training=False):
        """
        Computes the loss for the model.

        Args:
            features (dict): Dictionary of input features.
            training (bool): Whether the model is in training mode.

        Returns:
            tf.Tensor: Loss value.

        """
        query_embeddings = self.query_model({
            'user_id': features['user_id'],
            **{k: features[k] for k in self.additional_features if k in USER_FEATURES}
        })
        movie_embeddings = self.candidate_model({
            'movie_title': features['movie_title'],
            **{k: features[k] for k in self.additional_features if k in MOVIE_FEATURES}
        })
        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)


def get_movie_length_wrapper(movie_title):
    """
    Wrapper function to get the length of a movie.

    Args:
        movie_title (tf.Tensor): Tensor containing the movie title.

    Returns:
        int: Length of the movie.

    """
    return get_movie_length(movie_title.numpy().decode())


@cachier(separate_files=True)
def get_movie_length(movie_title_str):
    """
    Function to get the length of a movie.

    Args:
        movie_title_str (str): String containing the movie title.

    Returns:
        int: Length of the movie.

    """
    movielens_year = int(re.search('\((\d+)\)', movie_title_str).groups(0)[0])
    try:
        movie = [x for x in ia.search_movie(movie_title_str) if 'year' in x and x['year'] == movielens_year][0]
    except IndexError:
        try:
            movie = ia.search_movie(re.search('(.*?) \(', movie_title_str).groups(0)[0])[0]
        except IndexError:
            return 90
    ia.update(movie, ['technical'])
    try:
        runtime_str = movie.get('tech')['runtime']
    except KeyError:
        return 90
    try:
        return int(re.search('\((\d+)', runtime_str[0]).groups(0)[0])
    except AttributeError:
        try:
            return int(re.search('(\d+)', runtime_str[0]).groups(0)[0])
        except AttributeError:
            return 90


def add_movie_length_to_dataset(tf_dataset):
    """
    Function to add movie length to a dataset.

    Args:
        tf_dataset (tf.data.Dataset): Dataset to add movie length to.

    Returns:
        tf.data.Dataset: Dataset with movie length added.

    """
    return (tf_dataset.map(lambda x: {**x, 'movie_length': tf.py_function(func=get_movie_length_wrapper,
                                                                         inp=[x['movie_title']],
                                                                         Tout=[tf.int32])})
                      .map(lambda x: {**x, 'movie_length': x['movie_length'][0] if x['movie_length'] > 0 else 90}))


class MovieLensTrainer:
    """
    A class that represents a trainer for the MovieLens model.

    Args:
        num_epochs (int): The number of training epochs.
        embedding_size (int): The size of the embedding vectors.
        layer_sizes (list): A list of integers representing the sizes of the hidden layers.
        additional_feature_sets (list): A list of additional feature sets to be used in the model.
        retrain (bool): Whether to retrain the model if a saved model exists.

    Attributes:
        num_epochs (int): The number of training epochs.
        embedding_size (int): The size of the embedding vectors.
        layer_sizes (tuple): A tuple of integers representing the sizes of the hidden layers.
        additional_feature_sets (list): A list of additional feature sets to be used in the model.
        retrain (bool): Whether to retrain the model if a saved model exists.
        movies (tf.data.Dataset): The dataset containing movie data.
        ratings (tf.data.Dataset): The dataset containing rating data.
        unique_movie_titles (numpy.ndarray): An array of unique movie titles.
        unique_movie_genres (tf.Tensor): A tensor of unique movie genres.
        unique_user_ids (numpy.ndarray): An array of unique user IDs.
        max_timestamp (float): The maximum timestamp in the rating data.
        min_timestamp (float): The minimum timestamp in the rating data.
        additional_feature_info (dict): A dictionary containing additional feature information.

    Methods:
        get_datasets: Loads or creates the movie and rating datasets.
        train_all_models: Trains models for all combinations of additional feature sets.
        get_movielens_model: Retrieves a trained model for a specific set of additional features.
        train_movielens_model: Trains a model for a specific set of additional features.

    """

    def __init__(self, num_epochs, embedding_size, layer_sizes, additional_feature_sets, retrain):
        self.num_epochs = num_epochs
        self.embedding_size = embedding_size
        self.layer_sizes = tuple(layer_sizes)
        self.additional_feature_sets = additional_feature_sets
        self.retrain = retrain
        self.get_datasets()
        self.all_ratings = list(self.ratings.map(lambda x: {'movie_title': x["movie_title"],
                                                            'user_id': x['user_id'],
                                                            'bucketized_user_age': x['bucketized_user_age'],
                                                            'movie_genres': x['movie_genres'],
                                                            'timestamp': x['timestamp'],
                                                            'movie_length': x['movie_length']})
                                .apply(tf.data.experimental.dense_to_ragged_batch(len(self.ratings))))[0]
        all_movies = list(self.movies.apply(tf.data.experimental.dense_to_ragged_batch(len(self.movies))))[0]
        self.unique_movie_titles = np.unique(all_movies['movie_title'])
        self.unique_movie_genres, _ = tf.unique(all_movies['movie_genres'].flat_values)
        self.unique_user_ids = np.unique(self.all_ratings['user_id'])
        self.max_timestamp = self.all_ratings['timestamp'].numpy().max()
        self.min_timestamp = self.all_ratings['timestamp'].numpy().min()
        self.additional_feature_info = {'timestamp_buckets': np.linspace(self.min_timestamp, self.max_timestamp,
                                                                         num=1000),
                                        'unique_movie_genres': self.unique_movie_genres,
                                        'bucketized_user_age': self.all_ratings['bucketized_user_age'],
                                        'movie_length': self.all_ratings['movie_length']}

    def get_datasets(self):
        """
        Loads or creates the movie and rating datasets.

        If the datasets already exist, they are loaded from disk. Otherwise, they are created by
        downloading the MovieLens dataset and adding additional features.

        Returns:
            None

        """
        movies_data_path = './datasets/movies_data'
        ratings_data_path = './datasets/ratings_data'
        if os.path.exists(movies_data_path) and os.path.exists(ratings_data_path):
            self.movies = tf.data.Dataset.load(movies_data_path)
            self.ratings = tf.data.Dataset.load(ratings_data_path)
        else:
            self.movies = add_movie_length_to_dataset(tfds.load("movielens/1m-movies", split="train")
                                                      .map(lambda x: {**x, 'movie_title_text': x['movie_title']}))
            self.ratings = (add_movie_length_to_dataset(tfds.load("movielens/1m-ratings", split="train")
                                                        .map(lambda x: {**x, 'movie_title_text': x['movie_title']}))
                            .shuffle(100_000, seed=42, reshuffle_each_iteration=False))
            tf.data.Dataset.save(self.movies, movies_data_path)
            tf.data.Dataset.save(self.ratings, ratings_data_path)

    def train_all_models(self):
        """
        Trains models for all combinations of additional feature sets.

        Returns:
            dict: A dictionary mapping additional feature sets to trained models and their training history.

        """
        models = {}
        for additional_features in self.additional_feature_sets:
            model, history = self.get_movielens_model(tuple(additional_features))
            models[tuple(additional_features)] = (model, history)
        return models

    def get_movielens_model(self, additional_features):
        """
        Retrieves a trained model for a specific set of additional features.

        If a saved model exists for the given set of additional features, it is loaded from disk.
        Otherwise, a new model is trained and saved.

        Args:
            additional_features (tuple): A tuple representing the additional features to be used in the model.

        Returns:
            tuple: A tuple containing the trained model and its training history.

        """
        folder_name = f'saved_models/{self.num_epochs}_{self.embedding_size}_{self.layer_sizes}' \
                      f'_{tuple(sorted(additional_features))}'
        if os.path.exists(folder_name) and not self.retrain:
            model = tf.saved_model.load(f'{folder_name}/model')
            with open(f'{folder_name}/model_history.pkl', 'rb') as f:
                model_history = pickle.load(f)
            return model, model_history
        else:
            return self.train_movielens_model(additional_features, folder_name)

    def train_movielens_model(self, additional_features, folder_name):
        """
        Trains a model for a specific set of additional features.

        Args:
            additional_features (tuple): A tuple representing the additional features to be used in the model.
            folder_name (str): The name of the folder to save the trained model and its history.

        Returns:
            tuple: A tuple containing the trained model and its training history.

        """
        trainset = (self.ratings
                    .take(80_000)
                    .shuffle(100_000)
                    .apply(tf.data.experimental.dense_to_ragged_batch(2048))
                    .cache())
        testset = (self.ratings
                   .skip(80_000)
                   .take(20_000)
                   .apply(tf.data.experimental.dense_to_ragged_batch(2048))
                   .cache())
        model = MovieLensModel(self.layer_sizes, self.movies, self.unique_movie_titles, self.unique_user_ids,
                               self.embedding_size,
                               additional_features=additional_features,
                               additional_feature_info=self.additional_feature_info)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1), run_eagerly=True)
        model_history = model.fit(
            trainset,
            validation_data=testset,
            validation_freq=5,
            epochs=self.num_epochs,
            verbose=1)
        model.task = tfrs.tasks.Retrieval()
        model.compile()
        tf.saved_model.save(model, f'{folder_name}/model')
        with open(f'{folder_name}/model_history.pkl', 'wb') as f:
            pickle.dump(model_history.history, f)
        return tf.saved_model.load(f'{folder_name}/model'), model_history.history


def plot_training_runs(model_histories, datapane_token=None):
    """
    Plots the training runs for different models.

    Args:
        model_histories (dict): Dictionary mapping additional feature sets to trained models and their training history.
        datapane_token (str): Token for uploading the plot to Datapane.

    Returns:
        go.Figure: Plotly figure object.

    """
    first_key = list(model_histories.keys())[0]
