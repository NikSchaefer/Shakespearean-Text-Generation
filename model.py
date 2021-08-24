import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time


path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

text = open(path_to_file, "rb").read().decode(encoding="utf-8")

vocab = sorted(set(text))
vocab_size = len(vocab)


ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)


chars_from_ids = preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
)


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)


seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 10

embedding_dim = 256
rnn_units = 1024

dataset = (
    dataset.shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)

        # Embedding Layer map character id to a vector with embedding dimensions
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # type of RNN with units
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True
        )

        # one logit for each character in vocabulary
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(
        input_example_batch
    )  # (batch_size, sequence_length, vocab_size)


# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

# print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
# print("")
# print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())


example_batch_loss = loss(target_example_batch, example_batch_predictions)

mean_loss = example_batch_loss.numpy().mean()
print("Mean loss:        ", mean_loss)

print(tf.exp(mean_loss).numpy())


model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])


history = model.fit(dataset, epochs=EPOCHS)


class OneStepModel(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temp=1.0):
        super().__init__()
        self.temperature = temp
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(["[UNK]"])[:, None]

        sparse_mask = tf.SparseTensor(
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.spare_to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True
        )
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

one_step_model = OneStepModel(model, chars_from_ids, ids_from_chars)