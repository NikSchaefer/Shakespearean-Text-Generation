import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

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


model = tf.keras.models.load_model("save")

model.summary()
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
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

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

states = None

next_char = tf.constant(["ROMEO:"])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)

print(result[0].numpy().decode("utf-8"), "\n\n" + "_" * 80)
