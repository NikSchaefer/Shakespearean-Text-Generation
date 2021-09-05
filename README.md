# Shakespearean Text Generation with Recurrent Neural Networks

This project uses a Recurrent Neural Network to predict the next letter in text trained on a shakespearean dataset. Can be run in sucession to form multiple sentences.

used to make things like this:

```
Romeo:
Keep thou the read of his thoughts, King of Lancaster?
You would play aportment and a foregun to death.
That she I flame depass the souls to heal these peers

...
```

from an input of "Romeo:"

## How it works

This project uses Tensorflow as well as the high level API keras.

### preprocessing

the project reads its dataset from this [text file](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt). The project works by taking a Set of the text and using the Keras StringLookup function. This works by converting each different word and assigning a unique ID to it. We then use another function to convert the ID back.

We then batch the dataset into seperate texts for the model to predict. Each batch splits the input text and then the target text for the model

### model

The primary model is comprised of 3 layers. The first layer is an embedding layer that maps each character-ID to a Vector. This is what the RNN layer takes as Input. The second layer is the recurrent neural network layer. Finally the RNN connects to a Dense layer with one node for each letter. 

### training
the loss for this model is SpareCategoricalCrossentropy and uses an adam optimizer. I recommend around 20-30 Epochs depending on the GPU or CPU running the process

### One Step Model
finally there is another model that is used for the generation of the text and run in sucession. it works by taking the input and putting it in the model several times and then appending the result of each output to generate a finished text. 

## Dependencies

Install dependencies with Pip

`pip install tensorflow numpy`

Dependencies:

- Tensorflow
- Numpy

## License

[MIT](https://choosealicense.com/licenses/mit/)
