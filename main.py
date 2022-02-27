from this import d
import keras.losses
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from random import randint
import visualkeras
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from cv2 import resize
from os.path import exists
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train, x_test = x_train / 255.0, x_test / 255.0


st.title("rAI Generator")

"""
Hi all, I am Rai, and this is my project, rAI. rAI is an educational interactive data science webapp built to help familiarize people
with simple neural networks by constructing a model that functions great on paper, but is *exceedingly bad* in real life (like me, hence rAI).

rAIs are bad, like really bad, because they are the simplest a neural network can be. No convolutional laters, no LSTMs, just simple Dense layers feeding into each others.
Though this webapp is built with those who has zero experient with ML in mind, so this oversimplicity is at least intended.
"""

st.header("Oversimplification of a Neural Network")

st.write("When the topic of neural network is mentioned, some might visualize something like this in their head:")
ex = Sequential()
ex.add(layers.Dense(5, input_shape=(5,)))
ex.add(layers.Dense(8))
ex.add(layers.Dense(4))

ex._layers = ex._self_tracked_trackables

visualkeras.graph_view(ex, to_file="ex.png")
st.image("ex.png", caption="Simple neural network architecture")

"""
A web of *things* connecting with other *things* by *things*, and they would be right! Except that these things are more than just things, they're *numbers*.
Every node in the network, called a "neuron", holds a number (in this case) between 0 and 1, called its "activation". In the first layer of neurons,
you input in the values for their activation, and whatever activation in the final layer is, that is your output. How you *get* from the first layer to the
final layer is the goal of every neural network.

So how exactly do you get from one layer to another? That's right, by doing *math*. In a Dense layer, every neurons are connected with all neurons in its preceeding layers,
hence *Dense*. A neuron connected this way will have its activated calculated by the sum of all the activations of all neurons connected to it
multiplied by the strength of their connections, or "weight". For example, the activation of a neuron on layer two would be calculated by:
"""

st.latex("\sum_{n=0}^5 a_n \cdot w_n")

"""
You can notice that this sum can mean the activations of preceeding layers will deviate greatly from the original activation values between 0-1. Sometimes, this
is exactly what we want. Sometimes, we wish to avoid that. So how to we tell these neurons how to handle these deviations? That's right, we give them different
flavors, in the form of "activation functions". While some functions will normalize the results to be between 0-1 like sigmoid and softmax, 
others aim to do something completely different, like making all negative values into 0s, like ReLU.
"""
x = tf.convert_to_tensor(np.arange(-5.0, 6.0, 0.1))
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

act_func_map = {
    "Softmax": softmax(x),
    "ReLU": keras.activations.relu(x),
    "Sigmoid": keras.activations.sigmoid(x),
    "Tanh": keras.activations.tanh(x)
}
act_func = st.selectbox("Pick an activation function to see what they do:", ("Softmax", "ReLU", "Sigmoid", "Tanh"))

st.line_chart(pd.DataFrame(act_func_map[act_func], x))

"""
Finally (yes, it's finally over), each neuron will have its own "bias" to add to the weight before applying the activation function. In total, the activation
formula for each neuron in the second layer will be:
"""

st.latex("\delta((\sum_{n=0}^5 a_n \cdot w_n) + b)")

"""
Admittedly, that's a lot of maths, considering usually neural networks have much more neurons than just what was shown in the graph. But thankfully, we have
computers to do that for us! And here, I've got this website to help you make the computer do the work for you, so this is going to be extra easy!
"""

st.sidebar.title("Build-a-rAI")

col1, col2 = st.sidebar.columns(2)

input_layer_1 = col1.number_input("Pick number of input rows:", min_value=1, max_value=128, step=1,
                                  value=16, key="inp-1")
input_layer_2 = col2.number_input("Pick number of input columns:", min_value=1, max_value=128, step=1,
                                  value=16, key="inp-2")
middle_layers = st.sidebar.number_input("Pick number of middle layers:", min_value=1, max_value=6, step=1, value=2)
middle_layers_nodes = []
middle_layers_acts = []
for i in range(middle_layers):
    middle_layers_nodes.append(st.sidebar.number_input("Pick number of neuron for middle layer {0}:".format(i + 1),
                                                       min_value=1, max_value=784, step=1, value=32,
                                                       key="lin{0}".format(i)))
    middle_layers_acts.append(st.sidebar.selectbox("Enter activation function for this layer:", ("Softmax", "ReLU", "Sigmoid", "Tanh"),
                                                   key="act{0}".format(i)))

add_dropout = st.sidebar.checkbox("Add dropout layer?")
st.sidebar.caption("""
A dropout layer set the input to 0 at a rate of your choice. This seems counterintuitive, but it helps your model not rely too much on the dataset.
While the rAI is better with this, you better *not* drop out and stay in school.
""")
dropout_rate = 0
if add_dropout:
    dropout_rate = st.sidebar.number_input("Enter dropout rate:", max_value=1.0, min_value=0.1, step=0.1)
output_layer = st.sidebar.number_input("Pick number of neurons for output layer:", min_value=1, max_value=32, step=1)
train_batch_size = st.sidebar.number_input("Enter batch size:", min_value=1, max_value=128, step=1, value=64)
num_epoch = st.sidebar.number_input("Enter number of training rounds (epochs):", min_value=2, max_value=10, step=1)

st.header("MNIST Dataset")

rans = []
for i in range(9):
    rans.append(randint(1, 60000))

if st.button("Randomize!"):
    rans = []
    for i in range(9):
        rans.append(randint(1, 60000))

fig = plt.figure(figsize=(9, 9))
for i in range(1, 10):
    ran = rans[i - 1]
    fig.add_subplot(3, 3, i)
    plt.title(y_train[ran])
    plt.imshow(x_train[ran], cmap="gray")
    plt.axis("off")
st.pyplot(fig)

"""
The MNIST dataset is a simple dataset with 60,000 **28 by 28** images of hand-written digits. It is considered (by me, at least) to be the Hello world! of machine learning,
so this would be a great start to learn ML. Because these images are grayscale, they can be simplistically seen as 2D matrixes of numbers, 28 rows and 28 columns, with numbers
corresponding to each pixels on the image. The bigger the number is, the whiter the pixel.

So, it's time to get hands-on! On the left of the screen, inside the sidebar will be various parameters that you can tweak to make the rAI model better at recognizing handwritten
digits. The images on the screen will change to fit with the parameters chosen. Once you're done, click Train model and wait for a bit, and the model will be ready for you to test!
"""

st.header("Results")

model = Sequential()
model.add(layers.Flatten(input_shape=(input_layer_1, input_layer_2)))
for i in range(middle_layers):
    model.add(layers.Dense(middle_layers_nodes[i], activation=middle_layers_acts[i]))

if add_dropout:
    model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(output_layer))

model._layers = model._self_tracked_trackables

visualkeras.graph_view(model, to_file="out.png")
visualkeras.layered_view(model, to_file="layered.png", legend=True)
st.image("out.png", caption="Graph view of model")
st.image("layered.png", width=512, caption="Layered view of model")


model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

try:
    if st.sidebar.button("Train this model!"):
        with st.spinner("Training..."):
            model.fit(x_train, y_train, epochs=num_epoch, batch_size=train_batch_size)
            model.save("model.h5")

    if exists("model.h5"):
        loaded = keras.models.load_model("model.h5")
        st.success("Training complete! Model accuracy on 10,000 tests: {0:.2f}%"
                    .format(float(loaded.evaluate(x_test, y_test)[1]) * 100))
        
        with st.container():
            canvas = st_canvas(
                # fill_color="rgba(0, 0, 0, 1)",
                stroke_width=12,
                stroke_color="#ffffff",
                background_color="#000000",
                update_streamlit=True,
                key="canvas",
                height=128,
                width=128
            )

            if canvas.image_data is not None:
                img = canvas.image_data
                img = img[:, :, :-1]
                img = img.mean(axis=2)
                img = img / 255
                img = resize(img, dsize=(28, 28))
                predictions = loaded.predict(np.array([img, img]))
                probability = tf.nn.softmax(predictions).numpy()[0]
                st.image(img, width=128)
                st.write("rAI: This number is *ubdoubtedly* {0}".format(np.argmax(probability)))
        """
        Note that while the rAI isn't *terrible* at guessing the number you wrote, it isn't *exceedingly awesome* at it either.
        This could be due to the real world data (your handwritten digit) being quite different from the handwritten digits in
        the dataset. Fighting the disparity between those and finding the common ground is what makes a machine learning model
        functional.
        """
        st.download_button(
            "Download model",
            file_name="model.h5",
            data="h5"
        )
except tf.errors.InvalidArgumentError:
    st.error("Hmm... it seems like the dimensions of the output is wrong somehow. Recall that are 10 possible outcomes for any guess of a written digit.")
except ValueError:
    st.error("Hmm... it seems like the dimensions of the input is wrong somehow. Recall that the images that you want to input into the model are 28x28.")
