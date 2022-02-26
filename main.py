import keras.losses
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from random import randint
import visualkeras
import pandas as pd

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train, x_test = x_train / 255.0, x_test / 255.0

st.title("Neural Network Education")
# I'll definitely change the name later

st.sidebar.title("Build-a-NN")

col1, col2 = st.sidebar.columns(2)

input_layer_1 = col1.number_input("Pick number of input rows:", min_value=1, max_value=1024, step=1,
                                  value=64, key="inp-1")
input_layer_2 = col2.number_input("Pick number of input columns:", min_value=1, max_value=1024, step=1,
                                  value=64, key="inp-2")
middle_layers = st.sidebar.number_input("Pick number of middle layers:", min_value=1, max_value=5, step=1, value=2)
middle_layers_nodes = []
middle_layers_acts = []
for i in range(middle_layers):
    middle_layers_nodes.append(st.sidebar.number_input("Pick number of neuron for middle layer {0}".format(i + 1),
                                                       min_value=1, max_value=784, step=1, value=32,
                                                       key="lin{0}".format(i)))
    middle_layers_acts.append(st.sidebar.selectbox("Enter activation function:", ("Softmax", "ReLU"),
                                                   key="act{0}".format(i)))

add_dropout = st.sidebar.checkbox("Add dropout layer?")
dropout_rate = 0
if add_dropout:
    dropout_rate = st.sidebar.number_input("Enter dropout rate:", max_value=1.0, min_value=0.1, step=0.1)
output_layer = st.sidebar.number_input("Pick number of neurons for output layer:", min_value=1, max_value=32, step=1)
train_batch_size = st.sidebar.number_input("Enter batch size:", min_value=1, max_value=128, step=1, value=64)
num_epoch = st.sidebar.number_input("Enter number of training rounds (epochs):", min_value=2, max_value=10, step=1)

fig = plt.figure(figsize=(9, 9))
for i in range(1, 10):
    ran = randint(1, 60000)
    fig.add_subplot(3, 3, i)
    plt.title(y_train[ran])
    plt.imshow(x_train[ran], cmap="gray")
    plt.axis("off")
st.pyplot(fig)

act_map = {
    "Softmax": layers.Softmax(),
    "ReLU": layers.ReLU()
}

print(middle_layers_nodes)
print(middle_layers_acts)
model = Sequential()
model.add(layers.Flatten(input_shape=(input_layer_1, input_layer_2)))
for i in range(middle_layers):
    model.add(layers.Dense(middle_layers_nodes[i], activation=middle_layers_acts[i]))

if add_dropout:
    model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(output_layer))
model.summary()

model._layers = model._self_tracked_trackables

visualkeras.graph_view(model, to_file="out.png")
visualkeras.layered_view(model, to_file="layered.png", legend=True)
st.image("out.png")
im1, im2, im3 = st.columns(3)
im1.write("")
im2.image("layered.png")
im3.write("")

model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
try:
    if st.sidebar.button("Train this model!"):
        with st.spinner("Training..."):
            history = model.fit(x_train, y_train, epochs=num_epoch, batch_size=train_batch_size)
            with st.expander("See accuracy and loss graphs"):
                acc = pd.DataFrame(history.history["accuracy"], columns=["Accuracy"])
                loss = pd.DataFrame(history.history["loss"], columns=["Loss"])
                st.line_chart(acc)
                st.line_chart(loss)

        st.success("Training Complete! Model accuracy: {0:.2f}%"
                   .format(float(model.evaluate(x_train, y_train)[1]) * 100))
except tf.errors.InvalidArgumentError:
    st.error("Hmm... it seems like the dimensions of the output is wrong somehow... try to fix it?")
except ValueError:
    st.error("Hmm... it seems like the dimensions of the input is wrong somehow... try to fix it?")
