import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Sidebar with contact info
st.sidebar.title("Contact Me")
contact_option = st.sidebar.radio("Select contact method:", ["Email", "Mobile No"])
if contact_option == "Email":
    st.sidebar.write("pankaj.sajwan20@gmail.com")
else:
    st.sidebar.write("8477979148")

st.sidebar.title("About Me")
st.sidebar.write("**Name:** Pankaj Sajwan")
st.sidebar.write("**B.Tech (Mechanical Engineering)**")

st.title("Feature Extraction using Autoencoder")

# Dataset selection
dataset_option = st.selectbox("Select Dataset", ["MNIST", "Fashion MNIST"])

if dataset_option == "MNIST":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    st.write(f"Selected dataset: {dataset_option}")
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"y_train shape: {y_train.shape}")
    st.dataframe(X_train[0])
    
elif dataset_option == "Fashion MNIST":
    (X_train_f, y_train_f), (X_test_f, y_test_f) = fashion_mnist.load_data()
    X_train = X_train_f.astype('float32') / 255
    X_test = X_test_f.astype('float32') / 255
    st.write(f"Selected dataset: {dataset_option}")
    st.write(f"X_train shape: {X_train_f.shape}")
    st.write(f"y_train shape: {y_train_f.shape}")
    st.dataframe(X_train_f[0])

input_layer_str = st.text_input("Enter the size of image")

if input_layer_str:
    try:
        input_layer_shape = tuple(map(int, input_layer_str.split(',')))
    except ValueError:
        st.error("Invalid input! Please enter dimensions as comma-separated integers")
        input_layer_shape = None

else:
    input_layer_shape = None

# Display Sample Images
st.subheader("Sample Images from the Dataset")
num_samples = 5
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
for i in range(num_samples):
    axes[i].imshow(X_train[i], cmap='gray')
    axes[i].axis('off')
st.pyplot(fig)

        
if input_layer_shape:
    input_layer = Input(shape=input_layer_shape)
    flatten_layer = Flatten()(input_layer)
    h1 = Dense(units=1024, activation='relu', kernel_initializer='he_uniform')(flatten_layer)
    h2 = Dense(units=512, activation='relu', kernel_initializer='he_uniform')(h1)
    h3 = Dense(units=256, activation='relu', kernel_initializer='he_uniform')(h2)

    x = st.text_input("How much feature want to get")
    if x.isdigit():
        x = int(x)
        bottleneck_layer = Dense(units=x, activation='relu')(h3)
        st.success(f"Model created with input shape {input_layer_shape}")

        h4 = Dense(units=256, activation='relu', kernel_initializer='he_uniform')(bottleneck_layer)
        h5 = Dense(units=512, activation='relu', kernel_initializer='he_uniform')(h4)
        h6 = Dense(units=1024, activation='relu', kernel_initializer='he_uniform')(h5)
        output_layer = Dense(units=input_layer_shape[0] * input_layer_shape[1], activation='sigmoid')(h6)
        final_layer = Reshape(input_layer_shape)(output_layer)

        autoencoder = Model(inputs=input_layer, outputs=final_layer)
        autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

        if st.button("Train Autoencoder"):
            with st.spinner("Training..."):
                encoder_train = autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, validation_split=0.2, verbose=1)
                st.success("Model trained successfully!")

            encoder_model = Model(inputs=input_layer, outputs=bottleneck_layer)
            X_train_new = encoder_model.predict(X_train)
            X_test_new  = encoder_model.predict(X_test)
            
            train_features_df = pd.DataFrame(X_train_new)
            test_features_df = pd.DataFrame(X_test_new)
            train_csv = train_features_df.to_csv(index=False).encode('utf-8')
            test_csv = test_features_df.to_csv(index=False).encode('utf-8')

            st.download_button("Download Training Features CSV", train_csv, "train_features.csv", "text/csv")
            st.download_button("Download Test Features CSV", test_csv, "test_features.csv", "text/csv")

            st.write("Feature extraction complete. You can download the extracted features using the buttons above.")




    else:
        st.error("Please enter a valid number for the bottleneck layer")

else:
    st.warning("Please enter a valid input shape for the image")






    
