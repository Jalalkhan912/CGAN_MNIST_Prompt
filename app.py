import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# Load models
loaded_generator = tf.keras.models.load_model('generator.h5')
nlp_model = tf.keras.models.load_model('MNIST_NLP.h5')
latent_dim = 100  # Adjust this if your model uses a different latent dimension

# Load the JSON file correctly
with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()  # Read the file as a string

# Convert JSON string to tokenizer object
tokenizer = tokenizer_from_json(tokenizer_json)

# Function to generate specific digits
def generate_specific_digits(loaded_generator, digit, num_samples=5):
    noise = tf.random.normal([num_samples, latent_dim])
    labels = tf.fill([num_samples, 1], digit)
    generated_images = loaded_generator([noise, labels], training=False)
    generated_images = generated_images * 0.5 + 0.5  # Rescale to [0, 1]
    return generated_images

# Function to plot generated digits
def plot_generated_digits(images, digit):
    num_images = images.shape[0]
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        axs[i].imshow(images[i, :, :, 0], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Digit: {digit}')
    plt.tight_layout()

    # Save plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# Function to predict from text using NLP model
def predict_from_text(model, tokenizer, text, max_length=8):
    # Convert text to a sequence
    sequence = tokenizer.texts_to_sequences([text])
    
    # Pad the sequence to match max length
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    
    # Get predictions for digit and quantity
    digit_prob, quantity_prob = model.predict(padded_sequence)
    
    # Convert probabilities to integers
    digit_pred = np.argmax(digit_prob, axis=1)[0]
    quantity_pred = np.argmax(quantity_prob, axis=1)[0]
    
    return digit_pred, quantity_pred

# Streamlit UI
st.title("Handwritten Digit Generator")
st.write("Describe the digits you want to generate using natural language.")

# User input for text-based request
user_input = st.text_input("Enter your request (e.g., 'Generate 5 images of the digit 7'): ")

if st.button("Generate"):
    if user_input:
        with st.spinner("Processing your request..."):
            # Use NLP model to predict the digit and quantity
            digit_to_generate, num_samples = predict_from_text(nlp_model, tokenizer, user_input)

            # Generate the images
            generated_images = generate_specific_digits(loaded_generator, digit_to_generate, num_samples)

            # Plot the generated images
            image_buf = plot_generated_digits(generated_images, digit_to_generate)

        st.image(image_buf, caption=f"Generated {num_samples} images of digit {digit_to_generate}.")
        st.success("Images generated successfully!")
    else:
        st.warning("Please enter a request before generating images.")
