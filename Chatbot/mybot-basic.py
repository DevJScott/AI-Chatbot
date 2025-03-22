#!/usr/bin/env python3

import aiml
import wikipedia
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import tkinter as tk
from tkinter import filedialog
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

read_expr = Expression.fromstring


# --- Preprocessing function for text similarity ---
def preprocess_text(text):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)


# --- Image Classification Component ---
class PokemonClassifier:
    def __init__(self, model_path='pokemon_classifier.h5', class_names_path='pokemon_classes.json'):
        self.model_loaded = False

        try:
            # Load the model if it exists
            if os.path.exists(model_path):
                self.model = load_model(model_path)

                # Load class names if they exist
                if os.path.exists(class_names_path):
                    with open(class_names_path, 'r') as f:
                        self.class_names = json.load(f)
                else:
                    # Default class names if file doesn't exist
                    self.class_names = ["Bulbasaur", "Charmander", "Pikachu", "Squirtle", "Other"]

                # Get image dimensions from model
                self.img_width = self.model.input_shape[1]
                self.img_height = self.model.input_shape[2]
                self.model_loaded = True
                print("Image classifier loaded successfully!")
            else:
                print(f"Image classifier model not found at {model_path}")
        except Exception as e:
            print(f"Error loading image classifier: {e}")

    def classify_image(self):
        if not self.model_loaded:
            return "Image classification model is not available."

        # Create a temporary root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select a Pokemon image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        # Destroy the temporary window
        root.destroy()

        if not file_path:
            return "No image was selected."

        try:
            # Load and preprocess the image
            img = image.load_img(file_path, target_size=(self.img_width, self.img_height))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize

            # Make prediction
            predictions = self.model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])

            return f"I can see that this is a {predicted_class} (confidence: {confidence:.2f})."
        except Exception as e:
            return f"Sorry, I couldn't process that image: {str(e)}"


# --- Load AIML Bot ---
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")
print("AIML bot initialized successfully!")

# --- Load Q/A Knowledge Base ---
try:
    qa_df = pd.read_csv("qa_kb.csv")
    if 'Question' not in qa_df.columns or 'Answer' not in qa_df.columns:
        raise ValueError("CSV must have 'Question' and 'Answer' columns.")

    # Preprocess questions
    qa_df['Processed_Question'] = qa_df['Question'].apply(preprocess_text)

    # Prepare TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(qa_df['Processed_Question'].tolist())

    print("Q/A Knowledge base loaded successfully!")

except Exception as e:
    print(f"Error loading Q/A Knowledge Base: {e}")
    qa_df = None

# --- Load Logical Knowledge Base ---
kb = []
try:
    logic_data = pd.read_csv('logical-kb.csv', header=0, names=["Fact", "Type"], delimiter=",", quotechar='"')
    for _, row in logic_data.iterrows():
        expr = read_expr(row["Fact"])
        kb.append(expr)

    print("Logical knowledge base loaded successfully!")

except Exception as e:
    print(f"Error loading Logical KB: {e}")
    kb = []

# --- Initialize Image Classifier ---
pokemon_classifier = PokemonClassifier()


# --- TF-IDF Similarity Search ---
def find_most_similar_question(user_input):
    if qa_df is None:
        return None

    # Preprocess user input
    processed_input = preprocess_text(user_input)

    # Vectorize the input
    user_input_tfidf = vectorizer.transform([processed_input])

    # Calculate similarities
    similarities = cosine_similarity(user_input_tfidf, tfidf).flatten()
    max_sim = similarities.max()

    if max_sim < 0.3:
        return None  # Ignore weak matches

    return qa_df.iloc[similarities.argmax()]['Answer']


# --- Welcome User ---
print("Welcome to this Pokémon chatbot. Please feel free to ask questions!")

# --- Chat Loop ---
while True:
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break

    # Get AIML response
    answer = kern.respond(userInput)

    # --- AIML Command Handling ---
    if answer.startswith('#'):
        params = answer[1:].split('$')
        cmd = int(params[0])

        if cmd == 0:  # Exit command
            print(params[1])
            break

        elif cmd == 1:  # Wikipedia Query
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=True)
                print(wSummary)
            except Exception:
                print("Sorry, I do not know that. Please be more specific!")

        elif cmd == 31:  # "I know that * is *"
            obj, subj = params[1].split(' is ')
            expr = read_expr(f"{subj}({obj})")

            if ResolutionProver().prove(expr.negate(), kb):
                print(f"Error: '{obj} is {subj}' contradicts existing knowledge!")
            else:
                kb.append(expr)
                print(f"OK, I will remember that {obj} is {subj}")

        elif cmd == 32:  # "Check that * is *"
            obj, subj = params[1].split(' is ')
            expr = read_expr(f"{subj}({obj})")

            if ResolutionProver().prove(expr, kb):
                print("Correct.")
            else:
                negation_result = ResolutionProver().prove(expr.negate(), kb)
                print("Incorrect." if negation_result else "Sorry, I don't know.")

        elif cmd == 40:  # Image classification
            if pokemon_classifier.model_loaded:
                result = pokemon_classifier.classify_image()
                print(result)
            else:
                print("Image classification is not available.")

        elif cmd == 99:
            print("I did not get that, please try again.")

    elif answer:  # If AIML response is found
        print(answer)

    else:  # No AIML response, check similarity-based response
        similar_answer = find_most_similar_question(userInput)
        print(similar_answer if similar_answer else "Sorry, I don't know the answer. Can you rephrase?")
