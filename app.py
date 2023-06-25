
from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = tf.keras.models.load_model('suicide_detection_model.h5')

# Read the tokenizer CSV file
df = pd.read_csv('tokenizer.csv')
word_index = df.set_index('word')['index'].to_dict()

tokenizer = Tokenizer(num_words=10000)
tokenizer.word_index = word_index

# Create a Flask app
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text input from the form
    texts = [request.form.get('text')]
    print(texts)
    new_texts = tokenizer.texts_to_sequences(texts)
    new_texts = pad_sequences(new_texts, maxlen=200)
    print(new_texts)
    predictions = model.predict(new_texts)
    conf = predictions[0]
    # Print the predictions
    if conf > 0.5:
        prediction = 'contains a suicidal message'
    else:
        prediction = 'does not contain a suicidal message'

    # Render the prediction result template
    return render_template('predict.html', prediction=prediction, text=texts[0], confidence=int(conf * 100))

# Start the app
if __name__ == '__main__':
    app.run(debug=True, port=8001)
