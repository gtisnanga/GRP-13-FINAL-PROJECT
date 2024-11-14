import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import gradio as gr

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize the lemmatizer and define stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load your pre-trained model and vectorizer from pickle files
with open('./pickle_names/tuned_rf_tf_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('./models/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)   

with open('./models/labels.pkl', 'rb') as file:
    label_categories = pickle.load(file)    

# Define the text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lower case
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    # Remove punctuation and stop words, and lemmatize the tokens
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in string.punctuation and token not in stop_words
    ]
    return ' '.join(cleaned_tokens)

# Define the vectorization function
def vectorize(clean_data):
    return vectorizer.transform([clean_data])  # Transform the cleaned text to TF-IDF

# Define the prediction function
def predict(tf_data):
    prediction = model.predict(tf_data)
    return label_categories[prediction[0]]

# Main function that combines preprocessing, vectorization, and prediction
def execute_flow(username, product, text):
    clean_text = preprocess_text(text)
    vectorized_data = vectorize(clean_text)
    sentiment = predict(vectorized_data)
    # Custom response format
    return f"The user @{username} focused on the product: {product} and said '{text}'\n\nOur model predicted the sentiment to be a {sentiment} emotion!"

# Define Gradio interface elements
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>SentimentFlow</h1>")
    gr.Markdown(
        "<p style='text-align: center;'>This application uses Natural Language Processing to analyze the sentiment behind a text.</p>"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Tweet Here 🐦")
            username = gr.Textbox(label="Username")
            product = gr.Dropdown(
                ["Apple", "Google", "Samsung", "Microsoft"],
                label="Which product do you want to talk about",
            )
            text_input = gr.Textbox(
                label="Tweet",
                placeholder="Enter the text you'd like to analyze...",
            )
            analyze_button = gr.Button("Analyze", variant="primary")

        with gr.Column():
            gr.Markdown("### Prediction 🔭")
            output = gr.Textbox(label="", placeholder="The sentiment prediction will appear here...")

    analyze_button.click(
        fn=execute_flow,
        inputs=[username, product, text_input],
        outputs=output,
    )

# Launch the Gradio app
demo.launch(share=True)
