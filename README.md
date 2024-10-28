# Sentiment Analysis GRU Model

This project implements a Sentiment Analysis model using a neural network architecture built with **Keras**. The model classifies text into binary sentiment categories (positive or negative) by leveraging word embeddings and **Gated Recurrent Units (GRU)** layers. Additionally, the project includes a simple **Flask** web application to provide a real-time sentiment analysis interface.

## Installation

Clone the repository and install the necessary dependencies using `pip`.

```python
git clone https://github.com/QuocAnh189/Sentiment_Analysis_EventHub
pip install -r requirements.txt (note python 3.8.10)
```

## Key Features

### Tokenizer Initialization

- A `Tokenizer` object is initialized to handle the text data, limited to the **10,000** most frequent words.
- The tokenizer converts the text into integer sequences using `fit_on_texts(X)` to prepare the input for the model.

### Model Architecture

The model leverages three GRU layers for its architecture:

#### Embedding Layer

- Converts word indices into dense vectors with an embedding size of **20**.
- Handles an input dimension of **20,000** to accommodate the vocabulary.

#### GRU Layers

- **First GRU Layer**: Contains **16 units** and returns sequences.
- **Second GRU Layer**: Contains **8 units** and returns sequences.
- **Third GRU Layer**: Contains **4 units** and does not return sequences.

#### Dense Layer

- A fully connected **Dense layer** with **1 output neuron** using a sigmoid activation function for binary classification (positive/negative sentiment).

### Model Compilation

- **Loss Function**: Binary cross-entropy (`binary_crossentropy`), as it's a binary classification task.
- **Optimizer**: Adam optimizer with a learning rate of **0.001**.
- **Metrics**: The model is evaluated using **accuracy**.

## Usage

To use the Flask web application for real-time sentiment analysis:

### Running the Web Application

1. Start the Flask development server:

   ```bash
   flask run or python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/`. You'll see the home page where you can input text for classification.

3. Enter the text and submit it to get the sentiment prediction.

### Running with Docker

In project terminal run `docker-compose up --build`
Test api in Postman with BASE_URL and GET METHOD: `http://127.0.0.1:5000/predict`

### Model Prediction Flow

- **Tokenization & Padding**: Input text is tokenized using the pre-trained tokenizer and padded to match the expected input length of **241 tokens**.
- **Prediction Output**: The model outputs a probability between **0** and **1**. If the value is greater than **0.5**, the sentiment is classified as **Positive**, otherwise it's classified as **Negative**.

## Example

Here’s a quick example to demonstrate how the sentiment classification works:

1. Input: `"I love this event, it is interesting"`

   - Output: **Positive** (e.g., 0.85 probability)

2. Input: `"The event was terrible and very disappointing."`
   - Output: **Negative** (e.g., 0.15 probability)

## Author Contact

Contact me with any questions!<br>

Email: anquoc18092003@gmail.com
Facebook: https://www.facebook.com/tranphuocanhquoc2003

<p style="text-align:center">Thank You so much for your time !!!</p>
