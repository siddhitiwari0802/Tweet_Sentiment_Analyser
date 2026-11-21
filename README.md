# Tweet Sentiment Analysis

A sentiment analysis project that classifies tweets into three categories -- negative, neutral, and positive -- using a Bidirectional LSTM neural network built with TensorFlow/Keras. The trained model is served through a FastAPI endpoint for real-time predictions.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Text Preprocessing](#text-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Results](#results)
- [Saved Artifacts](#saved-artifacts)
- [Deployment (FastAPI Server)](#deployment-fastapi-server)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)

---

## About the Project

This project takes raw tweets, cleans them up, and feeds them into a stacked Bidirectional LSTM to figure out whether the overall tone is negative, neutral, or positive. The whole pipeline -- from data loading to model saving and a simple REST API -- is included.

The training was done on Google Colab using a T4 GPU.

---

## Dataset

- **Source file:** `Tweets.csv` (approximately 3.5 MB)
- **Total rows after loading:** 27,481 tweets, each with 4 columns: `textID`, `text`, `selected_text`, and `sentiment`.
- **Null values:** 1 null in `text`, 1 in `selected_text`, 0 in `sentiment` and `textID`.
- **Class distribution (before cleaning):**

| Sentiment | Count  |
|-----------|--------|
| Neutral   | 11,118 |
| Positive  | 8,582  |
| Negative  | 7,781  |

The dataset is moderately imbalanced -- neutral tweets make up roughly 40% of the data, while negative tweets account for about 28%. Class weights were computed and applied during training to handle this.

After dropping rows with missing text or empty cleaned text, the working dataset was split into three parts using stratified sampling:

| Split      | Size   |
|------------|--------|
| Training   | 80%    |
| Validation | 10%    |
| Test       | 10%    |

(The 20% held-out portion was split 50/50 into validation and test sets.)

---

## Text Preprocessing

Each tweet goes through a multi-step cleaning function before being fed to the model:

1. **Lowercasing** -- everything is converted to lowercase.
2. **Emoji conversion** -- emojis are turned into readable text tokens using the `emoji` library (e.g., a laughing face becomes `:face_with_tears_of_joy:`).
3. **Hashtag extraction** -- the `#` symbol is stripped but the word itself is kept (`#sad` becomes `sad`).
4. **Mention and URL removal** -- `@username` handles and `http...` links are dropped.
5. **HTML tag removal** -- any leftover HTML is stripped out.
6. **Repeated character normalization** -- words like `soooo` are compressed to `soo` to cut down on noise while keeping some emphasis.
7. **Punctuation removal** -- standard punctuation is removed.
8. **Stopword removal with negation preservation** -- common English stopwords are removed, but negation words (`not`, `no`, `never`, `don't`, etc.) are explicitly kept since they flip the meaning of a sentence and matter a lot for sentiment.
9. **Optional stemming** -- a Porter Stemmer step is available but was not enabled by default for the final model.
10. **Whitespace normalization** -- extra spaces are collapsed.

After cleaning, tweets that ended up completely empty were dropped.

**Tokenization:**

- Vocabulary size cap: 20,000 most frequent words.
- An `<OOV>` (out-of-vocabulary) token is used for unknown words.
- Sequences are padded/truncated to a fixed length of 40 tokens (post-padding, post-truncating).
- The tokenizer was fitted only on the training set to avoid data leakage.

---

## Exploratory Data Analysis

Two plots were generated:

1. **Sentiment distribution bar chart** -- shows the count of tweets per class, confirming the moderate imbalance.
2. **Cleaned tweet length histogram by sentiment** -- shows the distribution of word counts after cleaning, broken down by class. This helps understand whether longer or shorter tweets lean toward a particular sentiment.

---

## Model Architecture

A Sequential Keras model with the following layers:

| Layer                              | Details                                   |
|------------------------------------|-------------------------------------------|
| Embedding                          | Input dim = vocab size (up to 20,000), output dim = 128, input length = 40 |
| SpatialDropout1D                   | Rate = 0.2 (drops entire embedding channels -- helps with text data)       |
| Bidirectional LSTM (layer 1)       | 64 units, returns sequences, recurrent dropout = 0.1                       |
| Bidirectional LSTM (layer 2)       | 32 units, does not return sequences, recurrent dropout = 0.1              |
| Dropout                            | Rate = 0.4                                |
| Dense                              | 64 units, ReLU activation                 |
| Dropout                            | Rate = 0.3                                |
| Dense (output)                     | 3 units, softmax activation               |

The model uses two stacked Bidirectional LSTM layers so it can read each tweet both forward and backward, picking up context from both directions. The spatial dropout on the embedding layer is used instead of regular dropout because it drops entire feature channels rather than individual values, which tends to work better for text.

**Total parameters:** approximately 32 MB on disk (saved as `sentiment_model.keras`).

---

## Training Setup

- **Optimizer:** Adam with an initial learning rate of 0.001.
- **Loss function:** Sparse categorical cross-entropy (since labels are integer-encoded, not one-hot).
- **Batch size:** 64.
- **Max epochs:** 15.
- **Class weights:** Computed using sklearn's `compute_class_weight("balanced", ...)` to give more importance to under-represented classes during training.

**Callbacks used:**

| Callback             | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| EarlyStopping        | Monitors validation loss, stops if no improvement for 3 consecutive epochs, restores best weights. |
| ReduceLROnPlateau    | Halves the learning rate if validation loss stalls for 2 epochs (minimum LR = 0.00001).             |
| ModelCheckpoint      | Saves the model with the best validation accuracy to `best_model.keras`.                             |

**Training ran for 4 epochs** before early stopping kicked in. The learning rate was reduced once (from 0.001 to 0.0005) at epoch 4.

---

## Results

### Training Curves (4 epochs)

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss | Learning Rate |
|-------|----------------|--------------|------------|----------|---------------|
| 1     | ~0.57          | 0.6856       | ~0.88      | 0.7242   | 0.0010        |
| 2     | ~0.75          | 0.6780       | ~0.59      | 0.7340   | 0.0010        |
| 3     | ~0.83          | 0.6765       | ~0.43      | 0.7767   | 0.0010        |
| 4     | ~0.89          | 0.6820       | ~0.29      | 0.9215   | 0.0005        |

The model overfits after the first epoch -- training accuracy keeps climbing while validation accuracy stays around 68%. Early stopping restored the weights from epoch 1, which had the best validation accuracy of 68.56%.

### Test Set Evaluation

The best model (from epoch 1) was evaluated on the held-out test set (2,743 tweets):

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.68      | 0.66   | 0.67     | 778     |
| Neutral  | 0.65      | 0.65   | 0.65     | 1,107   |
| Positive | 0.75      | 0.77   | 0.76     | 858     |
| **Overall accuracy** | | | **0.69** | **2,743** |
| Macro avg | 0.69     | 0.69   | 0.69     | 2,743   |
| Weighted avg | 0.69  | 0.69   | 0.69     | 2,743   |

Key takeaways:

- **Overall accuracy is 69% on the test set.** Not great, not terrible for three-class tweet sentiment with a relatively small dataset and no pre-trained word embeddings.
- **Positive tweets are the easiest to classify** (F1 = 0.76), likely because positive language tends to be more distinctive.
- **Neutral tweets are the hardest** (F1 = 0.65), which makes sense -- the boundary between neutral and the other two classes is inherently fuzzy.
- The model clearly overfits during training (training accuracy reaches ~89% by epoch 4 while validation stays flat at ~68%). The early stopping and weight restoration help, but there is room for improvement.

### Sample Predictions

| Tweet                                                        | Predicted Sentiment | Confidence (neg / neu / pos)       |
|--------------------------------------------------------------|---------------------|------------------------------------|
| "Sooo SAD I will miss you here in San Diego!!!"              | NEGATIVE            | 0.9357 / 0.0569 / 0.0074          |
| "2am feedings for the baby are fun when he is all smiles..." | POSITIVE            | 0.0218 / 0.1030 / 0.8752          |
| "My Sharpie is running DANGERously low on ink"               | NEUTRAL             | 0.1915 / 0.6807 / 0.1278          |

The model is quite confident on clearly negative and positive tweets, and correctly identifies the ambiguous Sharpie tweet as neutral.

---

## Saved Artifacts

All files needed to load and run the model are in the `saved_model/` directory:

| File                       | Description                                                      |
|----------------------------|------------------------------------------------------------------|
| `sentiment_model.keras`    | The trained Keras model (~32 MB).                                |
| `tokenizer.pkl`            | The fitted Keras Tokenizer (pickle file, ~928 KB).               |
| `label_encoder.pkl`        | The sklearn LabelEncoder mapping (negative=0, neutral=1, positive=2). |
| `config.json`              | Stores `max_len` (40) so the serving code knows how to pad.      |

---

## Deployment (FastAPI Server)

A lightweight REST API is provided in `server.ipynb`. It uses FastAPI and loads the saved model, tokenizer, label encoder, and config at startup.

### Endpoint

**POST** `/predict`

**Request body (JSON):**

```json
{
  "text": "I love this weather!"
}
```

**Response (JSON):**

```json
{
  "sentiment": "positive",
  "confidence": {
    "negative": 0.0123,
    "neutral": 0.0456,
    "positive": 0.9421
  }
}
```

### How the server works

1. On startup, it loads `sentiment_model.keras`, `tokenizer.pkl`, `label_encoder.pkl`, and `config.json` from the `saved_model/` directory.
2. When a POST request hits `/predict`, it runs the same `clean()` function used during training on the input text.
3. The cleaned text is tokenized and padded to the same length (40) as the training data.
4. The model outputs probabilities for each of the three classes. The class with the highest probability is returned as the predicted sentiment, along with all three confidence scores.

### Dependencies for the server

- fastapi
- uvicorn
- tensorflow
- emoji
- nltk

### Running the server locally

```bash
pip install fastapi uvicorn tensorflow emoji nltk
uvicorn server:app --reload
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

**Note:** The server code was written and tested inside Google Colab. To run it locally, you would need to convert the `server.ipynb` notebook into a standalone `server.py` file and update the `SAVE_DIR` path to point to your local `saved_model/` folder.

---

## How to Run

### Training (Colab)

1. Upload `Tweets.csv` to your Google Drive.
2. Open `tweet_sentiment_analysis_upgraded.ipynb` in Google Colab.
3. Update the CSV path in cell 2 to match your Drive location.
4. Run all cells. Training takes roughly 10-12 minutes on a T4 GPU.
5. The model and supporting files will be saved to `saved_model/` in your Drive.

### Inference (Colab or local)

After training, use the `predict_sentiment()` helper function in the training notebook, or start the FastAPI server as described above.

---

## Project Structure

```
nlp_project/
  Tweets.csv                                  -- raw dataset
  tweet_sentiment_analysis_upgraded.ipynb      -- full training pipeline
  tweet_sentiment_analysis                     -- earlier version of the notebook (not used)
  server.ipynb                                 -- FastAPI server code
  saved_model/
    sentiment_model.keras                      -- trained model weights
    tokenizer.pkl                              -- fitted tokenizer
    label_encoder.pkl                          -- label encoder
    config.json                                -- max sequence length config
  README.md                                    -- this file
```

---

## Limitations and Future Work

- **Overfitting is the main issue.** Training accuracy shoots up to nearly 90% while validation accuracy stays around 68-69%. More regularization, data augmentation, or a bigger dataset would help.
- **No pre-trained embeddings.** Using GloVe, FastText, or a similar set of word vectors as the embedding layer instead of learning them from scratch would likely give a noticeable boost, especially with a dataset of this size.
- **The dataset is relatively small** (~27k tweets). Larger datasets like Sentiment140 (~1.6 million tweets) could improve generalization.
- **No transformer-based models were tried.** Fine-tuning something like DistilBERT or a similar model would almost certainly outperform this LSTM setup, though at the cost of more compute.
- **The server is a basic prototype.** It runs inside a Colab notebook and would need to be refactored into a proper Python file with error handling, logging, and configuration management before being used in any real setting.
- **No batched inference.** The API processes one tweet at a time. For bulk analysis, a batch endpoint would be more practical.
