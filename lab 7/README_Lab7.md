# Lab 7 -- IMDB Sentiment Analysis using Bidirectional LSTM

**Aim:** To build a deep learning model using word embeddings and a
Bidirectional LSTM to classify movie reviews as positive or negative.

**Dataset Used:** IMDB Movie Reviews dataset from
`tensorflow.keras.datasets.imdb`. The experiment uses the 10,000 most
frequent words, with each review padded or truncated to 200 tokens.

**Results:** The model uses an Embedding layer, Bidirectional LSTM,
Dropout, and a sigmoid output layer. It was trained for 5 epochs using
Adam optimizer and binary crossentropy loss. The model achieved **85.38%
test accuracy**. The sample positive and negative reviews were
classified correctly.

**Output Screenshots:** Add screenshots of the model summary,
training/validation accuracy and loss plots, test accuracy, and sample
sentiment predictions in the `screenshots` folder.

## Folder Structure

``` text
Lab7/
├── Copy_of_Lab7_Sentiment_Student_A.ipynb
├── README.md
└── screenshots/
    ├── model_summary.png
    ├── training_plots.png
    ├── test_accuracy.png
    └── sentiment_predictions.png
```
