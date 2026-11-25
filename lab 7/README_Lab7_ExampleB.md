# Lab 7 – Example B: Reuters Topic Classification with a Bidirectional LSTM

## Aim
To build and evaluate a Bidirectional LSTM model that classifies Reuters newswire articles into 46 different topic categories.

## Dataset Used
The experiment uses the Reuters dataset available through `tensorflow.keras.datasets.reuters`.

- Training samples: 8,982
- Test samples: 2,246
- Vocabulary size: 10,000 most frequent words
- Maximum sequence length: 200
- Number of topic classes: 46 (labels 0–45)

## Model Architecture
The model consists of the following layers:

1. **Input layer:** accepts sequences of 200 tokens.
2. **Embedding layer:** converts word indices into 128-dimensional trainable vectors.
3. **Bidirectional LSTM:** uses 64 units and returns one summary vector.
4. **Dropout layer:** uses a dropout rate of 0.5 to help reduce overfitting.
5. **Dense output layer:** contains 46 neurons with softmax activation, one for each topic.

- Total parameters: 1,384,750
- Trainable parameters: 1,384,750
- Non-trainable parameters: 0

## Training Configuration
- Optimizer: Adam
- Loss function: `sparse_categorical_crossentropy`
- Evaluation metric: Accuracy
- Epochs: 10
- Batch size: 64
- Validation split: 20%

The Reuters topic labels are integer values from 0 to 45. Therefore, `sparse_categorical_crossentropy` is used instead of categorical crossentropy, which expects one-hot encoded labels.

## Results
The model achieved a **test accuracy of 63.09%**.

Final training accuracy: 72.57%

Final validation accuracy: 65.50%

The notebook also displays training and validation loss/accuracy plots.

## Sample Predictions

| Sample | Actual topic | Predicted topic |
|---:|---:|---:|
| 1 | 3 | 4 |
| 2 | 10 | 4 |
| 3 | 1 | 16 |
| 4 | 4 | 4 |
| 5 | 4 | 16 |

## Conclusion
A Bidirectional LSTM model was trained on the Reuters newswire dataset to classify articles into 46 topic categories. The model achieved 63.09% test accuracy. This experiment demonstrates how a sequence-processing backbone can be reused for multi-class classification by changing the output layer to 46 softmax neurons and using sparse categorical crossentropy.

## Output Screenshots
Add screenshots of the following notebook outputs:

- Dataset shape and number of classes
- Model summary
- Training and validation loss/accuracy graphs
- Test accuracy
- Sample topic predictions

## Folder Structure

```text
Lab7_ExampleB/
├── Lab7_Topic_Student_B_(2).ipynb
├── README.md
└── screenshots/
    ├── dataset_output.png
    ├── model_summary.png
    ├── training_plots.png
    ├── test_accuracy.png
    └── sample_predictions.png
```
