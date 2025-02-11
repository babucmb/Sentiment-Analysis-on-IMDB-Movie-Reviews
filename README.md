# Sentiment-Analysis-on-IMDB-Movie-Reviews


## Project Overview
This project implements a sentiment analysis model using a deep learning approach with an LSTM-based neural network. The model is trained on the IMDB movie reviews dataset to classify reviews as either **positive** or **negative**.

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn


## Dataset
We use the **IMDB dataset** available in TensorFlow. This dataset contains 50,000 movie reviews, each labeled as either positive (1) or negative (0). We use the top 10,000 most frequent words from the dataset.

## Steps Followed
1. **Import Libraries**: Load required Python libraries such as TensorFlow, NumPy, Pandas, and Matplotlib.
2. **Load Dataset**: Fetch the IMDB dataset using TensorFlow's `imdb.load_data()` function.
3. **Preprocess Data**:
   - Limit the vocabulary to the top 10,000 words.
   - Pad sequences to ensure uniform input length.
4. **Build LSTM Model**:
   - Use an embedding layer to convert words into dense vectors.
   - Implement a bidirectional LSTM layer for better learning.
   - Add dropout layers to reduce overfitting.
   - Use a fully connected layer with a sigmoid activation function for binary classification.
5. **Compile & Train Model**:
   - Use `adam` optimizer and `binary_crossentropy` loss function.
   - Implement early stopping to prevent overfitting.
   - Train the model for 5 epochs with a batch size of 128.
6. **Evaluate Model**:
   - Calculate loss and accuracy on the test dataset.
   - Generate a confusion matrix.
   - Display a classification report.
7. **Visualize Results**:
   - Plot training accuracy and loss over epochs.
   - Display confusion matrix using Seaborn.

## How to Run
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
   ```
2. Run the script:
   ```bash
   python sentiment_analysis.py
   ```
3. The model will train and evaluate automatically.
4. The generated plots will be saved as `sentiment_analysis_output.png`.

## Output
- **Accuracy & Loss Graphs**: Shows model performance.
- **Confusion Matrix**: Displays classification results.
- **Classification Report**: Summarizes precision, recall, and F1-score.

## Future Improvements
- Use a pre-trained embedding like GloVe or Word2Vec for better accuracy.
- Increase dataset size or fine-tune hyperparameters.
- Experiment with different architectures (GRU, Transformer models).

## Author
*CMB - Computer Science & Engineering (AI & ML Specialization)*

