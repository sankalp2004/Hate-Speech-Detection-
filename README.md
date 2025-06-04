Hate Speech Detection using LSTM and GloVe

This project classifies an input text into one of three categories:
- Hate Speech
- Not Hate Speech
- Inconclusive

It uses an LSTM-based neural network with pretrained GloVe word embeddings for text representation and classification.

Key Features:

- LSTM Architecture: Captures sequential context in text
- GloVe Embeddings: 100D word vectors
- User Sensitivity Levels: Customize strictness of hate speech detection
- Multi-class Output: Includes "Inconclusive" category for borderline text

Sensitivity Control:

Model behavior can be adjusted based on user preference:
- High Sensitivity: More aggressive detection
- Low Sensitivity: More conservative

Getting Started:

1. Download the GloVe embeddings (100D) from https://nlp.stanford.edu/projects/glove/
2. Install the Python dependencies:

   pip install -r requirements.txt

3. Open the Jupyter Notebook file HateSpeechShown.ipynb and run all cells

Project Structure:

HateSpeechShown.ipynb       - Main notebook
README.md                   - Project overview
requirements.txt            - Python dependencies

License:

MIT License

Author:

Sankalp Jain
GitHub: https://github.com/sankalp2004