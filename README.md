Emotion Detection System
A modular Python application for detecting emotions in text using machine learning.
Features

Text-based emotion analysis with visualization
Batch processing of text files
Training interface for custom emotion models
Intuitive GUI built with Tkinter

Project Structure
emotion_detector/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── emotion_detector.py
├── ui/
│   ├── __init__.py
│   ├── app.py
│   ├── text_tab.py
└── utils/
|   ├── __init__.py
|   └── text_preprocessing.py
└── main.py
Requirements

Python 3.6+
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
neattext
joblib

Installation

Clone this repository:
git clone https://github.com/manishshivhare/Text-Emotion-Detection.git
cd Text-Emotion-Detection

Install the required dependencies:
pip install -r requirements.txt


Usage
Running the Application
python main.py
Using the Application

Text Analysis Tab

Enter text in the input area or select a sample text
Click "Analyze Emotion" to detect the emotions in the text
View the primary emotion and probability distribution chart


Batch Processing Tab

Select an input file (CSV or TXT) containing text to analyze
Choose an output file to save the results
Click "Process Batch" to analyze all texts and generate a report



Training Your Own Model

From the File menu, select "Train New Model"
Choose a dataset file or enable the option to create a sample dataset
Configure model parameters:

Model Type: Naive Bayes or Logistic Regression
Test Size: Percentage of data to use for testing (0.1-0.5)


Set the path to save the trained model
Click "Train Model" to start the training process
View training progress and results in the log

Dataset Format
The application supports the following dataset formats:

CSV File

Must contain 'text' and 'emotion' columns
Example:
text,emotion
"I am so happy today!",happy
"This makes me angry!",angry



TXT File

Each line must follow the format: text;emotion
Example:
I am so happy today!;happy
This makes me angry!;angry




License