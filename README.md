# glb_code
text summarization using KG and transformer

# Setup Instructions
## 1. Clone the Repository


git clone https://github.com/your-repo/text_summarization.git
cd text_summarization
# 2. Install Dependencies
Install the required Python packages using pip:


pip install -r requirements.txt

# 3. Prepare Your Dataset
Ensure your dataset is in the data/ directory in a CSV file named dataset.csv with two columns: text and summary.

Example:


text,summary
"This is the text data.", "This is the summary."
"Another text data example.", "Another summary example."

# Preprocessing
#4. Preprocess the Data
Run the preprocessing script to tokenize the text and build the knowledge graph:


python scripts/preprocess.py
# Model Training
## 5. Train the Model
Train the Transformer model with the processed data:


python scripts/train.py
# 6. Monitor Training
During training, you will see the validation loss after each epoch. You can adjust hyperparameters in the train.py script as needed.

# Summarization
## 7. Generate Summaries
Use the trained model to generate summaries for new text data:


python scripts/summarize.py
Modify the text variable in the summarize.py script to load your input text.

# Evaluation
## 8. Evaluate the Model
Use the provided metrics to evaluate the summarization performance:

# utils/metrics.py
from rouge_score import rouge_scorer

def calculate_metrics(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores
Implement your evaluation script or integrate this function into summarize.py.
