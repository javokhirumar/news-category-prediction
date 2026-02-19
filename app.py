import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Load tokenizer and model architecture
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

# Load your fine-tuned weights
model.load_state_dict(torch.load("news_model.pth", map_location=torch.device('cpu')))
model.eval()

# Map label IDs to class names
label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

# Prediction function
def predict_news_category(text):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1).item()
    return label_map[prediction]

# Gradio interface
interface = gr.Interface(
    fn=predict_news_category,
    inputs=gr.Textbox(lines=5, placeholder="Enter news text here..."),
    outputs="text",
    title="News Category Prediction",
    description="Enter the news text here to predict its category (World, Sports, Business, Sci/Tech)."
)

interface.launch()
