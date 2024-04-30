from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_model")
checkpoint = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Prepare input text
input_text = "What is the fed fund rate between January 2011 and March 2015?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
outputs = model(**inputs)
predicted_class = outputs.logits.argmax().item()

# Decode the predicted class
label_map = {0: "SQL", 1: "Visualization"} 
predicted_label = label_map[predicted_class]

print("Input Text:", input_text)
print("Predicted Label:", predicted_label)

