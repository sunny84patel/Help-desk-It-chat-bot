
from flask_pymongo import PyMongo
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import google.generativeai as genai


app = Flask(__name__)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

API_KEY = "AIzaSyCB0FsriiPfyTLwZGM9z_cDLdl03MFjeFQ"
gai = genai.configure(api_key=API_KEY)

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
]
model_gen = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config, safety_settings=safety_settings)

app.config["MONGO_URI"] = "mongodb://localhost:27017/qa_pairs"
mongo = PyMongo(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
# def chat():
#     question = request.get_json()['question']
    
#     # Fetch questions from MongoDB
#     qa_collection = mongo.db.qa_pairs
#     qa_pair = qa_collection.find_one({"question": question})
    
#     if qa_pair:
#         context = " ".join(qa_pair.values()) 

#         inputs = tokenizer.encode_plus(question, context, return_tensors="tf", add_special_tokens=True)


#         outputs= model(inputs)
#         start_logits,end_logits=outputs.start_logits,outputs.end_logits

#         answer_start = tf.argmax(start_logits, axis=1).numpy()[0]
#         answer_end = tf.argmax(end_logits, axis=1).numpy()[0]
#         tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
#         answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end + 1])

        
#     else:
#         response = model_gen.generate_content(question)
#         answer = response.text

#     return jsonify({'answer': answer})

def chat():
    question = request.get_json()['question']
    
    # Fetch questions from MongoDB
    qa_collection = mongo.db.qa_pairs
    qa_pairs = qa_collection.find({"question": question})
    print(qa_pairs)
    
    if qa_pairs:
        for qa_pair in qa_pairs:
            context = qa_pair["answer"]  # Use the answer as context
            
            inputs = tokenizer.encode_plus(question, context, return_tensors="tf", add_special_tokens=True)
            outputs = model(inputs)
            
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            answer_start = tf.argmax(start_logits, axis=1).numpy()[0]
            answer_end = tf.argmax(end_logits, axis=1).numpy()[0]
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
            answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end + 1])
            
            return jsonify({'answer': answer})
        
    # If no matching question found
    return jsonify({'answer': "Sorry, I don't have information about that yet."})



if __name__ == '__main__':
    app.run(debug=True)

