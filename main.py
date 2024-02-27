
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

qa_pairs = {
    "What does Inzint help you with?": "Inzint helps you to expand your digital footprint across mobile and web platforms.",
    "What kind of company is Inzint?": "Inzint is a leading software development multinational company providing superior software, web, mobile, and cloud solutions and services to companies globally.",
    "In which countries does Inzint have a strong technology development and innovation presence?": "Inzint has a strong technology development and innovation presence in the USA, Australia & India.",
    "How does Inzint transform businesses?": "Inzint transforms businesses using the expertise and tech background of the best minds at Inzint, resulting in customer satisfaction with the help of powerful and adaptable digital solutions.",
    "What kind of company culture does Inzint have?": "Inzint is a remote-first company, allowing employees to work from anywhere in the world.",
    "What approach does Inzint take towards its work?": "Inzint takes a focused, sound, and innovative approach towards its work, derived from the cumulative experience of its team members.",
    "What services does Inzint provide?": "Inzint provides mobile app development, web app development, and cloud computing solutions.",
    "What is Inzint's email address?": "You can reach Inzint at contact@inzint.com.",
    "What is Inzint's contact number?": "You can contact Inzint at +1 (253) 523-2373.",
    "What is Inzint's vision?": "Inzint's vision is to deliver custom software solutions and enhance digital journeys.",
    "How does Inzint help businesses succeed?": "Inzint helps businesses succeed through data transformation, digital enablement, and innovation partnerships.",
    "Who are the directors of Inzint?": "The directors of Inzint are Vikash Thakur and Jai Deep.",
    "What is Inzint's mission?": "Inzint's mission is to support enterprises in their digital business transformation.",
    "What are Inzint's values to society?": "Inzint's values are the guiding principles upon which the company was founded.",
    "What types of systems does Inzint support?": "Inzint supports various systems including Google Apps, Office365, and custom solutions tailored to your needs.",
    "How does Inzint handle flat-rate systems?": "Inzint offers flat-rate billing to budget your IT expenses and provides customized solutions.",
    "What solutions does Inzint offer for internal IT departments?": "Inzint offers scalable solutions tailored to meet the specific needs of internal IT departments.",
    "Who are some members of Team Inzint?": "Some members of Team Inzint include Vikas, Kakoli, Dhiraj, Garima, Amit, Aman, Dennis, Jayesh, Samkeet, Mordhawaj, Paras, Shivam, Hardik, Saksham, Mohini, Vartika Pandey, Vanshika, Kunal, Radhika.",
    "What collaborative approach does Inzint adopt for innovation?":"Inzint adopts a collaborative approach to innovation by leveraging innovation labs, future financial ecosystems, alliances & partners.",
    "How does Inzint's approach contribute to business success?": "Inzint's approach includes data transformation and digital enablement to accelerate the 'Data-to-Insight-to-Action' cycle and drive key business outcomes",
    "What are the reasons for choosing Inzint?": "Inzint emphasizes hiring smart people, providing superior customer service, dedicated support, and delivering cutting-edge IT solutions."



}



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    question = request.get_json()['question']
    context = " ".join(qa_pairs.values()) 

    inputs = tokenizer.encode_plus(question, context, return_tensors="tf", add_special_tokens=True)


    outputs= model(inputs)
    start_logits,end_logits=outputs.start_logits,outputs.end_logits


    answer_start = tf.argmax(start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(end_logits, axis=1).numpy()[0]

    if answer_start == answer_end: 
        # answer = "Sorry, I don't have information about that yet."
        response = model_gen.generate_content(question)
        answer = response.text
    else:
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
        answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end + 1])


    return jsonify({'answer': answer})




if __name__ == '__main__':
    app.run(debug=True)




