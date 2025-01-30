
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import google.generativeai as genai

app = Flask(__name__)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

API_KEY = ""
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
    "How does Inzint transform businesses?": "Inzint transforms businesses using the expertise and tech background of the best minds at Inzint",
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
    "What solutions does Inzint offer for internal IT departments?": "Inzint provides scalable solutions meticulously tailored to meet the specific needs of internal IT departments.",
    "Who are some key members of Team Inzint?": "Key members of Team Inzint include Vikas, Kakoli, Dhiraj, Garima, Amit, Aman, Dennis.",
    "What collaborative strategies does Inzint employ for innovation?": "Inzint employs collaborative strategies for innovation, harnessing innovation labs.",
    "How does Inzint's operational approach contribute to business success?": "Inzint's operational approach, comprising data transformation and digital enablement.",
    "What factors make Inzint an ideal choice?": "Inzint stands out due to its emphasis on hiring smart individuals.",
    "What industries does Inzint provide software consultation services to?": "Inzint provides software consultation services to various industries, including banking",
    "Which sectors does Inzint specialize in for software consultation?": "Inzint provides software consultation services to various industries, including banking,healthcare etc",
    "Are there any specific industries Inzint focuses on for its software consultation services?": "Inzint provides software consultation services",
    "Can you list some examples of industries that benefit from Inzint's software consultation services?": "Inzint provides software consultation services",
    "Are there any niche industries that Inzint caters to for software consultation?": "Inzint provides software consultation services to various industries, including banking",
    "Does Inzint tailor its software consultation services based on industry requirements?": "Inzint provides software consultation services to various industries, including banki.",
    "What are some of Inzint's core values?": "Inzint's core values include customer centricity, transparency, support, and quality.",
    "Can you elaborate on Inzint's commitment to customer centricity?": "Inzint's core values include customer centricity, transparency, support, and quality.",
    "How does Inzint uphold transparency in its operations?": "Inzint's core values include customer centricity, transparency, support, and quality.",
    "What forms of support does Inzint offer to its clients?": "Inzint's core values include customer centricity, transparency, support, and quality.",
    "How does Inzint ensure quality in its services?": "Inzint's core values include customer centricity, transparency, support, and quality.",
    "Are there any other core values that Inzint prioritizes besides the ones mentioned?": "Inzint's core values include customer centricity, transparency, support, and quality.",
    "What is Inzint's mission?": "Inzint's mission is to support enterprises towards digital business transformatio by adhering to customer value propositions ",
    "How does Inzint aim to facilitate digital business transformation?": "Inzint's mission is to support enterprises towards digital business transformation",
    "Can you explain how Inzint incorporates frontier technologies into its mission?": "Inzint's mission is to support enterprises towards digital business transformation",
    "What role do customer value propositions play in Inzint's mission?": "Inzint's mission is to support enterprises towards digital business transformation",
    "Is there a specific approach that Inzint follows to achieve its mission?": "Inzint's mission is to support enterprises towards digital business transformation",
    "What is Inzint's vision?": "Inzint's vision is to be a determined organization working to deliver custom software solutions to enhance digital journeys.",
    "How does Inzint define a custom software solution?": "Inzint's vision is to be a determined organization working to deliver custom software solutions.",
    "Can you elaborate on what Inzint means by 'digital journeys'?": "Inzint's vision is to be a determined organization working to deliver custom software ",
    "What distinguishes Inzint's approach to delivering custom software solutions?": "Inzint's vision is to be a determined organization working.",
    "How long has Inzint been providing IT services?": "Inzint has been at the forefront of the technology landscape since its inception in 2018.",
    "Are there any indicators of Inzint's experience?": "Inzint has been at the forefront of the technology landscape since its inception in 2018",
    "What services does the company provide?":"The company provides custom software development, cloud computing services, hosting services, mobile app development, web development, software consultation, and technology barrier identification and overcoming services.",
    
    "What is the focus of the custom software provided by the company?":"The custom software provided by the company is designed according to the client's needs using the latest technology.",
    
    "What technology does the company utilize for cloud computing?":"The company starts development with cloud computing and delivers hosting services. It utilizes remote resources for hiring without the need for additional entities, thus increasing productivity and adaptability.",
    
    "What expertise does the company have in mobile app development?":"The company offers mobile app development expertise in developing Android as well as iOS apps, working across multiple platforms.",
    
    "What kind of web development services does the company offer?":"The company offers scalable web development services, aiming to automate work processes and enhance user experience.",
    
    "How does the company provide software consultation?":"The company provides software consultation services where consultants help identify and overcome technology barriers that hinder business progress.",
    
    "What is the company's approach to consultancy?":"The company sees consultancy as a partnership and aims to help identify and overcome technology barriers that hinder business progress. It considers itself a partner of choice in consultancy.",
    
    "Where is the company based?":"The company, Inzint, pioneered software services in Noida and has been consistently delivering business value with the latest technology.",
    
    "What is the company's mission and vision?":"The company's mission and vision are determined to deliver custom software solutions to enhance the digital journey using frontier technology and to deliver challenges to businesses.",
    
    "Who are the technology partners of the company?":"One of the defining principles of Inzint is everything, however, it still relies on leading technology partners to provide products that meet the team's highly experienced management team's requirements.",
    
    "What is the company's approach to hiring?":"The company hires highly experienced and passionate engineers who are dedicated to providing reliable and secure technology solutions.",
    
    "What is the company's approach to customer service?":"The company strives to provide superior customer service to ensure every client is completely satisfied with the work and support provided by trustworthy, dedicated, and experienced engineers.",
    
    "What is the company's approach to quality?":"The company is committed to delivering outstanding and cutting-edge solutions that add real value to clients' businesses. It goes beyond just meeting the needs of the clients and ensures quality delivery.",
    
    "How does the company approach business success?":"The company believes that people impact success, thus it hires smart people who are passionate problem solvers. It is determined to deliver custom software solutions to enhance the digital journey using frontier technology."
}



@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     question = request.get_json()['question']
#     context = " ".join(qa_pairs.values()) 

#     inputs = tokenizer.encode_plus(question, context, return_tensors="tf", add_special_tokens=True)


#     outputs= model(inputs)
#     start_logits,end_logits=outputs.start_logits,outputs.end_logits


#     answer_start = tf.argmax(start_logits, axis=1).numpy()[0]
#     answer_end = tf.argmax(end_logits, axis=1).numpy()[0]

#     if answer_start == answer_end: 
#         # answer = "Sorry, I don't have information about that yet."
#         response = model_gen.generate_content(question)
#         answer = response.text
#     else:
#         tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
#         answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end + 1])


#     return jsonify({'answer': answer})




# if __name__ == '__main__':
#     app.run(debug=True)




@app.route('/chat', methods=['POST'])
def chat():
    question = request.get_json()['question']
    context = " ".join(qa_pairs.values()) 

    # Tokenize question and context
    inputs = tokenizer.encode_plus(question, context, return_tensors="tf", add_special_tokens=True)

    # Truncate input if it exceeds maximum sequence length
    max_seq_length = 512
    if inputs['input_ids'].shape[1] > max_seq_length:
        inputs['input_ids'] = inputs['input_ids'][:, :max_seq_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :max_seq_length]

    # Get model outputs
    outputs = model(inputs)
    start_logits, end_logits = outputs.start_logits, outputs.end_logits

    answer_start = tf.argmax(start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(end_logits, axis=1).numpy()[0]

    if answer_start == answer_end: 
        # If the model fails to find an answer, use generative AI
        response = model_gen.generate_content(question)
        answer = response.text
    else:
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
        answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end + 1])

    return jsonify({'answer': answer})


if __name__ == '__main__':
     
    app.run(debug=True)

