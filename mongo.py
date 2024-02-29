
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client.chatbotdb

qa_pairs = [
    {"question": "What does Inzint help you with?", "answer": "Inzint helps you to expand your digital footprint across mobile and web platforms."},
    {"question": "What kind of company is Inzint?", "answer": "Inzint is a leading software development multinational company providing superior software, web, mobile, and cloud solutions and services to companies globally."},
    {"question": "In which countries does Inzint have a strong technology development and innovation presence?", "answer": "Inzint has a strong technology development and innovation presence in the USA, Australia & India."},
    {"question": "How does Inzint transform businesses?", "answer": "Inzint transforms businesses using the expertise and tech background of the best minds at Inzint"},
    {"question": "What kind of company culture does Inzint have?", "answer": "Inzint is a remote-first company, allowing employees to work from anywhere in the world."},
    {"question": "What approach does Inzint take towards its work?", "answer": "Inzint takes a focused, sound, and innovative approach towards its work, derived from the cumulative experience of its team members."},
    {"question": "What services does Inzint provide?", "answer": "Inzint provides mobile app development, web app development, and cloud computing solutions."},
    {"question": "What is Inzint's email address?", "answer": "You can reach Inzint at contact@inzint.com."},
    {"question": "What is Inzint's contact number?", "answer": "You can contact Inzint at +1 (253) 523-2373."},
    {"question": "What is Inzint's vision?", "answer": "Inzint's vision is to deliver custom software solutions and enhance digital journeys."},
    {"question": "How does Inzint help businesses succeed?", "answer": "Inzint helps businesses succeed through data transformation, digital enablement, and innovation partnerships."},
    {"question": "Who are the directors of Inzint?", "answer": "The directors of Inzint are Vikash Thakur and Jai Deep."},
    {"question": "What is Inzint's mission?", "answer": "Inzint's mission is to support enterprises in their digital business transformation."},
    {"question": "What are Inzint's values to society?", "answer": "Inzint's values are the guiding principles upon which the company was founded."},
    {"question": "What types of systems does Inzint support?", "answer": "Inzint supports various systems including Google Apps, Office365, and custom solutions tailored to your needs."},
    {"question": "How does Inzint handle flat-rate systems?", "answer": "Inzint offers flat-rate billing to budget your IT expenses and provides customized solutions."},
    {"question": "What solutions does Inzint offer for internal IT departments?", "answer": "Inzint provides scalable solutions meticulously tailored to meet the specific needs of internal IT departments."},
    {"question": "Who are some key members of Team Inzint?", "answer": "Key members of Team Inzint include Vikas, Kakoli, Dhiraj, Garima, Amit, Aman, Dennis."},
    {"question": "What collaborative strategies does Inzint employ for innovation?", "answer": "Inzint employs collaborative strategies for innovation, harnessing innovation labs."},
    {"question": "How does Inzint's operational approach contribute to business success?", "answer": "Inzint's operational approach, comprising data transformation and digital enablement."},
    {"question": "What factors make Inzint an ideal choice?", "answer": "Inzint stands out due to its emphasis on hiring smart individuals."},
    {"question": "What industries does Inzint provide software consultation services to?", "answer": "Inzint provides software consultation services to various industries, including banking."},
    {"question": "Which sectors does Inzint specialize in for software consultation?", "answer": "Inzint provides software consultation services to various industries, including banking, healthcare, etc."},
]


qa_collection = db.qa_pairs
qa_collection.insert_many(qa_pairs)

