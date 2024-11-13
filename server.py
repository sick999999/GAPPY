# server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from llm import llm_rag
from llm2 import llm_rag2
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def home():
    user_message = request.json['message']
    ans = llm_rag(user_message)
    return jsonify({'message': ans})


@app.route('/receipt', methods=['POST'])
def receipt():
    cart = request.json['labels']
    print(cart)
    ans = llm_rag2(cart)
    print(ans)
    return {'message': ans}

if __name__ == '__main__':
    app.run(debug=True)

