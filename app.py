from flask import Flask, request, jsonify
from langchain_experimental.agents import create_csv_agent
# from langchain.community.llms import OpenAI
from langchain_openai import OpenAI
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

csv_file_path = './dseats_2024_training_dataset.csv'
df = pd.read_csv(csv_file_path)

agent = create_csv_agent(OpenAI(temperature=0),
                         csv_file_path,
                         verbose=True,
                         allow_dangerous_code=True)

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    request_data = request.get_json()
    user_query = request_data.get('query', '')

    response = agent.run(user_query)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
