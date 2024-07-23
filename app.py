from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

import os
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client

from main import PreProcessData

app = Flask(__name__)

# Load your model
model_path = 'swipestyle-ai.keras'
model = load_model(model_path)

# Load user and clothing data
load_dotenv(find_dotenv())

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client: Client = create_client(supabase_url, supabase_key)

user_data = supabase_client.from_('users').select('id').execute().data
clothing_data = PreProcessData().fetch_clothing_data()

usernames = {user['id']: i for i, user in enumerate(user_data)}
clothes = {item["productID"]: i for i, item in enumerate(clothing_data)}

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_name = data['user_name']
    top_n = data.get('top_n', 25)

    user_id = usernames[user_name]
    user_vector = model.layers[2].get_weights()[0][user_id]
    item_vectors = model.layers[3].get_weights()[0]

    scores = np.dot(item_vectors, user_vector)
    recommended_item_ids = np.argsort(scores)[::-1][:top_n]
    recommended_items = [clothing_data[item_id] for item_id in recommended_item_ids]

    return jsonify(recommended_items)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 
