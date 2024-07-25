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

@app.route('/createUser', methods=['POST'])
def create_user():
    data = request.get_json()
    user_name = data['user_name']

    if user_name in usernames:
        return jsonify({"message": "User already exists!"})

    user_id = len(usernames)
    usernames[user_name] = user_id

    return jsonify({"message": "User created!"})

@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    new_user_data = data['new_user_data']
    
    user_ids = []
    item_ids = []
    labels = []

    for user in new_user_data:
        user_id = usernames.get(user["id"])

        for item in user["liked_items"]:
            item_id = clothes.get(item["productID"])
            if item_id is None:
                item_id = len(clothes)
                clothes[item["productID"]] = item_id
            user_ids.append(user_id)
            item_ids.append(item_id)
            labels.append(1)

        for item in user["disliked_items"]:
            item_id = clothes.get(item["productID"])
            if item_id is None:
                item_id = len(clothes)
                clothes[item["productID"]] = item_id
            user_ids.append(user_id)
            item_ids.append(item_id)
            labels.append(-1)

    user_ids_array = np.array(user_ids)
    item_ids_array = np.array(item_ids)
    labels_array = np.array(labels)

    model.fit([user_ids_array, item_ids_array], labels_array, epochs=1, verbose=1)

    model.save('swipestyle-ai.keras')
    
    return jsonify({"message": "Model updated!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 
