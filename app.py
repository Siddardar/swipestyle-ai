from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Embedding, Flatten, Dot, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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

def create_model(num_users, num_items, embedding_size=50, reg_strength=1e-3, dropout_rate=0.5):
    regularizer = l2(reg_strength)
    user_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_size, input_length=1, embeddings_regularizer=regularizer)(user_input)
    user_vector = Flatten()(user_embedding)
    user_vector = Dropout(dropout_rate)(user_vector)

    item_input = Input(shape=(1,))
    item_embedding = Embedding(num_items, embedding_size, input_length=1, embeddings_regularizer=regularizer)(item_input)
    item_vector = Flatten()(item_embedding)
    item_vector = Dropout(dropout_rate)(item_vector)

    dot_product = Dot(axes=1)([user_vector, item_vector])
    model = Model(inputs=[user_input, item_input], outputs=dot_product)
    model.compile(optimizer=Adam(), loss='mse')
    return model

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_name = data['user_name']
    top_n = data.get('top_n', 25)
    brands = data.get('brands')
    
    allowed_items = {item["productID"]: i for i, item in enumerate(clothing_data) if item['brand'] in brands}


    user_id = usernames[user_name]
    user_vector = model.layers[2].get_weights()[0][user_id]
    item_vectors = model.layers[3].get_weights()[0]

    scores = np.dot(item_vectors, user_vector)
    recommended_item_ids = np.argsort(scores)[::-1][:top_n]
    recommended_items = [clothing_data[item_id] for item_id in recommended_item_ids if item_id in allowed_items.values()]
            
        

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
        if user_id is None:
            user_id = len(usernames)
            usernames[user["id"]] = user_id

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

    # Debugging: Print the shapes of the arrays
    print(f'user_ids_array shape: {user_ids_array.shape}')
    print(f'item_ids_array shape: {item_ids_array.shape}')
    print(f'labels_array shape: {labels_array.shape}')

    # Check if the arrays are not empty
    if user_ids_array.size == 0 or item_ids_array.size == 0 or labels_array.size == 0:
        return jsonify({"error": "No data to update"}), 400

    # Determine the new size of the embeddings
    max_user_id = user_ids_array.max()
    max_item_id = item_ids_array.max()
    num_users = max(max_user_id + 1, len(usernames))
    num_items = max(max_item_id + 1, len(clothes))

    # Create a new model with updated embedding sizes
    new_model = create_model(num_users, num_items)

    # Transfer weights
    old_user_weights = model.layers[2].get_weights()[0]
    new_user_weights = np.zeros((num_users, old_user_weights.shape[1]))
    new_user_weights[:old_user_weights.shape[0], :] = old_user_weights

    old_item_weights = model.layers[3].get_weights()[0]
    new_item_weights = np.zeros((num_items, old_item_weights.shape[1]))
    new_item_weights[:old_item_weights.shape[0], :] = old_item_weights

    new_model.layers[2].set_weights([new_user_weights])
    new_model.layers[3].set_weights([new_item_weights])

    # Incrementally train the new model
    new_model.fit([user_ids_array, item_ids_array], labels_array, epochs=1, verbose=1)

    # Save the updated model
    new_model.save(model_path)
    
    return jsonify({"message": "Model updated successfully!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 
