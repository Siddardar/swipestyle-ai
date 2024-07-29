from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Embedding, Flatten, Dot, Input, Dropout, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import os
import json
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client

from main import FetchData

app = Flask(__name__)

# Load your model
model_path = 'swipestyle-ai.keras'
model = None

clothing_data = FetchData().clothing_data()

with open('./data/usernames.json') as f:
    usernames = json.load(f)
with open('./data/clothes.json') as f:
    clothes = json.load(f)
with open('./data/brands.json') as f:
    brands = json.load(f)
with open('./data/genders.json') as f:
    genders = json.load(f)

def load_global_model():
    global model
    model = load_model(model_path)

load_global_model()

def get_embedding_layer(model, index):
    count = 0
    for layer in model.layers:
        if isinstance(layer, Embedding):
            if count == index:
                return layer
            count += 1
    return None


def create_model(num_users, num_items, num_brands, num_genders, embedding_size=50, reg_strength=1e-3, dropout_rate=0.5):
    regularizer = l2(reg_strength)

    # User input and embedding
    user_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_size, embeddings_regularizer=regularizer)(user_input)
    user_vector = Flatten()(user_embedding)
    user_vector = Dropout(dropout_rate)(user_vector)

    # Item input and embedding
    item_input = Input(shape=(1,))
    item_embedding = Embedding(num_items, embedding_size, embeddings_regularizer=regularizer)(item_input)
    item_vector = Flatten()(item_embedding)
    item_vector = Dropout(dropout_rate)(item_vector)

    # Brand input and embedding
    brand_input = Input(shape=(1,))
    brand_embedding = Embedding(num_brands, embedding_size, embeddings_regularizer=regularizer)(brand_input)
    brand_vector = Flatten()(brand_embedding)
    brand_vector = Dropout(dropout_rate)(brand_vector)

    # Gender input and embedding
    gender_input = Input(shape=(1,))
    gender_embedding = Embedding(num_genders, embedding_size, embeddings_regularizer=regularizer)(gender_input)
    gender_vector = Flatten()(gender_embedding)
    gender_vector = Dropout(dropout_rate)(gender_vector)

    # Combine item and brand vectors
    combined_item_brand_vector = Concatenate()([item_vector, brand_vector])

    combined_with_gender_vector = Concatenate()([combined_item_brand_vector, gender_vector])

    combined_vector = Concatenate()([user_vector, combined_with_gender_vector])
    
    output = Dense(1, activation='sigmoid')(combined_vector)

    model = Model(inputs=[user_input, item_input, brand_input, gender_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

    return model


@app.route('/recommend', methods=['POST'])
def recommend():
    global model
    data = request.get_json()
    user_id = data['user_id']
    top_n = data.get('top_n', 25)
    gender = data['gender']

    if user_id not in usernames:
        raise ValueError(f"User {user_id} not found in the data.")

    user_index = usernames[user_id]

    user_embedding_layer = get_embedding_layer(model, 0)
    user_vector = user_embedding_layer.get_weights()[0][user_index]

    item_embedding_layer = get_embedding_layer(model, 1)
    item_vectors = item_embedding_layer.get_weights()[0]

    scores = np.dot(item_vectors, user_vector)

    recommended_item_indices = np.argsort(scores)[::-1][:top_n]

    recommended_items = []

    for i in recommended_item_indices:
        if (len(recommended_items) == top_n):
            break
        if clothing_data[i]['gender'] == gender or clothing_data[i]['gender'] == 'Unisex':
            recommended_items.append(clothing_data[i])
        
        

    return jsonify(recommended_items)

# Route to update the model with new data
@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    user = data['user']

    user_ids = []
    item_ids = []
    brand_ids = []
    gender_ids = []
    labels = []
    global model
    # Update usernames and get user_id
    user_id = usernames.get(user["id"])
    if user_id is None:
        user_id = len(usernames)
        usernames[user["id"]] = user_id
        with open('./data/usernames.json', 'w') as f:
            json.dump(usernames, f)

    for item in user["liked_items"]:
        item_id = clothes.get(item["productID"])
        if item_id is None:
            item_id = len(clothes)
            clothes[item["productID"]] = item_id

        brand_id = brands.get(item["brand"])
        if brand_id is None:
            brand_id = len(brands)
            brands[item["brand"]] = brand_id

        user_ids.append(user_id)
        item_ids.append(item_id)
        brand_ids.append(brand_id)
        labels.append(1)

        gender_id = genders.get(user['gender'])
        if gender_id is None:
            gender_id = len(genders)
            genders[user['gender']] = gender_id
        gender_ids.append(gender_id)

    for item in user["disliked_items"]:
        item_id = clothes.get(item["productID"])
        if item_id is None:
            item_id = len(clothes)
            clothes[item["productID"]] = item_id

        brand_id = brands.get(item["brand"])
        if brand_id is None:
            brand_id = len(brands)
            brands[item["brand"]] = brand_id

        user_ids.append(user_id)
        item_ids.append(item_id)
        brand_ids.append(brand_id)
        labels.append(0)

        gender_id = genders.get(user['gender'])
        if gender_id is None:
            gender_id = len(genders)
            genders[user['gender']] = gender_id
        gender_ids.append(gender_id)

    user_ids_array = np.array(user_ids)
    item_ids_array = np.array(item_ids)
    brand_ids_array = np.array(brand_ids)
    gender_ids_array = np.array(gender_ids)
    labels_array = np.array(labels)

    # Check if the arrays are not empty
    if user_ids_array.size == 0 or item_ids_array.size == 0 or labels_array.size == 0:
        return jsonify({"error": "No data to update"}), 400

    # Gather all old and new entities
    num_old_users = get_embedding_layer(model, 0).input_dim
    num_old_items = get_embedding_layer(model, 1).input_dim
    num_old_brands = get_embedding_layer(model, 2).input_dim
    num_old_genders = get_embedding_layer(model, 3).input_dim

    num_new_users = len(usernames)
    num_new_items = len(clothes)
    num_new_brands = len(brands)
    num_new_genders = len(genders)

    # Total number of entities
    num_users = max(num_old_users, num_new_users)
    num_items = max(num_old_items, num_new_items)
    num_brands = max(num_old_brands, num_new_brands)
    num_genders = max(num_old_genders, num_new_genders)

    # Debugging: Print embedding sizes
    print(f"num_old_users: {num_old_users}, num_new_users: {num_new_users}, num_users: {num_users}")
    print(f"num_old_items: {num_old_items}, num_new_items: {num_new_items}, num_items: {num_items}")
    print(f"num_old_brands: {num_old_brands}, num_new_brands: {num_new_brands}, num_brands: {num_brands}")
    print(f"num_old_genders: {num_old_genders}, num_new_genders: {num_new_genders}, num_genders: {num_genders}")

    # Create a new model with updated embedding sizes
    new_model = create_model(num_users, num_items, num_brands, num_genders)

    for i in range(4):  # Assuming 4 embedding layers: user, item, brand, gender
        old_weights = get_embedding_layer(model, i).get_weights()[0]
        new_weights = get_embedding_layer(new_model, i).get_weights()[0]

        # If old weights are larger, resize new_weights
        if old_weights.shape[0] > new_weights.shape[0]:
            new_weights = np.resize(new_weights, old_weights.shape)

        new_weights[:old_weights.shape[0], :] = old_weights
        get_embedding_layer(new_model, i).set_weights([new_weights])

    # Incrementally train the new model
    new_model.fit([user_ids_array, item_ids_array, brand_ids_array, gender_ids_array], labels_array, epochs=1, verbose=1)

    # Save the updated model
    new_model.save(model_path)

    model = new_model
    
    return jsonify({"message": "Model updated successfully!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 
