import pymongo
import certifi
import requests
import os
import json

from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client



from keras.models import Model
from keras.layers import Embedding, Flatten, Dot, Input, Activation, Concatenate, Dense
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.initializers import RandomNormal

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score

import numpy as np
import pandas as pd




class FetchData:
    def __init__(self) -> None:
        
        load_dotenv(find_dotenv())
        mongodb_url = os.getenv("MONGODB_URL")
        self.mongodb_client = pymongo.MongoClient(mongodb_url, tlsCAFile=certifi.where())
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase_client: Client = create_client(supabase_url, supabase_key)

        self.request_client = requests.Session()
        self.custom_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }   
    
    def clothing_data(self) -> list:
        print("Fetching data from MongoDB")
        
        db = self.mongodb_client['my_data']
        clothing_data = []
        for i in db.list_collection_names():
            collection = db[i]
            for item in collection.find():
                    clothing_data.extend(item['clothes_data'])
        
        return clothing_data

    def user_data(self) -> list:
        print("Fetching data from Supabase")
        
        user_data = self.supabase_client.from_('users').select('id','name','liked_items', 'disliked_items', 'gender').execute()
        
        for i in user_data.data:
            i['liked_items'] = [
                {
                    "name": j['name'],
                    "productID": j['productID'],
                    "brand": j['brand']
                }
                for j in i['liked_items']
            ]
            i['disliked_items'] = [
                {
                    "name": j['name'],
                    "productID": j['productID'],
                    "brand": j['brand']
                }
                for j in i['disliked_items']
            ]
        return user_data.data

class SwipeStyle_AI:
    def __init__(self, clothing_data: list, user_data: list) -> None:


        self.usernames = {user["id"]: i for i, user in enumerate(user_data)}
        self.clothes = {item["productID"]: i for i, item in enumerate(clothing_data)}
        
        unique_brands = list(set(item["brand"] for item in clothing_data))  
        self.brands = {brand: i for i, brand in enumerate(unique_brands)}

        unique_genders = list(set(item["gender"] for item in clothing_data))
        self.genders = {gender: i for i, gender in enumerate(unique_genders)}
        
        self.num_users = len(self.usernames)
        self.num_clothes = len(self.clothes)
        self.num_brands = len(self.brands)
        self.num_genders = len(self.genders)

        #Save data for API Use
        with open('./data/usernames.json', 'w') as f:
            json.dump(self.usernames, f)
        
        with open('./data/clothes.json', 'w') as f:
            json.dump(self.clothes, f)

        with open('./data/brands.json', 'w') as f:
            json.dump(self.brands, f)

        with open('./data/genders.json', 'w') as f:
            json.dump(self.genders, f)



        embedding_size = 50
        regularizer = l2(1e-4)

        user_input = Input(shape=(1,))
        user_embedding = Embedding(self.num_users, embedding_size, 
                                   embeddings_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                   embeddings_regularizer=regularizer)(user_input)
        user_vector = Flatten()(user_embedding)

        item_input = Input(shape=(1,))
        item_embedding = Embedding(self.num_clothes, embedding_size, 
                                   embeddings_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                   embeddings_regularizer=regularizer)(item_input)
        item_vector = Flatten()(item_embedding)

        brand_input = Input(shape=(1,))
        brand_embedding = Embedding(self.num_brands, embedding_size, 
                                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                                    embeddings_regularizer=regularizer)(brand_input)
        brand_vector = Flatten()(brand_embedding)

        gender_input = Input(shape=(1,))
        gender_embedding = Embedding(self.num_genders, embedding_size,
                                     embeddings_initializer=RandomNormal(mean=0.0, stddev=0.5),
                                     embeddings_regularizer=regularizer)(gender_input)
        gender_vector = Flatten()(gender_embedding)
        
        

        # Combine item and brand vectors
        combined_item_brand_vector = Concatenate()([item_vector, brand_vector])

        combined_with_gender_vector = Concatenate()([combined_item_brand_vector, gender_vector])

        combined_vector = Concatenate()([user_vector, combined_with_gender_vector])
        
        output = Dense(1, activation='sigmoid')(combined_vector)

        self.model = Model(inputs=[user_input, item_input, brand_input, gender_input], outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

        

    def prepare_training_data(self, user_data: list):
        user_ids = []
        item_ids = []
        brand_ids = []
        labels = []
        genders = []

        for user in user_data:
            user_id = self.usernames[user["id"]]
            for item in user["liked_items"]:
                item_id = self.clothes[item["productID"]]
                brand_id = self.brands[item["brand"]]
                user_ids.append(user_id)
                item_ids.append(item_id)
                brand_ids.append(brand_id)
                labels.append(1)
                try:
                    genders.append(self.genders[item['gender']])
                except:
                    genders.append(self.genders[user['gender']])
                
            for item in user["disliked_items"]:
                item_id = self.clothes[item["productID"]]
                brand_id = self.brands[item["brand"]]
                user_ids.append(user_id)
                item_ids.append(item_id)
                brand_ids.append(brand_id)
                labels.append(0)
                try:
                    genders.append(self.genders[item['gender']])
                except:
                    genders.append(self.genders[user['gender']])
                

        user_ids_array = np.array(user_ids)
        item_ids_array = np.array(item_ids)
        brand_ids_array = np.array(brand_ids)
        genders_array = np.array(genders)
        labels = np.array(labels)
        
        return user_ids_array, item_ids_array, brand_ids_array, genders_array, labels

    def train_model(self, user_data: list, epochs=100) -> None:
        user_ids_array, item_ids_array, brand_ids_array, genders_array, labels = self.prepare_training_data(user_data)

        X_train, X_test, y_train, y_test = train_test_split(
            np.column_stack((user_ids_array, item_ids_array, brand_ids_array, genders_array)), labels, test_size=0.25
        )

        history = self.model.fit(
            [X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3]], y_train,
            epochs=epochs,
            verbose=1,
            validation_split=0.25
        )

        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        test_loss = self.model.evaluate([X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3]], y_test, verbose=1)
        print(f'Test Loss: {test_loss}')

        y_pred_scores = self.model.predict([X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3]])
        optimal_threshold = 0.5  
        y_pred_binary = (y_pred_scores > optimal_threshold).astype(int)
        y_true_binary = y_test

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=1)
        recall = recall_score(y_true_binary, y_pred_binary)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')

    def recommend_items(self, user_id: str, top_n=3) -> list:
        if user_id not in self.usernames:
            raise ValueError(f"User {user_id} not found in the data.")

        user_index = self.usernames[user_id]

        # Get the user vector from the user embedding layer
        user_vector = self.model.get_layer('embedding').get_weights()[0][user_index]

        # Get all item vectors from the item embedding layer
        item_vectors = self.model.get_layer('embedding_1').get_weights()[0]

        scores = np.dot(item_vectors, user_vector)

        # Get top N items with the highest scores
        recommended_item_indices = np.argsort(scores)[::-1][:top_n]

        # Map indices back to item IDs
        recommended_items = [list(self.clothes.keys())[i] for i in recommended_item_indices]

        return recommended_items

    def save_model(self, model_name: str) -> None:
        self.model.save(model_name)


if __name__ == "__main__":

    fetch = FetchData()
    clothing_data = fetch.clothing_data()
    user_data = fetch.user_data()

    ss_ai = SwipeStyle_AI(clothing_data, user_data)
    ss_ai.train_model(user_data, epochs=100)

    #Get recommendations
    recommendations = ss_ai.recommend_items("82689450-fd93-4a65-b813-e70a2a51f557")
    print(f"Recommendations for user: {recommendations}")

    #Save model
    ss_ai.save_model("swipestyle-ai.keras")