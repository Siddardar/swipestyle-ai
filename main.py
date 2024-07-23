import pymongo
import certifi
import requests
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from tensorflow import keras

from keras.models import Model
from keras.layers import Embedding, Flatten, Dot, Input, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.initializers import RandomNormal

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score

import numpy as np
import pandas as pd


class PreProcessData:
    def __init__(self) -> None:
        
        load_dotenv('env/.env')
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
    
    def fetch_clothing_data(self) -> list:
        print("Fetching data from MongoDB")
        
        db = self.mongodb_client['my_data']
        collection = db['brands_women']

        clothing_data = []
        for item in collection.find():
            data = db[f'{item["brand"].lower()}_women_tops']
            for i in data.find():
                clothing_data.extend(i['clothes_data'])
        
        return clothing_data

    def fetch_user_data(self) -> list:
        print("Fetching data from Supabase")
        
        user_data = self.supabase_client.from_('users').select('id','name','liked_items', 'disliked_items').execute()
        return user_data.data

    def data_cleaning(self, data: list) -> list:
        print("Cleaning data")
        
        for i in data:
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
        return data

class SwipeStyle_AI:
    def __init__(self, clothing_data: list, user_data: list) -> None:
        self.usernames = {user["id"]: i for i, user in enumerate(user_data)}
        self.clothes = {item["productID"]: i for i, item in enumerate(clothing_data)}
        
        self.num_users = len(self.usernames)
        self.num_clothes = len(self.clothes)

        embedding_size = 50
        regularizer = l2(1e-4)  # Reduced regularization

        user_input = Input(shape=(1,))
        user_embedding = Embedding(self.num_users, embedding_size, embeddings_initializer=RandomNormal(mean=0.0, stddev=0.05), embeddings_regularizer=regularizer)(user_input)
        user_vector = Flatten()(user_embedding)

        item_input = Input(shape=(1,))
        item_embedding = Embedding(self.num_clothes, embedding_size, embeddings_initializer=RandomNormal(mean=0.0, stddev=0.05), embeddings_regularizer=regularizer)(item_input)
        item_vector = Flatten()(item_embedding)

        dot_product = Dot(axes=1)([user_vector, item_vector])
        output = Activation('sigmoid')(dot_product)  

        self.model = Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')  # Adjusted learning rate and loss function

    def prepare_training_data(self, user_data: list):
        user_ids = []
        item_ids = []
        labels = []

        for user in user_data:
            user_id = self.usernames[user["id"]]
            for item in user["liked_items"]:
                item_id = self.clothes[item["productID"]]
                user_ids.append(user_id)
                item_ids.append(item_id)
                labels.append(1)
            for item in user["disliked_items"]:
                item_id = self.clothes[item["productID"]]
                user_ids.append(user_id)
                item_ids.append(item_id)
                labels.append(0)  # Use 0 for dislikes to match sigmoid output

        user_ids_array = np.array(user_ids)
        item_ids_array = np.array(item_ids)
        labels = np.array(labels)
        
        return user_ids_array, item_ids_array, labels

    def train_model(self, user_data: list, epochs=100) -> None:
        user_ids_array, item_ids_array, labels = self.prepare_training_data(user_data)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            np.column_stack((user_ids_array, item_ids_array)), labels, test_size=0.25
        )

        history = self.model.fit(
            [X_train[:, 0], X_train[:, 1]], y_train,
            epochs=epochs,
            verbose=1,
            validation_split=0.25
        )

        # Plot training history
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Evaluate the model
        test_loss = self.model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, verbose=1)
        print(f'Test Loss: {test_loss}')

        # Predict and calculate metrics
        y_pred_scores = self.model.predict([X_test[:, 0], X_test[:, 1]])
        print(f"Predicted scores range: {y_pred_scores.min()} to {y_pred_scores.max()}")
        print(f"Predicted scores mean: {y_pred_scores.mean()}")

        # Experiment with different thresholds
        optimal_threshold = 0.5  
        y_pred_binary = (y_pred_scores > optimal_threshold).astype(int)
        y_true_binary = y_test

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=1)
        recall = recall_score(y_true_binary, y_pred_binary)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')

        # Print label distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")

    def recommend_items(self, user_name: str, top_n=3) -> list:
        user_id = self.usernames[user_name]
        user_vector = self.model.layers[2].get_weights()[0][user_id]
        item_vectors = self.model.layers[3].get_weights()[0]

        scores = np.dot(item_vectors, user_vector)
        recommended_item_ids = np.argsort(scores)[::-1][:top_n]
        recommended_items = [list(self.clothes.keys())[list(self.clothes.values()).index(item_id)] for item_id in recommended_item_ids]

        return recommended_items

    def save_model(self, model_name: str) -> None:
        self.model.save(model_name)
    


if __name__ == "__main__":
    pp_data = PreProcessData()
    clothing_data = pp_data.fetch_clothing_data()
    raw_data = pp_data.fetch_user_data()
    user_data = pp_data.data_cleaning(raw_data)
    
    ss_ai = SwipeStyle_AI(clothing_data, user_data)
    ss_ai.train_model(user_data, epochs=100)

    #Get recommendations
    recommendations = ss_ai.recommend_items("82689450-fd93-4a65-b813-e70a2a51f557")
    print(f"Recommendations for user: {recommendations}")

    #Save model
    ss_ai.save_model("swipestyle-ai.keras")