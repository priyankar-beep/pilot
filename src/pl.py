import tensorflow as tf
from transformers import TFViTModel, TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder

# Define the PlantDataGenerator class
class PlantDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, plant_labels, disease_labels, descriptions, batch_size=32, image_size=(224, 224)):
        self.image_paths = image_paths
        self.plant_labels = plant_labels
        self.disease_labels = disease_labels
        self.descriptions = descriptions
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_samples = len(image_paths)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.feature_extractor = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        batch_image_paths = [self.image_paths[i] for i in batch_indices]
        batch_plant_labels = [self.plant_labels[i] for i in batch_indices]
        batch_disease_labels = [self.disease_labels[i] for i in batch_indices]
        images = np.array([self.load_image(p) for p in batch_image_paths])
        texts = [self.descriptions[d] for d in batch_disease_labels]
        tokenized_texts = self._preprocess_texts(texts)
        return [images, tokenized_texts], [batch_plant_labels, batch_disease_labels]

    def on_epoch_end(self):
        data = list(zip(self.image_paths, self.plant_labels, self.disease_labels))
        np.random.shuffle(data)
        self.image_paths, self.plant_labels, self.disease_labels = zip(*data)
    
    def load_image(self, image_path):
        image = load_img(image_path, target_size=self.image_size)
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        return image_array
    
    def _preprocess_texts(self, texts):
        encoded_texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors='np')
        return (encoded_texts['input_ids'], encoded_texts['attention_mask'])

# Function to get image paths and labels
def get_image_paths_and_labels(dataset_dir):
    image_paths = []
    plant_labels = []
    disease_labels = []
    for folder_name in os.listdir(dataset_dir):
        if '__' not in folder_name:
            continue
        plant_name, disease_status = folder_name.split('__')
        folder_path = os.path.join(dataset_dir, folder_name)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image_paths.append(img_path)
            plant_labels.append(plant_name)
            disease_labels.append(disease_status)
    return image_paths, plant_labels, disease_labels

# Function to load disease descriptions
def load_descriptions(description_file):
    descriptions = {}
    with open(description_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ':' in line:
                disease_name, description = line.strip().split(':', 1)
                descriptions[disease_name.strip()] = description.strip()
    return descriptions

# Function to encode labels
def encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels), le.classes_

# Main processing
dataset_dir = '/home/hubble/Downloads/plnat_disease_papers/plant_village/data'
description_file = '/home/hubble/work/serenade/src/captions.txt'


# Get image paths and labels
image_paths, plant_labels, disease_labels = get_image_paths_and_labels(dataset_dir)
# Load descriptions
descriptions = load_descriptions(description_file)
# Encode labels
encoded_plant_labels, plant_classes = encode_labels(plant_labels)
encoded_disease_labels, disease_classes = encode_labels(disease_labels)

print(f"Number of images: {len(image_paths)}")
print(f"Sample image path: {image_paths[0]}")
print(f"Sample plant label: {plant_labels[0]}")
print(f"Sample disease label: {disease_labels[0]}")
print(f"Number of unique plants: {len(plant_classes)}")
print(f"Number of unique diseases: {len(disease_classes)}")
print(f"Descriptions: {descriptions}")

# Split the dataset
train_image_paths, val_image_paths, train_plant_labels, val_plant_labels, train_disease_labels, val_disease_labels = train_test_split(
    image_paths, encoded_plant_labels, encoded_disease_labels, test_size=0.2, random_state=42
)

# Create data generators
train_generator = PlantDataGenerator(
    train_image_paths, train_plant_labels, train_disease_labels, descriptions, batch_size=32
)
val_generator = PlantDataGenerator(
    val_image_paths, val_plant_labels, val_disease_labels, descriptions, batch_size=32
)

from transformers import ViTFeatureExtractor, TFViTModel
from datasets import load_dataset
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
def create_model(num_plants, num_diseases):
    # Image branch with TFViTModel from Hugging Face
    # vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = TFViTModel.from_pretrained('google/vit-base-patch16-224-in21k', config=vit_config)
        
    
    # Define a function to convert image input to tensor
    @tf.function
    def preprocess_image(image_input):
        return tf.convert_to_tensor(image_input, dtype=tf.float32)
    image_input = Input(shape=(224, 224, 3), name='image_input')
    preprocessed_image = preprocess_image(image_input)
    # Image input
    
    dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    image_tensor = tf.convert_to_tensor(dummy_image)
    vit_output = vit_model(image_input, return_dict=True)



    # Convert KerasTensor to TensorFlow tensor
    image_tensor = Lambda(lambda x: tf.convert_to_tensor(x))(image_input)
    print(image_tensor.shape)
    
    vit_output = vit_model(image_tensor, return_dict=True)

    # Extract features using TFViTModel
    vit_features = vit_model(image_tensor).last_hidden_state[:, 0]  # Use the CLS token representation

    # Text branch with BERT
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    text_input_ids = Input(shape=(None,), dtype=tf.int32, name='input_ids')
    text_attention_mask = Input(shape=(None,), dtype=tf.int32, name='attention_mask')
    
    # Extract features using BERT
    bert_features = bert_model([text_input_ids, text_attention_mask]).last_hidden_state[:, 0]  # Use the CLS token representation
    
    # Combine features
    combined_features = Concatenate()([vit_features, bert_features])
    
    # Plant name prediction branch
    plant_output = Dense(num_plants, activation='softmax', name='plant_output')(vit_features)
    
    # Disease prediction branch
    disease_output = Dense(num_diseases, activation='softmax', name='disease_output')(combined_features)
    
    # Create the model
    model = tf.keras.Model(
        inputs=[image_input, text_input_ids, text_attention_mask], 
        outputs=[plant_output, disease_output]
    )
    
    return model

# Define number of classes (adjust these to match your data)
num_plants = len(set(encoded_plant_labels))  # Number of unique plant labels
num_diseases = len(set(encoded_disease_labels))  # Number of unique disease labels

# Create and compile the model
model = create_model(num_plants, num_diseases)
# model.compile(optimizer='adam', 
#               loss={'plant_output': 'categorical_crossentropy', 'disease_output': 'categorical_crossentropy'}, 
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# # Save the model
# model.save('plant_disease_model.h5')

