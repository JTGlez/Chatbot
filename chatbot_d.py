import random  # Selección de respuesta aleatoria.
import json  # Corpus.
import pickle  # Serialización.
import numpy as np
import discord  # Discord

import nltk  # Natural Language Toolkit.
from nltk.stem import WordNetLemmatizer  # Lematización.
import tensorflow

from tensorflow.keras.models import load_model

# Importación de modelo de chatbot y los archivos generados en el proceso de entrenamiento.
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_tienda.h5')

# Limpiador de las entradas: elimina los signos de puntuación, los espacios en blanco, etc.


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# Genera el arreglo de 0s que contiene un 1 si la palabra está en el corpus.


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Función que genera una predicción con la red neuronal, y la clasifica en una clase de respuesta.


def predict_class(sentence):
    # Arreglo bag of words.
    bow = bag_of_words(sentence)

    for i in range(len(bow)):
        # Si todos los valores en el arreglos son 0, entonces no se encontró ninguna palabra.
        # Si un valor es 1, entonces se encontró coincidencia en el corpus y se clasifica. perro
        if bow[i] != 0:
            # Predicción.
            res = model.predict(np.array([bow]))[0]
            ERROR_THRESHOLD = 0.25
            # Selección de la respuesta.
            results = [[i, r]
                       for i, r in enumerate(res) if r > ERROR_THRESHOLD]

            # Búsqueda de la clase con mayor probabilidad.
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                return_list.append(
                    {'intent': classes[r[0]], 'probability': str(r[1])})
            #print("Mensaje ",type(return_list))
            return return_list
    return None

# Función que genera una respuesta a partir de una entrada procesada con predict_class.


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    # Verifica el tag que se asignó a la entrada y busca un respuesta aleatoria de esa categoría.
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            # print(i['tag'])
            break
    return result, tag

# Ciclo while del chatbot. Termina cuando se obtiene una respuesta de la clase 'goodbye'.
#print("¡Bot en línea!")


# Token del bot
TOKEN = "OTgyODA4OTgxOTUxMDkwNzY4.G-d7Xt.ns3-aTo5LQltyr4zPqE8duX4jzjVglbGE8A2pI"
client = discord.Client()
#message = input()


@client.event
async def on_message(message):
    username = str(message.author).split('#')[0]
    user_message = str(message.content)
    channel = str(message.channel.name)
    if message.author == client.user:
        return
    # if message != "":
    if message.content.startswith("$iabot"):
        ints = predict_class(message.content)
        if ints is not None:
            res, tag = get_response(ints, intents)
            await message.channel.send(res)
        else:
            await message.channel.send("Disculpa, no entiendo tu mensaje. :(")
    else:
        await message.channel.send("Escribe algo!")
client.run(TOKEN)
