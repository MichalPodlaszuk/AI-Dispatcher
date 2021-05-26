import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, remove_stop
from speech.speech_handling import speech_recognizer, speak

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../data/data_clean/intents.json', 'r') as f, open('../data/data_clean/levels.json') as g:
    intents = json.load(f)
    levels = json.load(g)

FILE = '../data/model_data/data.pth'
FILE_LEVELS = '../data/model_data/data_levels.pth'
data = torch.load(FILE)
data_levels = torch.load(FILE_LEVELS)

output_size = data['output_size']
input_size = data['input_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

output_size_levels = data_levels['output_size']
input_size_levels = data_levels['input_size']
hidden_size_levels = data_levels['hidden_size']
all_words_levels = data_levels['all_words']
tags_levels = data_levels['tags']
model_state_levels = data_levels['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

model_levels = NeuralNet(input_size_levels, hidden_size_levels, output_size_levels).to(device)
model_levels.load_state_dict(model_state_levels)
model_levels.eval()

bot_name = 'AI-Dispatcher'
print(f'{bot_name}: Hello, this is 911 dispatcher, what is your emergency *this is a test version, say quit to stop*')
last_response = ['Hello this is 911 dispatcher, what is your emergency']
previous_tag = []
location = []
emergency = []


def classify(sentence, all_words, model, tags, file):
    sentence_without_stop_words = remove_stop(sentence)
    X = bag_of_words(sentence_without_stop_words, all_words)
    X_reshaped = X.reshape(1, X.shape[0])
    X_tensor = torch.from_numpy(X_reshaped).to(device)

    output = model(X_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.6:
        for intent in file['intents']:
            if tag == intent['tag']:
                response = random.choice(intent["responses"])
                if file == intents:
                    print(f'{bot_name}: {response}, detected intent: {tag}')
                else:
                    print(f'{response}, detected tag: {tag}')
    else:
        print(f"{bot_name}: Please repeat your message more clearly")
        response = 'Please repeat your message more clearly'
        tag = 'unkn'
    return tag, response


while True:
    sentence = speech_recognizer(chat=True, previous=last_response[-1])
    if sentence == 'quit':
        break

    tag, response = classify(sentence, all_words, model, tags, intents)
    last_response.append(response)
    previous_tag = tag
    if tag == 'emergency':
        print(f'detected tag:{tag}, starting emergency classification')
        level, action = classify(sentence, all_words_levels, model_levels, tags_levels, levels)
        print(f'emergency level detected: {level}')
        emergency.extend((sentence, level, action))
        if level == '1' or level == '2':
            print('redirecting')
            speak('your call will be redirected please stay on the line')
            break
        else:
            print('asking for location')
            last_response.append('Okay, tell me where you are')
    if tag == 'location' and emergency[1] == '3' or emergency[1] == '4':
        location.append(sentence)
        last_response.append('Okay tell me again exactly what happened')
    if tag == 'random' and len(location) == 0:
        last_response.append(random.choice(['I need to know your location please', 'try to find out where exactly you are']))
        