import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words, remove_stop
from speech.speech_handling import speech_recognizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../data/data_clean/intents.json', 'r') as f, open('../data/data_clean/levels.json') as g:
    intents = json.load(f)
    levels = json.load(g)

FILE = '../data/model_data/data.pth'
FILE_LEVELS = '..data/model_data/data_levels.pth'
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
last_response = ['Hello, this is 911 dispatcher, what is your emergency']
while True:
    sentence = speech_recognizer(chat=True, previous=last_response[-1])
    if sentence == 'quit':
        break

    tokenized_sentence = tokenize(sentence)
    sentence_without_stop_words = remove_stop(tokenized_sentence)
    X = bag_of_words(sentence_without_stop_words, all_words)
    X_reshaped = X.reshape(1, X.shape[0])
    X_tensor = torch.from_numpy(X_reshaped).to(device)

    output = model(X_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.6:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent["responses"])
                print(f'{bot_name}: {response}, detected intent: {tag}')
                last_response.append(response)
                if tag == 'emergency':
                    output_level = model_levels(X_tensor)
                    _, predicted_levels = torch.max(output_level, dim=1)
                    tag_levels = tags_levels[predicted_levels.item()]

                    probs_levels = torch.softmax(output_level, dim=1)
                    prob_levels = probs_levels[0][predicted_levels.item()]
                    if prob.item() > 0.6:
                        for level in levels['intents']:
                            if tag == levels['tag']:
                                action = random.choice(levels['responses'])
                                print(f'action: {action}, detected level: {tag}')
    else:
        print(f"{bot_name}: Please repeat your message more clearly")