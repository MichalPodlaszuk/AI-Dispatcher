import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

output_size = data['output_size']
input_size = data['input_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'AI-Dispatcher'
print(f'{bot_name}: Hello, this is 911 dispatcher, what is your emergency')
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    tokenized_sentence = tokenize(sentence)
    X = bag_of_words(tokenized_sentence, all_words)
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
                print(f'{bot_name}: {random.choice(intent["responses"])}, detected intent: {tag}')
    else:
        print(f"{bot_name}: Please repeat your message more clearly")