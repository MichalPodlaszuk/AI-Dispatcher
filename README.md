# AI-Dispatcher
Capstone project for strive school
## Quick introduction
This is a repo that I will be updating throughout the next couple weeks. It's basically a somewhat big project that I'm working on alone. I'm trying to do something that matters and I think this is a really good idea.
## What it does?
This will be a chatbot that performs a conversation and determines whether the call is urgent or not.
## Why?
Dispatch centers tend to be overloaded with calls even though they are still hiring. This creates a huge problem sometimes because people can actually lose life due to the fact that call center couldn't answer the phone. AI could be really helpful here since it is automated.
## Walkthrough

### Coming up with the idea

- The primary idea behind the project is to let AI tell if the call is important or can wait and either place it in a queue to a real person or send police/ambulance/firefighters directly where the emergency is, or both.

### Getting the data

- The biggest problem here is lack of any datasets, there's no labeled data and one thing that comes to my mind is to get some examples of different cases and try to find similarities among them that could help me to classify them. Possible source of transcripts is FOIA requests.

- I didn't have time to wait for public transcripts and recordings so I got the data from existing sites that have PDFs and downloaded them through VPN due to legal reasons.

### Preprocessing the data

- The PDFs I got were all protected view, I will now have to figure out the way to extract valuable data from them.
  
- I wasn't planning on using Computer Vision in my project but it turns out I am forced to. I will have to use Google OCR engine and openCV to extract text from those pdfs.

### Cleaning the data

- There was quite a lot of useless text inside the data, I extracted manually only those parts and lines that I was interested in which was lines of the user.

- I then divided the data into 4 classes, emergency: location, description and random, I think names speak for themselves.

### Creating the chatbot

- The chatbot that I created captures your voice, uses speech to text translation, classifies the text based on previous 4 classes and speaks out an appropriate answer.

- To narrow down the amount of words the bot needs to preprocess I removed unnecessary words like stopwords and punctuation, this makes the model more robust.

### Augmenting the data

- The amount of data I was able to get was not going to lie quite small, I had to augment it, and I used 2 techniques, back translation and something called thesaurus, back translation is translating the sentence to another language, here I used chinese, and translating it back to english, this keeps the sense of the sentence but changes words in it and the order of those. Thesaurus on the other hand is creating more data points by substituting original words with their synonyms.

### Creating classifier for emergency level

- So the main functionality of that project is that it classifies the type of emergency between 5 levels from 1 to 5, 1 being the most urgent and 5 being the least.

- I took the emergencies from original dataset and created another dataset in which I divided the emergencies between before mentioned levels.

- I then created a classifier that is actually able to classify the emergency itself which I then implemented into my chatbot conversation loop, the classifier gets activated whenever the sentence is classifier as an emergency.

## Overview

In my opinion the project is done, and it works quite well once loaded. I had a lot of fun with it and I'm quite proud with the idea, the only thing that troubled me the most was the amount of data I got which is really poor. I think if I had enough data to work with I could a bit more slowly but surely make the model more robust and more functional. If I have some more data and any more ideas in the future I will try to implement that for sure because I really like this idea and I think it could be really useful with more data and work put into it but for that I would need to be allowed to get more data which, sadly, I'm not at this point.

Anyway if you've come this far and have some thoughts or ideas feel free to contact me on [LinkedIn](https://www.linkedin.com/in/micha%C5%82-podlaszuk-612a99200/)