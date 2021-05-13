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

[x] Done

### Getting the data

- The biggest problem here is lack of any datasets, there's no labeled data and one thing that comes to my mind is to get some examples of different cases and try to find similarities among them that could help me to classify them. Possible source of transcripts is FOIA requests.

- I didn't have time to wait for public transcripts and recordings so I got the data from existing sites that have PDFs and downloaded them through VPN due to legal reasons.

[x] Done

### Preprocessing the data

- The PDFs I got were all protected view, I will now have to figure out the way to extract valuable data from them.
  
- I wasn't planning on using Computer Vision in my project but it turns out I am forced to. I will have to use Google OCR engine and openCV to extract text from those pdfs.

[x] Done

### Cleaning the data

- ...

[ ] Done