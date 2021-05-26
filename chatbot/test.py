from sentence_transformers import SentenceTransformer, util
import numpy as np
from functools import lru_cache
import time



def semantic_similarity(sentence1, sentence2):
    t1 = time.time()
    
    model = SentenceTransformer('stsb-mpnet-base-v2')
    
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    print("Sentence 1:", sentence1)
    print("Sentence 2:", sentence2)
    print("Similarity score:", cosine_scores.item())

    t2 = time.time()
    t_result = t2-t1
    print(t_result)
semantic_similarity('abdominal pain', 'your mom')