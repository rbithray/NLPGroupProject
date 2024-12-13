#%%
import numpy as np
from bertopic import BERTopic
import torch
import pandas as pd
import json
from os import path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sentence_transformers import SentenceTransformer


#%%
# File paths
comments_path = 'data/comments.ndjson'
submissions_path = 'data/submissions.ndjson'

# Load data
comments = pd.read_json(comments_path, lines=True)
submissions = pd.read_json(submissions_path, lines=True)

removed_markers = ["[removed]", "[deleted]"]
comments = comments[~comments['body'].isin(removed_markers)]
comments = comments[comments['body'].notnull()]  # Remove null comments


# Apply preprocessing to the filtered comments

# Use SentenceTransformers for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize BERTopic model
model = BERTopic(embedding_model=embedding_model)

# Fit the model
topics, probs = model.fit_transform(comments['body'])

# Visualize topics

#%%
# Find similar topics related to "regulation"
similar_topics, similarity = model.find_topics("regulation", top_n=5)
print(similar_topics)
# Retrieve the most similar topic
if similar_topics:
    regulation_topic = model.get_topic(similar_topics[0])
    regulation_topic_id = similar_topics[0]
    print(f"Topic-ID voor 'regulation': {regulation_topic_id}")

    # Voeg de gegenereerde topics toe aan de oorspronkelijke DataFrame
    comments['topic'] = topics

    # Filter opmerkingen die zijn toegewezen aan het relevante topic-ID
    regulation_comments = comments[comments['topic'] == regulation_topic_id]

    # Print of verwerk de opmerkingen
    print(f"Aantal opmerkingen over 'regulation': {len(regulation_comments)}")
    print(regulation_comments['body'])
else:
    print("No topics found related to 'politics'.")
model.visualize_topics()
# %%
