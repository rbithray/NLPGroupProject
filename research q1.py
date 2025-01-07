#%%
import numpy as np
from bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA


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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df = 10)
# Initialize BERTopic model
model = BERTopic(embedding_model=embedding_model, vectorizer_model=vectorizer_model)

# Fit the model
topics, probs = model.fit_transform(comments['body'])

# Visualize topics

#%%

# Define the list of keywords to search for
keywords = ["regulations", "regulation", "regulatory","law","laws", 
            "government", "policy", "policies","politics","rules", 
            "control","management","governance","conduct","statute",
            "ordenance","order","monitoring","supervision","restriction",
            "oversight","limitations","guidelines","compliance","enforcement",
            "authority","bylaw","license","standards","mandate","police",
            "judge","court","fairness","ethics"]

# Create an empty set to store unique indices
all_indices = set()

# Iterate over each keyword to find similar topics and their indices
for keyword in keywords:
    similar_topics, similarity = model.find_topics(keyword, top_n=5)
    
    if similar_topics:
        # Get the most similar topic ID for the current keyword
        topic_id = similar_topics[0]

        # Use numpy array indexing or boolean mask for topic matching
        topic_mask = [t == topic_id for t in topics]  # Create a boolean mask
        topic_comments = comments[topic_mask]        # Apply the mask to the DataFrame

        # Add the indices of comments related to this topic to the set
        all_indices.update(topic_comments.index)

        print(f"Keyword '{keyword}' - Topic ID: {topic_id}, Matches: {len(topic_comments)}")
    else:
        print(f"No topics found for keyword '{keyword}'.")

# Convert the set of indices to a sorted list (optional)
all_indices_sorted = sorted(list(all_indices))

# Display the indices of all relevant comments
print(f"Total unique relevant comment indices: {len(all_indices_sorted)}")
print(all_indices_sorted)

import json

# Je lijst met indices
all_indices_sorted = sorted(list(all_indices))

# Opslaan als JSON-bestand
with open('indices.json', 'w') as f:
    json.dump(all_indices_sorted, f)

print("Indices opgeslagen in 'indices.json'")

#%%
model.visualize_topics()
# %%
# Get topic frequencies
topic_frequencies = model.get_topic_info()

# Display the top N topics by frequency
print(topic_frequencies.head(17))

# Display the top words for the most frequent topics
for topic_id in topic_frequencies['Topic'].head(10):
    if topic_id != -1:  # Skip outliers
        print(f"Topic {topic_id}:")
        print(model.get_topic(topic_id))

# %%
print(model.get_topic(6))
# %%
