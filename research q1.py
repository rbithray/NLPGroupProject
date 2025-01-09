#%%
import numpy as np
from bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


#%%
# Random state
np.random.seed(42)
umap_model = UMAP(random_state=42)
# File paths
comments_path = 'data/comments.ndjson'
submissions_path = 'data/submissions.ndjson'

# Load data
comments = pd.read_json(comments_path, lines=True)
submissions = pd.read_json(submissions_path, lines=True)


removed_markers = ["[removed]", "[deleted]"]
comments = comments[~comments['body'].isin(removed_markers)]
comments = comments[comments['body'].notnull()]  # Remove null comments
comments = pd.concat([comments['body'], submissions['title']], axis=0)
#%%
# model
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df = 10)
# Initialize BERTopic model
model = BERTopic(vectorizer_model=vectorizer_model, umap_model=umap_model)

# Fit the model
topics, probs = model.fit_transform(comments)
#%%
#visualisation
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
#taking a closer look at specific topics
print(model.get_topic(15))
# %%
