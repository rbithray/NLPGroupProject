#%%
# Imports and preprocessing of data
from bertopic import BERTopic
from umap import UMAP
import numpy as np
import pandas as pd
from os import path
from sklearn.feature_extraction.text import CountVectorizer

# Random seed to increase reproducibility
np.random.seed(42)

comments_path = 'data/comments.ndjson'
submissions_path = 'data/submissions.ndjson'


comments_df = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)
submission_df = pd.read_json(path.join('data', 'submissions.ndjson'), lines=True)

comments = comments_df[(comments_df['body'] != '[removed]') & (comments_df['body'] != '[deleted]')]
# submissions = pd.concat([submission_df['title'], submission_df[submission_df['selftext']]])


comments['date'] = pd.to_datetime(comments['created_utc'], unit='s').dt.strftime("%m, %d, %Y")
submission_df['date'] = pd.to_datetime(submission_df['created_utc'], unit='s').dt.strftime("%m, %d, %Y")

text = pd.concat([comments['body'], submission_df['title']], axis=0).reset_index(drop=True)
timestamps = pd.concat([comments['date'], submission_df['date']], axis=0).reset_index(drop=True)
timestamps = pd.to_datetime(timestamps, format='%m, %d, %Y')
#%%
print(submission_df['date'].shape)
print(comments['date'].shape)
print(text.shape)
print(timestamps.shape)

# print(timestamps.iloc[-10:])
print(timestamps.min())
print(timestamps.max())


#%%
# Initializing and fitting the model
umap_model = UMAP(random_state=42)
vectorizer_model = CountVectorizer(ngram_range=(1, 2), min_df = 10)

model = BERTopic(vectorizer_model=vectorizer_model, umap_model=umap_model)
topics, probs = model.fit_transform(text.tolist())

#%%
# Generate topics over time
topics_over_time = model.topics_over_time(
    text.tolist(),
    timestamps.tolist(),
    nr_bins=7
)
print(topics_over_time.head())
#%%

custom_labels = {
    0: 'Chatbots and Conversations',
    1: 'AI-Generated Art and Creativity',
    2: 'AI and Music Creation',
    3: 'ChatGPT and Information Retrieval',
    4: 'Energy Comparisons and Diversity'
}
model.set_topic_labels(custom_labels)

model.visualize_topics_over_time(
    topics_over_time,
    custom_labels=True,
    top_n_topics=5
)
# %%
