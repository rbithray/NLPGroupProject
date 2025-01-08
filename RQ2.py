#%%
# Imports and preprocessing of data
from bertopic import BERTopic
import numpy as np
import pandas as pd
from os import path


comments_path = 'data/comments.ndjson'
submissions_path = 'data/submissions.ndjson'


comments_df = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)
submission_df = pd.read_json(path.join('data', 'submissions.ndjson'), lines=True)

comments = comments_df[(comments_df['body'] != '[removed]') & (comments_df['body'] != '[deleted]')]

comments['date'] = pd.to_datetime(comments['created_utc'], unit='s').dt.strftime("%m, %d, %Y")
submission_df['date'] = pd.to_datetime(submission_df['created_utc'], unit='s').dt.strftime("%m, %d, %Y")

text = pd.concat([comments['body'], submission_df['title']], axis=0)
timestamps = pd.concat([comments['date'], submission_df['date']], axis=0)

#%%
print(len(text.tolist()))


#%%
# Initializing and fitting the model
model = BERTopic()
topics, probs = model.fit_transform(text.tolist())

#%%
# Generate topics over time
topics_over_time = model.topics_over_time(
    text.tolist(),
    timestamps.tolist(),
    nr_bins=7
)

model.visualize_topics_over_time(topics_over_time, top_n_topics=10)

# model.visualize_topics_over_time(topics_over_time, top_n_topics=10)



# %%
print(model.topic_representations_)



# %%
# Gathering the top 5 topic representations over time

print(model.get_topic(0)) # Topics related to art and artists
print(model.get_topic(1)) # Topics related to AI and humans
print(model.get_topic(2)) # Topics related to chatbots
print(model.get_topic(3)) # Topics related to music
print(model.get_topic(4)) # Topics related to AI tiles or something
# %%
