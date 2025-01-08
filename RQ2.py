#%%
!pip install --upgrade nbformat
#%%
from bertopic import BERTopic
import numpy as np
import pandas as pd
import json
from os import path
import nbformat

comments_path = 'data/comments.ndjson'
submissions_path = 'data/submissions.ndjson'


comments_df = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)
submission_data = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)


comments = comments_df[(comments_df['body'] != '[removed]') & (comments_df['body'] != '[deleted]')]
comments['date'] = pd.to_datetime(comments['created_utc'], unit='s').dt.strftime("%d, %m, %Y")
print(comments['date'])
#%%
# Initializing and fitting the model
model = BERTopic()
topics, probs = model.fit_transform(comments['body'])

#%%
# model.visualize_topics()
# model.get_topic_info()

new_topic_labels = {
    0: "AI and Humanity",
    1: "Art and copyright",
    2: "AI and consciousness",
    3: "Chatbots and conversation",
    4: "AI agents and tools"
}


topics_over_time = model.topics_over_time(comments['body'], comments['date'], nr_bins=7, datetime_format='%d, %m, %Y')
model.set_topic_labels(new_topic_labels)

model.visualize_topics_over_time(topics_over_time, custom_labels=new_topic_labels, top_n_topics=5)

# model.visualize_topics_over_time(topics_over_time, top_n_topics=10)



# %%
print(model.topic_representations_)



# %%
