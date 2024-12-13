from bertopic import BERTopic
import numpy as np
import pandas as pd
import json
from os import path

comments_path = 'data/comments.ndjson'
submissions_path = 'data/submissions.ndjson'


# comment_data = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)
# submission_data = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)
def load_comment_data(path):
    comment_data = []
    with open(comments_path, 'r') as f:
        for line in f:
            data = json.loads(line)
    if data['body'] != '[removed]' or data['body'] != '[deleted]':
        comment_data.append(data['body'])
    
    return comment_data

comment_data = load_comment_data(comments_path)

model = BERTopic()
# print(comment_data['body'][:10])
topics, probs = model.fit_transform(comment_data['body'])
model.visualize_topics()

