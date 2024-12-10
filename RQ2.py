from bertopic import BERTopic
import numpy as np
import pandas as pd
import json
from os import path

comments_path = 'data/comments.ndjson'
submissions_path = 'data/submissions.ndjson'


comment_data = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)
submission_data = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)

print(comment_data['body'][:10])
