import numpy as np
from bertopic import BERTopic
import torch 
import pandas as pd
import json
from os import path

comments = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)
submissions = pd.read_json(path.join('data', 'submissions.ndjson'), lines=True)

print(comments.columns)