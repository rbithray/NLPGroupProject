import numpy as np
from bertopic import BERTopic
import torch 
import pandas as pd
import json
import os

import json

# Open the NDJSON file
from os import path:

with open('data\comments.ndjson', 'r') as f:
    data = [json.loads(line) for line in f]

# `data` is now a list of dictionaries
for item in data:
    print(item)
