#%%
from bertopic import BERTopic
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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

vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df = 10)
# Initialize BERTopic model
model = BERTopic(vectorizer_model=vectorizer_model)

# Fit the model
topics, probs = model.fit_transform(comments['body'])
#%%

# Define the list of keywords to search for
keywords = [
    "regulations", "regulation", "regulatory", "law", "laws", 
    "government", "policy", "policies", "politics", "rules", 
    "control", "management", "governance", "conduct", "statute",
    "ordinance", "order", "monitoring", "supervision", "restriction",
    "oversight", "limitations", "guidelines", "compliance", "enforcement",
    "authority", "bylaw", "license", "standards", "mandate", "police",
    "judge", "court", "fairness", "ethics", "country",
    "act", "charter", "code", "decree", "edict", "legislation", 
    "proclamation", "provision", "ruling", "sanction", "treaty", 
    "protocol", "constitution", "convention", "ordinance", "clause",
    "inspection", "audit", "adjudication", "penalties", "fines", 
    "licensing", "mandates", "verification", "certification", 
    "accountability", "adherence", "obligations", "restrictions", 
    "surveillance", "checks", "safeguards",
    "bureaucracy", "legislature", "senate", "congress", "parliament", 
    "ministry", "executive", "agency", "department", "cabinet", 
    "council", "committee", "assembly", "tribunal", "sovereignty", 
    "jurisdiction", "administration",
    "equity", "morality", "justice", "integrity", "honesty", 
    "transparency", "accountability", "responsibility", "standards", 
    "good practices",
    "oversight", "governance", "dominion", "superintendence", 
    "command", "directive", "guidance", "arbitration", "mediation",
    "benchmark", "criteria", "protocols", "quality assurance", 
    "specification", "standardization", "prescriptions", "guidelines", 
    "recommendations", "ruleset", "framework"
]


# Create an empty set to store unique indices
all_indices = set()

# Iterate over each keyword to find similar topics and their indices
for keyword in keywords:
    similar_topics, similarity = model.find_topics(keyword, top_n=5)
    
    if similar_topics:
        # Get the most similar topic ID for the current keyword
        topic_id = similar_topics[0]

        # boolean mask for topic matching
        topic_mask = [t == topic_id for t in topics]  # Create a boolean mask
        topic_comments = comments[topic_mask]        

        all_indices.update(topic_comments.index)

        print(f"Keyword '{keyword}' - Topic ID: {topic_id}, Matches: {len(topic_comments)}")
    else:
        print(f"No topics found for keyword '{keyword}'.")

# Convert the set of indices to a sorted list (optional)
all_indices_sorted = sorted(list(all_indices))


print(f"Total unique relevant comment indices: {len(all_indices_sorted)}")
print(all_indices_sorted)

import json


all_indices_sorted = sorted(list(all_indices))
comments_body = comments["body"]  # De comments

# een dictionary met indices en bijbehorende comments
comments_with_indices = {index: comments_body[index] for index in all_indices_sorted}


with open('comments.json', 'w', encoding='utf-8') as f:
    json.dump(comments_with_indices, f, ensure_ascii=False, indent=4)

print("Comments opgeslagen in 'comments.json'")

