import numpy as np
from bertopic import BERTopic
import pandas as pd
from os import path

comments = pd.read_json(path.join('data', 'comments.ndjson'), lines=True)
submissions = pd.read_json(path.join('data', 'submissions.ndjson'), lines=True)

model = BERTopic()

topics, probs = model.fit_transform(comments['body'])
model.visualize_topics()

similar_topics, similarity = model.find_topics("regulation", top_n=5)
model.get_topic(similar_topics[0])

if similar_topics:
    regulation_topic_id = similar_topics[0]
    print(f"Topic-ID voor 'regulation': {regulation_topic_id}")

    # Voeg de gegenereerde topics toe aan de oorspronkelijke DataFrame
    comments['topic'] = topics

    # Filter opmerkingen die zijn toegewezen aan het relevante topic-ID
    regulation_comments = comments[comments['topic'] == regulation_topic_id]

    # Print of verwerk de opmerkingen
    print(f"Aantal opmerkingen over 'regulation': {len(regulation_comments)}")
    print(regulation_comments['body'])
else:
    print("Geen relevante topics gevonden voor 'regulation'.")


politics_topics, similarity = model.find_topics("politics", top_n=5)
model.get_topic(politics_topics[0])

if similar_topics:
    politics_topic_id = politics_topics[0]
    print(f"Topic-ID voor 'politics': {politics_topic_id}")

    # Voeg de gegenereerde topics toe aan de oorspronkelijke DataFrame
    comments['topic'] = topics

    # Filter opmerkingen die zijn toegewezen aan het relevante topic-ID
    politics_comments = comments[comments['topic'] == politics_topic_id]

    # Print of verwerk de opmerkingen
    print(f"Aantal opmerkingen over 'politics': {len(politics_comments)}")
    print(politics_comments['body'])
else:
    print("Geen relevante topics gevonden voor 'politics'.")


print("subject politics indices:" ,politics_comments.index)
print("subject regulation indices: ", regulation_comments.index)