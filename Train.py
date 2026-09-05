# Load Dataset

import json
import pandas as pd

def load_dataset(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        inside_comment = False

        for line in f:

            line = line.strip()

            if not line:
                continue

            # Skip the comment block
            if line.startswith("/*"):
                inside_comment = True
                continue

            if inside_comment:
                if "*/" in line:
                    inside_comment = False
                continue

            # Process JSON lines
            records.append(json.loads(line))

    return pd.DataFrame(records)
    


df=load_dataset("data/KMC-Qwen-KG_triples.jsonl")

print(df.head())

# Unicode Normalization

import unicodedata
import re

def unicode_normalization(text):

    if text is None:
        return ""

    text=unicodedata.normalize("NFKC",text)

    return text
    
# Scientific Text Cleaning
def scientific_cleaning(text):

    text=unicode_normalization(text)

    text=re.sub(r"http\S+"," ",text)

    text=re.sub(r"\[[0-9]+\]"," ",text)

    text=re.sub(r"\([^)]*et al\.,?.*?\)"," ",text)

    text=re.sub(r"\s+"," ",text)

    return text.strip()

# Sliding Window Chunking
def sliding_window(text,
                   window=250,
                   overlap=50):

    words=text.split()

    chunks=[]

    start=0

    while start<len(words):

        end=start+window

        chunk=" ".join(words[start:end])

        chunks.append(chunk)

        start=end-overlap

    return chunks
    
# Few-shot Chain-of-Thought Prompt
FEW_SHOT_PROMPT="""

You are a scientific knowledge extraction expert.

Extract triples.

Example:

Text:

Aspirin reduces fever.

Reason:

Subject=Aspirin

Relation=reduces

Object=fever

Output:

(Aspirin, reduces, fever)

Now analyze:

"""

# Ollama Triple Extraction
import ollama

def extract_triples(text):

    prompt=FEW_SHOT_PROMPT+text

    response=ollama.chat(

        model="qwen2.5",

        messages=[

            {"role":"user",

             "content":prompt}

        ]
    )

    return response["message"]["content"]
    
import ollama

def extract_triples(text):

    prompt=FEW_SHOT_PROMPT+text

    response=ollama.chat(

        model="qwen2.5",

        messages=[

            {"role":"user",

             "content":prompt}

        ]
    )

    return response["message"]["content"]

# Parse Triples
import re

def parse_triples(output):

    triples=[]

    pattern=r"\((.*?),(.*?),(.*?)\)"

    matches=re.findall(pattern,output)

    for s,r,o in matches:

        triples.append([

            s.strip(),

            r.strip(),

            o.strip()

        ])

    return triples
    
# LaBSE Cross-Lingual Entity Alignment
from sentence_transformers import SentenceTransformer

model=SentenceTransformer(
    "sentence-transformers/LaBSE"
)


# Entity Embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("sentence-transformers/LaBSE")

def align_entities(entities, threshold=0.90, batch_size=512):

    embeddings = model.encode(
        entities,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    mapping = {}

    for i, entity in enumerate(entities):
        mapping[entity] = entity

    # Only compare nearby entities (window)
    window = 20

    for i in range(len(entities)):

        end = min(i + window, len(entities))

        sims = cosine_similarity(
            embeddings[i:i+1],
            embeddings[i+1:end]
        )[0]

        for j, score in enumerate(sims):

            if score > threshold:
                mapping[entities[i+1+j]] = entities[i]

    return mapping
    

#Apply Entity Alignment
def merge_triples(triples):

    entities = []

    # Collect all entity names
    for head, relation, tail, head_type, tail_type in triples:
        entities.append(head)
        entities.append(tail)

    entities = list(set(entities))

    print("Unique entities:", len(entities))

    # Align entities
    mapping = align_entities(entities)

    merged = []

    # Replace aligned entity names
    for head, relation, tail, head_type, tail_type in triples:

        merged.append({
            "head": mapping.get(head, head),
            "relation": relation,
            "tail": mapping.get(tail, tail),
            "head_type": head_type,
            "tail_type": tail_type
        })

    return merged

# Neo4j Graph Construction
from neo4j import GraphDatabase

driver=GraphDatabase.driver(

"bolt://localhost:7687",

auth=("neo4j","password")
)

#Insert Nodes
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)

def insert_graph(triples):

    with driver.session() as session:

        for t in triples:

            session.run("""

            MERGE (h:Entity {
                name:$head,
                type:$head_type
            })

            MERGE (t:Entity {
                name:$tail,
                type:$tail_type
            })

            MERGE (h)-[:RELATION {
                name:$relation
            }]->(t)

            """,

            head=t["head"],
            tail=t["tail"],
            relation=t["relation"],
            head_type=t["head_type"],
            tail_type=t["tail_type"])

# Main Pipeline
from tqdm import tqdm

df=load_dataset("data/KMC-Qwen-KG_triples.jsonl")
all_triples = []

for _, row in df.iterrows():

    for t in row["triples"]:

        head = t["head"]
        relation = t["relation"]
        tail = t["tail"]
        head_type = t["head_type"]
        tail_type = t["tail_type"]

        all_triples.append(
            (head, relation, tail, head_type, tail_type)
        )

print("Total triples:", len(all_triples))


merged=merge_triples(all_triples)

insert_graph(merged)

print("Knowledge Graph Created")
