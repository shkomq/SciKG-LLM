# ==========================================================
# Shko Muhammed Qader
# PhD Student, Tianjin University, China
# Load Dataset + Multilingual LLM Framework
# Scientific Knowledge Graph Construction
# from Low-Resource Language Literature
#
# Output:
#   1. multilingual_triples.csv
#   2. multilingual_entities.csv
# ==========================================================

import json
import pandas as pd
import unicodedata
import re
from tqdm import tqdm

# ----------------------------------------------------------
# Load Dataset (.jsonl)
# ----------------------------------------------------------
def load_dataset(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        inside_comment = False

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("/*"):
                inside_comment = True
                continue

            if inside_comment:
                if "*/" in line:
                    inside_comment = False
                continue

            records.append(json.loads(line))

    return pd.DataFrame(records)


# ----------------------------------------------------------
# Unicode Normalization
# ----------------------------------------------------------
def unicode_normalization(text):

    if text is None:
        return ""

    return unicodedata.normalize("NFKC", str(text))


# ----------------------------------------------------------
# Scientific Text Cleaning
# ----------------------------------------------------------
def scientific_cleaning(text):

    text = unicode_normalization(text)

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\([^)]*et al\.,?.*?\)", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = load_dataset("data/KMC-Qwen-KG_triples.jsonl")

print("Documents :", len(df))


# ----------------------------------------------------------
# Extract multilingual triples from dataset
# ----------------------------------------------------------
triples = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    if "triples" not in row:
        continue

    for t in row["triples"]:

        triples.append({
            "head": scientific_cleaning(t["head"]),
            "relation": scientific_cleaning(t["relation"]),
            "tail": scientific_cleaning(t["tail"]),
            "head_type": scientific_cleaning(t["head_type"]),
            "tail_type": scientific_cleaning(t["tail_type"])
        })


triples_df = pd.DataFrame(triples)

print("Total Triples :", len(triples_df))


# ----------------------------------------------------------
# Remove duplicate triples
# ----------------------------------------------------------
triples_df = triples_df.drop_duplicates(
    subset=["head", "relation", "tail"]
).reset_index(drop=True)


print("Unique Triples :", len(triples_df))


# ----------------------------------------------------------
# Build multilingual entity table
# ----------------------------------------------------------
head_entities = triples_df[["head", "head_type"]].rename(
    columns={
        "head": "entity",
        "head_type": "entity_type"
    }
)

tail_entities = triples_df[["tail", "tail_type"]].rename(
    columns={
        "tail": "entity",
        "tail_type": "entity_type"
    }
)

entities_df = pd.concat(
    [head_entities, tail_entities],
    ignore_index=True
).drop_duplicates().reset_index(drop=True)

print("Unique Entities :", len(entities_df))


# ----------------------------------------------------------
# Save Outputs
# ----------------------------------------------------------
triples_df.to_csv(
    "multilingual_triples.csv",
    index=False,
    encoding="utf-8-sig"
)

entities_df.to_csv(
    "multilingual_entities.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV files saved successfully.")
print(" -> multilingual_triples.csv")
print(" -> multilingual_entities.csv")


# ==========================================================
# SciKG-LLM
# A Multilingual Framework for Scientific Knowledge Extraction
# and Multilingual Knowledge Graph Construction from
# Low-Resource Scientific Literature
#
# Output:
#   1. SciKG_LLM_Triples.csv
#   2. SciKG_LLM_KnowledgeGraph.csv
# ==========================================================

import json
import pandas as pd
import unicodedata
import re
from tqdm import tqdm

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
def load_dataset(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        inside_comment = False

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("/*"):
                inside_comment = True
                continue

            if inside_comment:
                if "*/" in line:
                    inside_comment = False
                continue

            records.append(json.loads(line))

    return pd.DataFrame(records)


# ----------------------------------------------------------
# Unicode Normalization
# ----------------------------------------------------------
def normalize_text(text):

    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------------------------------
# Scientific Cleaning
# ----------------------------------------------------------
def clean_scientific_text(text):

    text = normalize_text(text)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\([^)]*et al\.,?.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------------------------------
# SciKG-LLM Knowledge Extraction
# ----------------------------------------------------------
df = load_dataset("data/KMC-Qwen-KG_triples.jsonl")

print("Documents :", len(df))

triples = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    if "triples" not in row:
        continue

    for triple in row["triples"]:

        head = clean_scientific_text(triple["head"])
        relation = clean_scientific_text(triple["relation"])
        tail = clean_scientific_text(triple["tail"])

        head_type = clean_scientific_text(triple["head_type"])
        tail_type = clean_scientific_text(triple["tail_type"])

        triples.append({

            "Source": head,
            "Relation": relation,
            "Target": tail,
            "Source_Type": head_type,
            "Target_Type": tail_type

        })


# ----------------------------------------------------------
# Create Triple DataFrame
# ----------------------------------------------------------
triples_df = pd.DataFrame(triples)

triples_df.drop_duplicates(
    subset=["Source", "Relation", "Target"],
    inplace=True
)

triples_df.reset_index(drop=True, inplace=True)

print("Unique Scientific Triples :", len(triples_df))


# ----------------------------------------------------------
# Create Knowledge Graph Edge List
# ----------------------------------------------------------
kg_df = triples_df.copy()

kg_df.insert(0, "Edge_ID", range(1, len(kg_df)+1))

kg_df["Weight"] = 1

kg_df["Language"] = "Multilingual"

kg_df["Framework"] = "SciKG-LLM"


# ----------------------------------------------------------
# Save CSV Files
# ----------------------------------------------------------
triples_df.to_csv(
    "SciKG_LLM_Triples.csv",
    index=False,
    encoding="utf-8-sig"
)

kg_df.to_csv(
    "SciKG_LLM_KnowledgeGraph.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===================================")
print("SciKG-LLM Processing Completed")
print("===================================")
print("Triples       :", len(triples_df))
print("Knowledge Edges :", len(kg_df))
print("\nSaved Files")
print("1. SciKG_LLM_Triples.csv")
print("2. SciKG_LLM_KnowledgeGraph.csv")
# ==========================================================
# Kurdish Medical Corpus (KMC) Preprocessing
#
# Dataset : KMC-Qwen-KG_triples.jsonl
#
# Steps
# 1. Unicode Normalization
# 2. Scientific Text Cleaning
# 3. Sliding Window Triple Chunking
#
# Output:
#    KMC_Preprocessed_Documents.csv
# ==========================================================

import json
import pandas as pd
import unicodedata
import re
from tqdm import tqdm


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
def load_dataset(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        inside_comment = False

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("/*"):
                inside_comment = True
                continue

            if inside_comment:
                if "*/" in line:
                    inside_comment = False
                continue

            records.append(json.loads(line))

    return pd.DataFrame(records)


# ----------------------------------------------------------
# Unicode Normalization
# ----------------------------------------------------------
def unicode_normalization(text):

    if text is None:
        return ""

    return unicodedata.normalize("NFKC", str(text))


# ----------------------------------------------------------
# Scientific Cleaning
# ----------------------------------------------------------
def scientific_cleaning(text):

    text = unicode_normalization(text)

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"\([^)]*et al\.,?.*?\)", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ----------------------------------------------------------
# Sliding Window
# ----------------------------------------------------------
def sliding_window(text,
                   window_size=250,
                   overlap=50):

    words = text.split()

    if len(words) == 0:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + window_size, len(words))

        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = load_dataset("data/KMC-Qwen-KG_triples.jsonl")

print("Documents :", len(df))


# ----------------------------------------------------------
# Generate Preprocessed Dataset
# ----------------------------------------------------------
processed = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    doc_id = row["id"]

    triple_sentences = []

    for triple in row["triples"]:

        head = scientific_cleaning(triple["head"])
        relation = scientific_cleaning(triple["relation"])
        tail = scientific_cleaning(triple["tail"])

        sentence = f"{head} {relation} {tail}"

        triple_sentences.append(sentence)

    # Merge all triples of one document into one multilingual text
    document_text = " ".join(triple_sentences)

    normalized_text = unicode_normalization(document_text)

    cleaned_text = scientific_cleaning(document_text)

    chunks = sliding_window(
        cleaned_text,
        window_size=250,
        overlap=50
    )

    for chunk_id, chunk in enumerate(chunks, start=1):

        processed.append({

            "Document_ID": doc_id,
            "Chunk_ID": chunk_id,
            "Original_Text": document_text,
            "Normalized_Text": normalized_text,
            "Cleaned_Text": cleaned_text,
            "Chunk_Text": chunk

        })


# ----------------------------------------------------------
# Save CSV
# ----------------------------------------------------------
processed_df = pd.DataFrame(processed)

processed_df.to_csv(
    "KMC_Preprocessed_Documents.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n======================================")
print("KMC Preprocessing Completed")
print("======================================")
print("Processed Documents :", len(df))
print("Generated Chunks    :", len(processed_df))
print("Saved File          : KMC_Preprocessed_Documents.csv")

# ==========================================================
# SciKG-LLM
# Few-shot Chain-of-Thought Scientific Triple Extraction
#
# Input :
#    KMC_Preprocessed_Documents.csv
#
# Output :
#    SciKG_LLM_Extracted_Triples.csv
# ==========================================================

import pandas as pd
import ollama
import re
from tqdm import tqdm


# ----------------------------------------------------------
# Load Preprocessed Dataset
# ----------------------------------------------------------
#df = pd.read_csv("KMC_Preprocessed_Documents.csv")
df = pd.read_csv("KMC_Preprocessed_Documents.csv").head(10)

print("Loaded Chunks :", len(df))
print(df.columns)


# ----------------------------------------------------------
# Few-shot CoT Prompt
# ----------------------------------------------------------
FEW_SHOT_PROMPT = """
You are a scientific knowledge extraction expert.

Extract all scientific knowledge triples from the following scientific text.

Return ONLY triples in the format

(Subject, Relation, Object)

Do not explain anything.

Text:
"""


# ----------------------------------------------------------
# Ollama Triple Extraction
# ----------------------------------------------------------
def extract_triples(text):

    response = ollama.chat(

        model="qwen2.5",

        messages=[

            {
                "role": "user",
                "content": FEW_SHOT_PROMPT + str(text)
            }

        ]

    )

    return response["message"]["content"]


# ----------------------------------------------------------
# Parse Extracted Triples
# ----------------------------------------------------------
def parse_triples(output):

    pattern = r"\((.*?),(.*?),(.*?)\)"

    matches = re.findall(pattern, output)

    triples = []

    for head, relation, tail in matches:

        triples.append(

            [
                head.strip(),
                relation.strip(),
                tail.strip()
            ]

        )

    return triples


# ----------------------------------------------------------
# Extract Triples
# ----------------------------------------------------------
results = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    chunk_text = str(row["Chunk_Text"])

    try:

        llm_output = extract_triples(chunk_text)

        triples = parse_triples(llm_output)

        if len(triples) == 0:

            results.append({

                "Document_ID": row["Document_ID"],
                "Chunk_ID": row["Chunk_ID"],
                "Head": "",
                "Relation": "",
                "Tail": "",
                "LLM_Output": llm_output

            })

        else:

            for head, relation, tail in triples:

                results.append({

                    "Document_ID": row["Document_ID"],
                    "Chunk_ID": row["Chunk_ID"],
                    "Head": head,
                    "Relation": relation,
                    "Tail": tail,
                    "LLM_Output": llm_output

                })

    except Exception as e:

        print(f"Error in Document {row['Document_ID']} Chunk {row['Chunk_ID']}")

        results.append({

            "Document_ID": row["Document_ID"],
            "Chunk_ID": row["Chunk_ID"],
            "Head": "",
            "Relation": "",
            "Tail": "",
            "LLM_Output": str(e)

        })


# ----------------------------------------------------------
# Save CSV
# ----------------------------------------------------------
triples_df = pd.DataFrame(results)

triples_df.to_csv(

    "SciKG_LLM_Extracted_Triples.csv",

    index=False,

    encoding="utf-8-sig"

)


print("\n========================================")
print("SciKG-LLM Triple Extraction Completed")
print("========================================")
print("Processed Chunks :", len(df))
print("Extracted Triples :", len(triples_df))
print("Saved File : SciKG_LLM_Extracted_Triples.csv")



# ==========================================================
# SciKG-LLM
# LaBSE-based Cross-Lingual Entity Alignment
# using Cosine Similarity Matching
#
# Input :
#     SciKG_LLM_Extracted_Triples.csv
#
# Output :
#     SciKG_LLM_Aligned_Triples.csv
#     SciKG_LLM_Entity_Mapping.csv
# ==========================================================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ----------------------------------------------------------
# Load Extracted Triples
# ----------------------------------------------------------
df = pd.read_csv("SciKG_LLM_Extracted_Triples.csv")

print("Total Triples :", len(df))


# ----------------------------------------------------------
# Load LaBSE Model
# ----------------------------------------------------------
print("Loading LaBSE Model...")

model = SentenceTransformer(
    "sentence-transformers/LaBSE"
)

print("Model Loaded")


# ----------------------------------------------------------
# Collect Unique Entities
# ----------------------------------------------------------
entities = sorted(
    set(df["Head"].dropna().tolist()) |
    set(df["Tail"].dropna().tolist())
)

print("Unique Entities :", len(entities))


# ----------------------------------------------------------
# Generate Entity Embeddings
# ----------------------------------------------------------
embeddings = model.encode(
    entities,
    batch_size=256,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ----------------------------------------------------------
# Cross-Lingual Entity Alignment
# ----------------------------------------------------------
threshold = 0.90

mapping = {}

for entity in entities:
    mapping[entity] = entity


print("Performing Entity Alignment...")
for i in tqdm(range(len(entities) - 1)):

    remaining_embeddings = embeddings[i + 1:]

    if len(remaining_embeddings) == 0:
        break

    sims = cosine_similarity(
        embeddings[i:i+1],
        remaining_embeddings
    )[0]

    for j, score in enumerate(sims):

        if score >= threshold:

            entity1 = entities[i]
            entity2 = entities[i + 1 + j]

            mapping[entity2] = entity1


# ----------------------------------------------------------
# Save Entity Mapping
# ----------------------------------------------------------
mapping_df = pd.DataFrame({

    "Original_Entity": list(mapping.keys()),
    "Aligned_Entity": list(mapping.values())

})

mapping_df.to_csv(
    "SciKG_LLM_Entity_Mapping.csv",
    index=False,
    encoding="utf-8-sig"
)


# ----------------------------------------------------------
# Replace Entities
# ----------------------------------------------------------
aligned_df = df.copy()

aligned_df["Head"] = aligned_df["Head"].map(
    lambda x: mapping.get(x, x)
)

aligned_df["Tail"] = aligned_df["Tail"].map(
    lambda x: mapping.get(x, x)
)


# ----------------------------------------------------------
# Remove Duplicate Triples
# ----------------------------------------------------------
aligned_df.drop_duplicates(
    subset=["Head", "Relation", "Tail"],
    inplace=True
)

aligned_df.reset_index(drop=True, inplace=True)


# ----------------------------------------------------------
# Save Aligned Triples
# ----------------------------------------------------------
aligned_df.to_csv(
    "SciKG_LLM_Aligned_Triples.csv",
    index=False,
    encoding="utf-8-sig"
)


# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
print("\n====================================")
print("Cross-Lingual Entity Alignment Done")
print("====================================")

print("Original Triples :", len(df))
print("Aligned Triples  :", len(aligned_df))
print("Unique Entities  :", len(entities))

print("\nSaved Files")
print("1. SciKG_LLM_Entity_Mapping.csv")
print("2. SciKG_LLM_Aligned_Triples.csv")



# ==========================================================
# Neo4j Property Graph Construction
# Searchable Multilingual Scientific Knowledge Graph
#
# Input:
#    SciKG_LLM_Aligned_Triples.csv
#
# Outputs:
#    Neo4j_Nodes.csv
#    Neo4j_Relationships.csv
#    Multilingual_Scientific_KG.png (600 dpi)
# ==========================================================

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Noto Sans"

# ----------------------------------------------------------
# Font (supports many Unicode characters if installed)
# ----------------------------------------------------------
matplotlib.rcParams["font.family"] = ["Noto Sans", "DejaVu Sans"]

# ----------------------------------------------------------
# Load Aligned Triples
# ----------------------------------------------------------
df = pd.read_csv("SciKG_LLM_Aligned_Triples.csv")

print("Triples :", len(df))

# Remove rows with missing values
df = df.dropna(subset=["Head", "Relation", "Tail"])

# ----------------------------------------------------------
# Create Node Table
# ----------------------------------------------------------
nodes = pd.DataFrame({
    "Node": pd.concat([df["Head"], df["Tail"]], ignore_index=True)
})

nodes = nodes.drop_duplicates().reset_index(drop=True)

nodes.insert(0, "Node_ID", range(1, len(nodes)+1))

nodes["Label"] = "Entity"

nodes.to_csv(
    "Neo4j_Nodes.csv",
    index=False,
    encoding="utf-8-sig"
)

# ----------------------------------------------------------
# Create Relationship Table
# ----------------------------------------------------------
relationships = df.copy()

relationships.insert(
    0,
    "Relationship_ID",
    range(1, len(relationships)+1)
)

relationships.to_csv(
    "Neo4j_Relationships.csv",
    index=False,
    encoding="utf-8-sig"
)

# ----------------------------------------------------------
# Construct Property Graph
# ----------------------------------------------------------
G = nx.DiGraph()

for _, row in df.iterrows():

    G.add_node(row["Head"])

    G.add_node(row["Tail"])

    G.add_edge(
        row["Head"],
        row["Tail"],
        relation=row["Relation"]
    )

print("Nodes :", G.number_of_nodes())
print("Edges :", G.number_of_edges())

# ----------------------------------------------------------
# Draw Graph (Publication Figure)
# ----------------------------------------------------------
plt.figure(figsize=(12,12))

pos = nx.spring_layout(
    G,
    seed=42,
    k=1.5
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=900
)

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowsize=14,
    width=1.2
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=16
)

plt.title(
    "Multilingual Scientific Knowledge Graph",
    fontsize=18
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "Multilingual_Scientific_KG.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
print("\n======================================")
print("Neo4j Property Graph Construction Done")
print("======================================")

print("Nodes :", len(nodes))
print("Relationships :", len(relationships))

print("\nSaved Files")
print("1. Neo4j_Nodes.csv")
print("2. Neo4j_Relationships.csv")
print("3. Multilingual_Scientific_KG.png (600 dpi)")
exec(open('perf.py').read())