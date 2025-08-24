import os
import re
import csv
import json
import time
from tqdm import tqdm
from moonshot import Moonshot  # Kimi K2 API client

# ====== CONFIGURATION ======
client = Moonshot(api_key="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") 

input_dir = r"C:\Users\yashi\Downloads"  # raw TXT files folder
clean_dir = os.path.join(input_dir, "Delhi_Laws_Clean")
chunk_dir = os.path.join(input_dir, "Delhi_Laws_Chunks")
output_csv = r"C:\Users\yashi\Downloads\delhi_law_embeddings.csv"

chunk_size_words = 400
batch_size = 150
start_index = 0

# ====== CREATE FOLDERS ======
os.makedirs(clean_dir, exist_ok=True)
os.makedirs(chunk_dir, exist_ok=True)

# ====== CLEANING FUNCTION ======
def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if re.fullmatch(r"\d+", stripped):
            continue
        if re.match(r"Page\s+\d+(\s+of\s+\d+)?", stripped, re.IGNORECASE):
            continue
        if "delhi gazette" in stripped.lower():
            continue
        if re.search(r"https?://|www\.", stripped):
            continue
        if "downloaded from" in stripped.lower():
            continue
        if stripped.lower().startswith("disclaimer"):
            continue
        if "prs" in stripped.lower():
            continue

        stripped = re.sub(r'[^\x00-\x7F]+', ' ', stripped)

        if re.fullmatch(r"[\W_]+", stripped):
            continue

        stripped = re.sub(r"\s{2,}", " ", stripped)
        cleaned_lines.append(stripped)

    final_lines = []
    previous_blank = False
    for line in cleaned_lines:
        if line == "":
            if not previous_blank:
                final_lines.append(line)
            previous_blank = True
        else:
            final_lines.append(line)
            previous_blank = False

    return "\n".join(final_lines)

# ====== CHUNKING FUNCTION ======
def chunk_text(text, chunk_size=chunk_size_words):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))
    return chunks

# ====== EMBEDDING FUNCTION ======
def get_embedding(text, model="moonshot-k2-embedding", retries=3, delay=2):
    text = text.replace("\n", " ")
    for attempt in range(retries):
        try:
            response = client.embeddings.create(model=model, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}, retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
            time.sleep(delay)
            delay *= 2
    raise Exception("Failed to get embedding after retries.")

# ====== STEP 1: CLEAN + CHUNK ======
txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
print(f"Found {len(txt_files)} TXT files to process.")

for filename in txt_files:
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    cleaned_text = clean_text(raw_text)

    clean_path = os.path.join(clean_dir, filename.replace(".txt", "_clean.txt"))
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    chunks = chunk_text(cleaned_text, chunk_size=chunk_size_words)

    base_name = filename.replace(".txt", "")
    for idx, chunk in enumerate(chunks, start=1):
        chunk_filename = f"{base_name}_chunk{idx}.txt"
        with open(os.path.join(chunk_dir, chunk_filename), "w", encoding="utf-8") as cf:
            cf.write(chunk)

    print(f"Processed {filename} → {len(chunks)} chunks created.")

print("\n✅ All files cleaned and chunked successfully!")

# ====== STEP 2: CREATE EMBEDDINGS (BATCH-WISE) ======
chunk_files = [f for f in os.listdir(chunk_dir) if f.lower().endswith(".txt")]
print(f"Found {len(chunk_files)} chunks to process.")

batch_files = chunk_files[start_index : start_index + batch_size]
file_mode = "w" if start_index == 0 else "a"

with open(output_csv, mode=file_mode, newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if start_index == 0:
        writer.writerow(["filename", "chunk_text", "embedding"])

    for filename in tqdm(batch_files, desc="Processing chunks"):
        file_path = os.path.join(chunk_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            chunk_text_content = f.read().strip()

        if not chunk_text_content:
            continue

        embedding = get_embedding(chunk_text_content)
        embedding_str = json.dumps(embedding)
        writer.writerow([filename, chunk_text_content, embedding_str])

print(f"\n✅ Batch complete! Next run start_index = {start_index + batch_size}")