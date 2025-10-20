from pathlib import Path
import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

input_dir = Path(r"C:\Users\Elio\Desktop\410final\FakeNewsNet\code\fakenewsnet_dataset\politifact")

records = []

for f in input_dir.rglob("*.json"):
    try:
        with open(f, "r", encoding="utf-8") as infile:
            data = json.load(infile)
    except Exception as e:
        print(f"[skip] {f.name}: {e}")
        continue

    if "fake" in f.parts:
        label = "fake"
    elif "real" in f.parts:
        label = "real"
    else:
        label = "unknown"

    if isinstance(data, dict):
        data = [data]

    for item in data:
        records.append({
            "id": item.get("id") or item.get("_id") or "",
            "title": item.get("title") or "",
            "text": item.get("text") or item.get("content") or "",
            "author": item.get("author") or item.get("authors") or "",
            "label": label
        })

df = pd.DataFrame(records)

df = df[df["text"].str.len() > 200]
df["author"] = df["author"].apply(lambda a: ", ".join(a) if isinstance(a, list) else a)

df["text"] = (
    df["text"]
    .astype(str)
    .str.replace(r"[\r\n\t]+", " ", regex=True) 
    .str.replace(r"\s+", " ", regex=True)      
    .str.strip()
)

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)
df["clean_text"] = df["text"].apply(clean_text)


print(f"Extracted {len(df)} records")
print(df.head())

output_path = r"C:\Users\Elio\Desktop\410final\UIUC-CS410-fake_news_detection\politifact_extracted.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
