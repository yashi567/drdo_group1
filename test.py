import time
from pathlib import Path
from extract_clean_chunk import (
    extract_text_txt,
    normalize_whitespace,
    remove_common_headers_footers,
    chunk_text
)

fp = Path(r"C:\Users\ironm\Desktop\YashiDRDO\Delhi Commission for Women Act.txt")
print(f"Testing: {fp}")

t0 = time.time()
text = extract_text_txt(fp)
print("Extracted chars:", len(text), f"({time.time()-t0:.3f}s)")

t1 = time.time()
text = normalize_whitespace(text)
print("After normalize:", len(text), f"({time.time()-t1:.3f}s)")

t2 = time.time()
text = remove_common_headers_footers(text)
print("After header/footer removal:", len(text), f"({time.time()-t2:.3f}s)")

t3 = time.time()
chunks = chunk_text(text, chunk_size=1200, overlap=200)
print("Chunks created:", len(chunks), f"({time.time()-t3:.3f}s)")

if chunks:
    print("\nFirst chunk preview:\n", chunks[0][2][:400], "...")
