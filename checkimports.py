import importlib

needs = {
    "src.ocr": ["extract_text_from_image"],
    "src.preprocess": ["clean_and_chunk"],
    "src.embedder": ["embed_chunks"],
    "src.indexing": ["store_vectors", "search_vectors"],
    "src.retriever": ["answer_query"],
}

ok = True
for mod, names in needs.items():
    try:
        m = importlib.import_module(mod)
        print(f"✓ imported {mod}")
        for n in names:
            f = getattr(m, n, None)
            if not callable(f):
                ok = False
                print(f"  ✗ missing function: {mod}.{n}")
    except Exception as e:
        ok = False
        print(f"  ✗ import failed for {mod}: {e}")

if ok:
    print("\nALL GOOD ✅")
else:
    print("\nFix the ✗ items above ❗")
