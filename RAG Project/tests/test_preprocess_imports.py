def test_imports_work():
    import src.ocr as ocr
    import src.preprocess as prep
    assert hasattr(prep, 'normalize_text')
    assert hasattr(ocr, 'extract_text_from_image')
