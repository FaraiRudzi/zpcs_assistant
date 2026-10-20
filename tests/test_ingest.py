from app.ingest import chunk_pages, clean_text, is_heading

def test_clean_text_joins_hyphenated_words():
    assert clean_text("rehabili-\ntation  plan") == "rehabilitation plan"

def test_heading_detection():
    assert is_heading("Section 2 Eligibility for Vocational Training")
    assert is_heading("4.2 Skills Development")
    assert not is_heading("This is an ordinary sentence that ends with a full stop.")

def test_chunks_keep_page_and_section():
    text = "Section 1 Purpose\n" + " ".join(["word"] * 400)
    chunks = chunk_pages([(3, text)])
    assert len(chunks) >= 2
    assert all(c["page"] == 3 and c["section"].startswith("Section 1") for c in chunks)
