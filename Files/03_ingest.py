from pypdf import PdfReader

reader = PdfReader("data/code_penal.pdf")
#print(len(reader.pages))                      # how many pages

#print(reader.pages[0].extract_text()[:500])  # first 500 characters of page 1



full_text = ""

for page in reader.pages:
    text = page.extract_text()
    full_text = full_text + text

#print(len(full_text))

chunks = []                                    # empty list of pieces
size = 800                                     # characters per piece
overlap = 150                                  # characters shared between 2 pieces

for start in range(0, len(full_text), size - overlap):
    piece = full_text[start : start + size]    # slice: from start, 800 characters
    chunks.append(piece)                       # add the piece to the list

print(len(chunks))
print(chunks[500])
