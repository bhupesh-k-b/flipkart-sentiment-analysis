import re
import unicodedata

def normalize(text):
    text=unicodedata.normalize('NFKC',str(text)).lower()
    text=re.sub(r"n['’]t\b",' not',text)
    text=re.sub(r'https?://\S+',' ',text)
    text=re.sub(r'[^a-z\s]',' ',text)
    return re.sub(r'\s+',' ',text).strip()
