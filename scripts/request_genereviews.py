import requests

# NBKid que tu veux récupérer
nbk_id = "NBK1168"

# URL pour récupérer le texte complet
url = f"https://www.ncbi.nlm.nih.gov/books/NBK{nbk_id}/?report=reader&format=text"

r = requests.get(url)
if r.status_code == 200:
    text = r.text
    with open(f"NBK{nbk_id}.txt", "w", encoding="utf-8") as f:
        f.write(text)
else:
    print(f"Erreur {r.status_code} pour NBK{nbk_id}")
