import os
from bs4 import BeautifulSoup

def extract_reuters_text(data_dir):

    texts = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".sgm"):
            continue

        filepath = os.path.join(data_dir, filename)

        with open(filepath, "r", encoding="latin-1") as f:
            soup = BeautifulSoup(f, "html.parser")

            for article in soup.find_all("reuters"):
                text_tag = article.find("text")
                if not text_tag:
                    continue

                body = text_tag.find("body")
                title = text_tag.find("title")

                if body and body.text.strip():
                    texts.append(body.text.strip())
                elif title and title.text.strip():
                    texts.append(title.text.strip())

    return texts
