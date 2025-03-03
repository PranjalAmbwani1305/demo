import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Pinecone API Setup
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
index_name = "helpddesk"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Scraper Function
def scrape_laws(url):
    """Scrapes laws, extracts articles & chapters, and structures them properly."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"Failed to access website (Status Code: {response.status_code})")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extracting Chapters & Articles
    chapters = []
    current_chapter = None

    for element in soup.find_all(["h2", "h3", "p"]):  # Adjust if structure differs
        text = element.get_text(strip=True)

        if text.lower().startswith("chapter"):  # Detect chapter headings
            current_chapter = {"chapter": text, "articles": []}
            chapters.append(current_chapter)

        elif text.lower().startswith("article") and current_chapter:
            current_chapter["articles"].append(text)

    return chapters

# Upload and Scrape
st.title("Saudi Laws Scraper & Pinecone Search")
website_url = st.text_input("Enter Website URL:")

if st.button("Scrape & Upload"):
    if website_url:
        scraped_data = scrape_laws(website_url)
        
        if scraped_data:
            st.success(f"Successfully scraped {len(scraped_data)} chapters.")
            
            # Upload to Pinecone
            for i, chapter in enumerate(scraped_data):
                chapter_text = " ".join(chapter["articles"])
                vector = embedder.encode(chapter_text)

                index.upsert([
                    (str(i), vector.tolist(), {"chapter": chapter["chapter"], "text": chapter_text})
                ])
            
            st.success("Laws uploaded to Pinecone successfully!")
        else:
            st.error("Failed to scrape website content.")
    else:
        st.error("Please enter a valid URL.")

# Search Laws
query = st.text_input("Search for Articles & Chapters:")
if query:
    query_vector = embedder.encode(query)
    results = index.query(vector=query_vector.tolist(), top_k=5, include_metadata=True)

    for match in results["matches"]:
        st.subheader(match["metadata"]["chapter"])
        st.write(match["metadata"]["text"])
