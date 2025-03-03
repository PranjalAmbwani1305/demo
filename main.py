import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from transformers import pipeline

# Load Pinecone API key from secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]


from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "helpdesk"

index = pc.Index(index_name)

# Load embedding model
embedder = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Function to scrape laws from the main page and subpages
def scrape_saudi_laws():
    base_url = "https://www.saudiembassy.net"
    main_url = f"{base_url}/laws"
    response = requests.get(main_url)
    
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    laws = []

    # Extract main law links
    for law_section in soup.find_all("div", class_="field-content"):
        if not law_section.find("a"):
            continue

        title = law_section.find("a").text.strip()
        sub_link = law_section.find("a")["href"]
        full_link = f"{base_url}{sub_link}" if sub_link.startswith("/") else sub_link

        # Fetch subpage content
        law_text = fetch_law_text(full_link)
        
        laws.append({"title": title, "link": full_link, "text": law_text})

    return laws

# Function to fetch text from subpages
def fetch_law_text(url):
    response = requests.get(url)
    
    if response.status_code != 200:
        return "Content could not be retrieved."

    soup = BeautifulSoup(response.text, "html.parser")
    
    paragraphs = soup.find_all("p")
    law_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
    
    return law_text

# Function to upload data to Pinecone
def upload_to_pinecone(laws):
    for i, law in enumerate(laws):
        vector = embedder(law["text"])[0]  # Convert law text into a vector
        index.upsert([(str(i), vector, {"title": law["title"], "link": law["link"]})])

    return "Laws successfully uploaded to Pinecone!"

# Streamlit UI
st.title("Saudi Laws Scraper & Search")

if st.button("Scrape & Upload Laws"):
    laws_data = scrape_saudi_laws()
    if laws_data:
        result = upload_to_pinecone(laws_data)
        st.success(result)
    else:
        st.error("Failed to scrape the website.")

# Search Function
query = st.text_input("Search Laws:")
if query:
    query_vector = embedder(query)[0]
    
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    
    for match in results["matches"]:
        st.subheader(match["metadata"]["title"])
        st.write(f"[Read More]({match['metadata']['link']})")
