import os
import json
import hashlib
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Constants
MAIN_URL = "https://forum.ceg.vote/latest"
OUTPUT_FILE = "knowledge/metis/ceg.json"
MAX_PROPOSALS = 20  # Maximum number of proposals to collect
COLLECT_COMMENTS = True  # Toggle whether to collect comments

# Configure Chrome options for headless mode
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")

# Automatically download and install the appropriate ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

def generate_date_hash_id(url):
    hash_str = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{hash_str}"

def fetch_proposals():
    driver.get(MAIN_URL)
    proposals = []
    
    # Locate proposal entries on the main page
    rows = driver.find_elements(By.CSS_SELECTOR, "tr[data-topic-id]")
    for row in rows:
        try:
            if len(proposals) >= MAX_PROPOSALS:  # Stop if max proposals reached
                break
            
            # Extract proposal data
            topic_id = row.get_attribute("data-topic-id")
            title_element = row.find_element(By.CSS_SELECTOR, "a.title")
            title = title_element.text.strip()
            url = title_element.get_attribute("href")
            views = row.find_element(By.CSS_SELECTOR, "td.num.views .number").text.strip()
            comments = row.find_element(By.CSS_SELECTOR, "td.num.posts-map .number").text.strip()
            
            # Extract creation and latest activity dates
            activity_cell = row.find_element(By.CSS_SELECTOR, "td.activity")
            date_title = activity_cell.get_attribute("title")
            
            # Extract dates using regex
            created_date = None
            latest_activity = None
            
            created_match = re.search(r"Created: (.*?)(?:\n|$)", date_title)
            if created_match:
                created_date = created_match.group(1).strip()
                
            latest_match = re.search(r"Latest: (.*?)(?:\n|$)", date_title)
            if latest_match:
                latest_activity = latest_match.group(1).strip()
            
            # Generate unique ID
            unique_id = generate_date_hash_id(url)
            
            # Append to proposals list
            proposals.append({
                "id": unique_id,
                "topic_id": topic_id,
                "title": title,
                "url": url,
                "views": views,
                "comments": comments,
                "created_date": created_date,
                "latest_activity": latest_activity
            })
        except Exception as e:
            print(f"Error processing row: {e}")
    return proposals

def fetch_proposal_details(proposal):
    try:
        driver.get(proposal["url"])
        
        # Wait and scrape main content
        content_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "cooked"))
        )
        content = content_element.text.strip()
        proposal["content"] = content
        
        # Collect comments if enabled
        if COLLECT_COMMENTS:
            comments_elements = driver.find_elements(By.CSS_SELECTOR, "div.topic-post")
            comments = []
            for comment in comments_elements:
                author = comment.find_element(By.CSS_SELECTOR, "div.names a").text.strip()
                comment_text = comment.find_element(By.CLASS_NAME, "cooked").text.strip()
                comments.append({"author": author, "comment": comment_text})
            proposal["comments_details"] = comments
    except Exception as e:
        print(f"Error fetching details for {proposal['title']}: {e}")

def save_to_json(proposals):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as file:
        json.dump(proposals, file, indent=4)
    print(f"Data successfully saved to {OUTPUT_FILE}")

def main():
    try:
        # Step 1: Delete existing file if it exists
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
            print(f"Removed existing file: {OUTPUT_FILE}")
        
        # Step 2: Fetch proposals from main page
        proposals = fetch_proposals()
        
        # Step 3: Fetch details for each proposal
        for proposal in proposals:
            fetch_proposal_details(proposal)
        
        # Step 4: Save proposals to JSON
        save_to_json(proposals)
        
        print(f"Successfully collected {len(proposals)} latest proposals")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()