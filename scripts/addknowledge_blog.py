import os
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import hashlib

# Constants
BLOG_URL = "https://metis.io/blog"
MAX_BLOGS = 10
DAYS_TO_KEEP = 99999  # Number of days after which articles are removed (set high for no maintenance)
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the JSON path relative to the script's location
JSON_PATH = os.path.join(SCRIPT_DIR, "../knowledge/metis/blog.json")


def get_latest_blog_id():
    if not os.path.exists(JSON_PATH):
        return None
    try:
        with open(JSON_PATH, "r") as file:
            data = json.load(file)
        return (
            data["blogs"][0]["id"] if data["blogs"] else None
        )  # Get the latest ID from the top blog
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def fetch_blog_posts():
    response = requests.get(BLOG_URL)
    if response.status_code != 200:
        raise Exception("Failed to fetch blog page.")
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.find_all(
        "div", {"role": "listitem", "class": "collection-item tech w-dyn-item"}
    )


def parse_blog_item(item):
    link_tag = item.find("a", {"aria-label": "link-article"})
    link = link_tag["href"]
    date = item.find("div", class_="text-1-pc").text.strip()
    author = (
        item.find("div", class_="autor-tag").text.strip()
        if item.find("div", class_="autor-tag")
        else "Unknown"
    )
    title = item.find("div", {"fs-cmsfilter-field": "title"}).text.strip()
    summary = item.find("div", class_="text-intro-pc").text.strip()
    return {
        "url": f"https://metis.io{link}",
        "date": date,
        "author": author,
        "title": title,
        "summary": summary,
    }


def scrape_blog_post(full_url):
    response = requests.get(full_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch blog post page: {full_url}")
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.get_text()
    return content


def generate_date_hash_id(blog):
    # Try to parse the blog's publication date
    try:
        pub_date = datetime.strptime(blog["date"], "%b %d, %Y")
        # Format as DDMMYYYY
        date_str = pub_date.strftime("%d%m%Y")
        
        # Create a hash of the blog's URL using SHA-256 and take the first 8 characters
        hash_str = hashlib.sha256(blog["url"].encode()).hexdigest()[:8]
        # Combine date with the hash
        return f"{date_str}{hash_str}"
    except (ValueError, KeyError):
        # If date is missing or has incorrect format, use hash of the url
        print(f"Warning: Invalid date format for blog: '{blog['url']}'. Using url hash instead.")
        url_hash = hashlib.sha256(blog["url"].encode()).hexdigest()[:16]
        return url_hash


def remove_old_articles(blogs):
    cutoff_date = datetime.now() - timedelta(days=DAYS_TO_KEEP)
    filtered_blogs = []
    
    for blog in blogs:
        try:
            blog_date = datetime.strptime(blog["date"], "%b %d, %Y")
            if blog_date > cutoff_date:
                filtered_blogs.append(blog)
        except ValueError:
            # Keep blogs with unparseable dates (they use url hash ID)
            filtered_blogs.append(blog)
            
    return filtered_blogs


def main():
    # Load existing data
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r") as file:
                data = json.load(file)
            blogs = data.get("blogs", [])
        except (json.JSONDecodeError, KeyError):
            blogs = []
    else:
        blogs = []

    # Fetch blog posts
    posts = fetch_blog_posts()
    new_posts = []

    for item in posts:
        blog = parse_blog_item(item)
        if any(b["url"] == blog["url"] for b in blogs) or len(new_posts) >= MAX_BLOGS:
            break
        blog["id"] = generate_date_hash_id(blog)
        blog["content"] = scrape_blog_post(blog["url"])
        new_posts.append(blog)

    if new_posts:
        blogs = new_posts + blogs  # Add new posts to the top
        blogs = remove_old_articles(blogs)  # Remove old articles if necessary
        try:
            with open(JSON_PATH, "w") as file:
                json.dump(
                    {"latest_id": blogs[0]["id"], "blogs": blogs},  # Set the latest ID
                    file,
                    indent=4,
                )
            print(f"Successfully added {len(new_posts)} new blog posts.")
        except IOError as e:
            print(f"Error writing to JSON file: {e}")


if __name__ == "__main__":
    main()