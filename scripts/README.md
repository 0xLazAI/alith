# Knowledge Scraping: Blogs

This project is designed to scrape and maintain a knowledge base from the Metis blog. The system efficiently fetches new blog posts, extracts their content, and saves the data in a structured JSON format for future use by an AI Agent's knowledge base.

## Features

* **Daily Automation**: Automatically scrapes the latest blog posts once per day using GitHub Actions.
* **Unique Identification**: Generates unique IDs for blog posts using a combination of the date and URL hash.
* **Customizable Retention**: Old blog articles are automatically removed after a configurable period.
* **Content Storage**: Stores blog content in a JSON file located in the `knowledge/metis` directory.

## Prerequisites

1. **Python Version**: Ensure Python 3.x is installed.
2. **Dependencies**: The script relies on the following Python libraries:
   * `requests`
   * `beautifulsoup4`
   * `json`

## Key Files

* `addknowledge_blog.py`: Main script that performs the scraping, processing, and storage of blog data.
* **JSON Output**: The script saves the scraped data in a file located at `knowledge/metis/blog.json`.

## How It Works

1. **Fetch Blog Posts**:
   * Scrapes the Metis Blog to retrieve new blog posts.
2. **Process Blog Data**:
   * Extracts key details like title, date, author, summary, and full content.
   * Assigns a unique ID to each blog post based on its URL and the current date.
3. **Update Knowledge Base**:
   * Adds new posts to the JSON file while retaining recent articles based on a configurable retention period.

## Usage

### General

**Automatic run**
you dont really need to do anything. The Github Actions flow is running and updating the file anyway. Only point your [Alith Agent](https://alith.lazai.network/docs/features/knowledge)
 to get knowledge from the target location *for blogs it's `knowledge/metis/blog.json`) and you are goof to go. If you need to customize it -> enjoy: 

### Local Script Execution

1. **Install Dependencies**:
```bash
pip install requests beautifulsoup4
```

2. **Run the Script**:
```bash
python scripts/addknowledge_blog.py
```

### GitHub Actions Workflow

The process is automated through the `knowledge_scraping.yml` file located in `.github/workflows`. The workflow:
* Runs daily at midnight UTC.
* Automates setup (dependencies, environment).
* Executes the script to scrape and commit updated knowledge.

To manually trigger the workflow:
1. Navigate to the "Actions" tab in the repository.
2. Select the workflow and click "Run workflow."

## Configuration

### Constants

* `BLOG_URL`: URL of the blog to scrape (`https://metis.io/blog` by default).
* `MAX_BLOGS`: Maximum number of new blogs to process at a time (default is 10).
* `DAYS_TO_KEEP`: Retention period for blog articles in days (default is set very high for minimal maintenance).

### JSON File Location

The scraped data is stored in `knowledge/metis/blog.json`. Modify `JSON_PATH` in the script to change the storage location if required.

## Example JSON Output

The stored JSON data follows this structure:

```json
{
    "latest_id": "DDMMYYYYabcdef12",
    "blogs": [
        {
            "id": "DDMMYYYYabcdef12",
            "url": "https://metis.io/some-blog-post",
            "date": "Mar 12, 2025",
            "author": "Author Name",
            "title": "Blog Title",
            "summary": "Short summary of the blog",
            "content": "Full blog post content..."
        }
    ]
}
```

## Error Handling

* **Network Failures**: The script raises exceptions if the blog or individual posts fail to load.
* **Malformed Data**: Blogs with missing or incomplete data are skipped.
* **File Write Errors**: Logs an error message if the JSON file cannot be written.

## Future Enhancements

* Extend scraping functionality to include **Metis Gov** data.
* Add logging for better traceability of errors and processes.
* Support other blog formats or platforms.
