import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json

# GOOGLE_API_KEY = 'AIzaSyCzH7EiW_ZaCpxv6lCR5PB33CQiG1fSvnQ'
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-pro')

# import time
# import json

# def generate_keywords(content, category):

#     # Construct the prompt for Gemini Pro to generate keywords
#     prompt = f"""
#     Generate 10 **concise and relevant keywords** in list format within [] directly related to the following content: {content}. The keywords should represent the main topics, ideas, and entities discussed in the content. Include the main category '{category}' as one of the keywords. Ensure the keywords are **short, specific, and focused** on the core themes of the content that could be useful to build a recommendation system.

#     Avoid generating headings, titles, or overly general terms. Focus on specific concepts, people, events, locations, and keywords that provide a meaningful summary of the content.
#     Ensure you follow strict list format.
#     Content:
#     {content}

#     Example:
#     Content: "India is facing severe environmental challenges due to the rapid industrialization and urbanization in many regions."

#     Output: ["india", "environment", "sustainable development", "urbanization", "pollution", "swachh bharat", "government", "challenges", "industrialization", "cleanliness"]

#     Note: The keywords should be strictly one word in list format within [] the category as the first keyword itself.
#     """
    
#     try:
#         # Call Gemini Pro's API to generate the keywords
#         response = model.generate_content([prompt, category, content])
#         print("Raw Response Text:", response.text)

#         # Clean and parse the response text
#         response_text = response.text.strip()

#         if response_text.startswith("[") and response_text.endswith("]"):
#             # Handle JSON-like format
#             try:
#                 keywords = json.loads(response_text)
#             except json.JSONDecodeError:
#                 raise ValueError("Failed to parse response as JSON.")
        
#         elif '*' in response_text:  # Check for bullet format
#             # Handle bullet points format, clean and split
#             keywords = [line.strip('*').strip() for line in response_text.splitlines() if line.strip().startswith('*')]
        
#         else:
#             # Handle generic comma-separated values format
#             keywords = [kw.strip() for kw in response_text.split(',')]

#         # Ensure the keywords are properly stripped and in list format
#         keywords = [keyword.strip().strip('"').strip("'") for keyword in keywords]

#         # Ensure there are exactly 10 keywords (if not, fallback to category)
#         if len(keywords) < 10:
#             keywords = [category] * 10
#         elif len(keywords) > 10:
#             keywords = keywords[:10]

#     except Exception as e:
#         print(f"Error generating keywords: {e}")
#         # Fallback to 10 instances of the category if generation fails
#         keywords = [category] * 10

#     return keywords

# def add_keywords_to_dataset(combined_df, time_gap=5):

#     keywords_list = []

#     for index, row in combined_df.iterrows():
#         content = row['content']
#         category = row['category']

#         # Generate keywords for the content
#         keywords = generate_keywords(content, category)
#         keywords_list.append(keywords)

#         # Add a time gap between requests to avoid hitting rate limits
#         time.sleep(time_gap)

#     # Add the keywords column to the dataset
#     combined_df['keywords'] = keywords_list
#     return combined_df



def scrape_category(categories, output_csv, pages=20):

    all_data = []

    # Loop through each category
    for category, csv_file in categories.items():
        print(f"Scraping {category.capitalize()}...")

        base_url = f"https://www.thehindu.com/{category}/?page="
        
        # Initialize lists to store data
        titles, subtitles, authors, contents, article_urls = [], [], [], [], []

        # Loop through pages for each category
        for page_number in range(1, pages + 1):
            print(f"Scraping {category.capitalize()} - Page {page_number}...")

            # Construct the URL for the page
            main_page_url = f"{base_url}{page_number}"
            response = requests.get(main_page_url)
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all article links on the page
            article_links = []
            for element in soup.find_all("div", class_="element row-element"):
                link = element.find("a", href=True)
                if link:
                    article_links.append(link['href'])

            # Remove duplicates
            article_links = list(set(article_links))

            # Scrape each article link
            for article_url in article_links:
                try:
                    # Request the article page
                    article_response = requests.get(article_url)
                    article_soup = BeautifulSoup(article_response.content, "html.parser")

                    # Extract title, subtitle, and author
                    title = article_soup.find("h1", class_="title").text.strip() if article_soup.find("h1", class_="title") else "No Title"
                    subtitle = article_soup.find("h2", class_="sub-title").text.strip() if article_soup.find("h2", class_="sub-title") else "No Subtitle"
                    author = article_soup.find("a", class_="person-name lnk").text.strip() if article_soup.find("a", class_="person-name lnk") else "Miscellaneous"

                    # Extract content (from the first two <p> tags inside specific div)
                    content = ""
                    content_div = article_soup.find("div", class_=["articlebodycontent", "col-xl-9 col-lg-12 col-md-12 col-sm-12 col-12"])
                    if content_div:
                        paragraphs = content_div.find_all("p", limit=2)
                        content = " ".join([p.text.strip() for p in paragraphs])
                        
                    combined_content = f"{title} {subtitle} {content}"

                    # Append data to lists
                    titles.append(title)
                    subtitles.append(subtitle)
                    authors.append(author)
                    contents.append(combined_content)
                    article_urls.append(article_url)

                    print(f"Scraped article: {title}")
                except Exception as e:
                    print(f"Failed to scrape {article_url}: {e}")

        # Create a DataFrame for the current category
        data = {
            'title': titles,
            'subtitle': subtitles,
            'author': authors,
            'content': contents,
            'link': article_urls
        }
        df = pd.DataFrame(data)

        # Add the 'category' column during concatenation
        df['category'] = category.split('/')[1] if '/' in category else category

        # Append the DataFrame to the list
        all_data.append(df)

        # Save the data to individual CSV file
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Data for {category} saved to '{csv_file}'.")

    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['unique_id'] = combined_df.index + 1

    # combined_df = add_keywords_to_dataset(combined_df,time_gap=5)

    # Save the combined dataset to the final CSV file
    combined_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Combined dataset saved as '{output_csv}'.")

# Example usage for different categories
if __name__ == "__main__":
    categories = {
        "sci-tech/science": "the_hindu_articles_science_pages.csv",
        "sci-tech/health": "the_hindu_articles_health_pages.csv",
        "news/national": "the_hindu_articles_india_pages.csv",
        "news/international": "the_hindu_articles_world_pages.csv",
        "entertainment/movies":"the_hindu_articles_movies_pages.csv",
        "business":"the_hindu_articles_business_pages.csv"
    }

    output_csv = "combined_news_articles.csv"
    scrape_category(categories, output_csv, pages=20)
