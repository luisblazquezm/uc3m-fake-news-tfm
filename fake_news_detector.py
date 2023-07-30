import argparse
import sys
import time

from fake_news_tools.text.text_processor import TextProcessor
from fake_news_tools.image.image_processor import ImageProcessor
from fake_news_tools.social_media.social_media_processor import SocialMediaProcessor

def main():
    # Start counting time for the task to be completed
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Fake News Detector")

    # Arguments for text and link
    parser.add_argument("--text", type=str, help="News text")
    parser.add_argument("--link", type=str, help="News link")

    # Argument for image
    parser.add_argument("--image", type=str, help="Path to news image")

    # Arguments for social media analysis
    parser.add_argument("--social_media", type=str, help="Social media account name")

    # Argument for URL verification
    parser.add_argument("--url", type=str, help="News URL")

    # Argument for automated web search
    parser.add_argument("--search_keyword", type=str, help="Keyword for automated web search")

    args = parser.parse_args()

    # Processing based on the input type
    input_item = ""
    if args.text:
        handler = TextProcessor()
        input_item = args.text
    elif args.image:
        handler = ImageProcessor()
        input_item = args.image
    elif args.link:
        handler = SocialMediaProcessor(type_process="analyze_url")
        input_item = args.link
    elif args.social_media:
        handler = SocialMediaProcessor(type_process="analyze_social_media")
        input_item = args.social_media
    elif args.url:
        handler = SocialMediaProcessor(type_process="verify_url")
        input_item = args.url
    elif args.search_keyword:
        handler = SocialMediaProcessor(type_process="search_keyword")
        input_item = args.search_keyword
    else:
        print("[ERROR] You must provide text, link, image, social media account, URL, or search keyword.")
        sys.exit()

    # Run and process input to the script
    result = handler.process(input_item)

    if result < 0:
        sys.exit()
    else:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[SUCCESS] Total time taken: {total_time:.2f} seconds")
        

if __name__ == "__main__":
    """
    Text Analysis: 
        python fake_news_detector.py --text "This is a news article about a recent event."

    URL Analysis:
        python fake_news_detector.py --link "https://www.example.com/fake-news-article"

    Image Analysis:
        python fake_news_detector.py --image "path/to/image.jpg"

    Social Media Analysis:
        python fake_news_detector.py --social_media "example_account"

    URL Verification:
        python fake_news_detector.py --url "https://www.example.com/news-article"

    Automated Web Search:
        python fake_news_detector.py --search_keyword "fake news"
    """
    main()
