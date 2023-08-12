import argparse
import sys
import time
import logging
import colorlog

from fake_news_tools.text.text_processor import TextProcessor
from fake_news_tools.social_media.social_media_processor import SocialMediaProcessor

# Configure logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s - %(message)s",
    log_colors={
        'DEBUG': 'reset',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)  # Set the logger level to INFO

def show_results(results):
    logger.info("=" * 70)
    logger.info("{:^60}".format("Model Results"))
    logger.info("=" * 70)

    for model, result in results['values'].items():
        if 'Fake' == result:  # Si es Fake (True), establece el color rojo
            color = "\033[91m"  # Código de color ANSI para rojo
            prediction = "Fake"
        elif 'Not Fake' == result:  # Si no es Fake (False), establece el color verde
            color = "\033[92m"  # Código de color ANSI para verde
            prediction = "Not Fake"
        else:
            color = "\033[90m"  # Código de color ANSI para verde
            prediction = "Not available for the given input"

        logger.info("-" * 70)
        logger.info("{:^60}".format(model))
        logger.info(f"Prediction: {color}{prediction}\033[0m")  # Restablecer el color a normal
        logger.info("-" * 70)

    if 'Fake' == results['conclusion']:  # Si es Fake (True), establece el color rojo
        color_conclusion = "\033[91m"  # Código de color ANSI para rojo
        prediction_conclusion = "Fake"
    elif 'Not Fake' == results['conclusion']:  # Si no es Fake (False), establece el color verde
        color_conclusion = "\033[92m"  # Código de color ANSI para verde
        prediction_conclusion = "Not Fake"
    else:
        color_conclusion = "\033[90m"  # Código de color ANSI para verde
        prediction_conclusion = "Not available for the given input"

    print("")
    print("")
    print("")
    logger.info("=" * 70)
    logger.info(f"Conclusion: {color_conclusion}{prediction_conclusion}\033[0m")  # Restablecer el color a normal
    logger.info("=" * 70)

def main():
    # Start counting time for the task to be completed
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Fake News Detector")

    # Arguments for text and link
    parser.add_argument("--text", type=str, help="News text")
    parser.add_argument("--link", type=str, help="News link")

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
        print("")
        print("")
        logger.info("Choose an option:")
        logger.info("1. Analyze news title")
        logger.info("2. Analyze news text")
        choice = input("Enter your choice (1/2): ")

        if choice == "1":
            # Analyze the title
            print("")
            print("")
            logger.info("----- Analyzing news title")
            handler = TextProcessor(type_process="title", logger=logger)
            input_item = args.text
        elif choice == "2":
            # Analyze the title
            print("")
            print("")
            logger.info("------- Analyzing news content")
            handler = TextProcessor(type_process="text", logger=logger)
            input_item = args.text
        else:
            logger.error("Invalid choice. Please select either 1 or 2.")
            sys.exit()
    elif args.link:
        handler = SocialMediaProcessor(type_process="analyze_url", logger=logger)
        input_item = args.link
    elif args.social_media:
        handler = SocialMediaProcessor(type_process="analyze_social_media", logger=logger)
        input_item = args.social_media
    elif args.url:
        handler = SocialMediaProcessor(type_process="verify_url", logger=logger)
        input_item = args.url
    elif args.search_keyword:
        handler = SocialMediaProcessor(type_process="search_keyword", logger=logger)
        input_item = args.search_keyword
    else:
        logger.error("You must provide text, link, image, social media account, URL, or search keyword.")
        sys.exit()

    # Run and process input to the script
    is_success, result = handler.process(input_item)

    # Supongamos que tienes un diccionario llamado 'resultados' con los nombres de los modelos como claves
    # y los resultados (True para Fake y False para No Fake) como valores.

    if not is_success:
        logger.error("No results found for the available models")
        sys.exit()
    else:
        show_results(result)
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        

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
