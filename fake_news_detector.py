import argparse
import sys
import time

from fake_news_tools.text.text_processor import TextProcessor
from fake_news_tools.image.image_processor import ImageProcessor
from fake_news_tools.social_media.social_media_processor import SocialMediaProcessor

"""
from tensorflow.keras.models import load_model

# Load the model from the file
loaded_model = load_model('lstm_model_fake_news.h5')
"""

def show_results(results):
    print("=" * 60)
    print("{:^60}".format("Model Results"))
    print("=" * 60)

    for model, result in results['values'].items():
        if result:  # Si es Fake (True), establece el color rojo
            color = "\033[91m"  # Código de color ANSI para rojo
            prediction = "Fake"
        else:  # Si no es Fake (False), establece el color verde
            color = "\033[92m"  # Código de color ANSI para verde
            prediction = "No Fake"

        print("-" * 60)
        print("{:^60}".format(model))
        print(f"Predicción: {color}{prediction}\033[0m")  # Restablecer el color a normal
        print("-" * 60)

    if results['conclusion']:  # Si es Fake (True), establece el color rojo
        color_conclusion = "\033[91m"  # Código de color ANSI para rojo
        prediction_conclusion = "Fake"
    else:  # Si no es Fake (False), establece el color verde
        color_conclusion = "\033[92m"  # Código de color ANSI para verde
        prediction_conclusion = "No Fake"

    print(f"Conclusion: {color_conclusion}{prediction_conclusion}\033[0m")  # Restablecer el color a normal
    print("=" * 40)

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
    results = handler.process(input_item)

    # Supongamos que tienes un diccionario llamado 'resultados' con los nombres de los modelos como claves
    # y los resultados (True para Fake y False para No Fake) como valores.

    if len(results) < 0:
        sys.exit()
    else:
        show_results(results)
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
