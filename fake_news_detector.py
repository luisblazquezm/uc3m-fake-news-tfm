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


def print_result(result, print_each_model=True): 
    logger.info("=" * 70)
    logger.info("{:^60}".format("Model Results"))
    logger.info("=" * 70)

    if print_each_model:
        for model, value in result['values'].items():
            if 'Fake' == value["class"]:  # Si es Fake (True), establece el color rojo
                color = "\033[91m"  # Código de color ANSI para rojo
                prediction = "Fake"
            elif 'Not Fake' == value["class"]:  # Si no es Fake (False), establece el color verde
                color = "\033[92m"  # Código de color ANSI para verde
                prediction = "Not Fake"
            else:
                color = "\033[90m"  # Código de color ANSI para gris
                prediction = "Not available for the given input"

            if value["accuracy"] >= 75:  # Si es Fake (True), establece el color rojo
                color_accuracy = "\033[92m"  # Código de color ANSI para verde
            elif value["accuracy"] >= 50 and value["accuracy"] < 75:  
                color_accuracy = "\033[38;5;226m"  # Código de color ANSI para amarillo
            elif value["accuracy"] >= 25 and value["accuracy"] < 50: 
                color_accuracy = "\033[38;5;208m"  # Código de color ANSI para naranja
            elif value["accuracy"] > 0 and value["accuracy"] < 25:  
                color_accuracy = "\033[91m"  # Código de color ANSI para rojo
            else:
                color_accuracy = "\033[90m"  # Código de color ANSI para gris

            accuracy = "{:.2f}".format(value["accuracy"])
            logger.info("-" * 70)
            logger.info("{:^60}".format(model))
            logger.info(f"Prediction: {color}{prediction}\033[0m")  # Restablecer el color a normal
            logger.info(f"Accuracy of classification: {color_accuracy}{accuracy} %\033[0m")  # Restablecer el color a normal
            logger.info("-" * 70)

    if 'Fake' == result['conclusion']:  # Si es Fake (True), establece el color rojo
        color_conclusion = "\033[91m"  # Código de color ANSI para rojo
        prediction_conclusion = "Fake"
    elif 'Not Fake' == result['conclusion']:  # Si no es Fake (False), establece el color verde
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

def show_results(results, num_results, input_item=""):
    
    if num_results > 1:
        logger.info("=" * 70)
        logger.info("{:^60}".format("Results for top 25 news items for search keyword '" + input_item + "'"))
        logger.info("=" * 70)

        for result in results:  
            url = result["url"]      
            logger.info("-" * 70)
            logger.info(f"Original source: {url}")  # Restablecer el color a normal
            logger.info("-" * 70)

            if result["prediction"] is not None:
                if len(result["prediction"].keys()) > 0:
                    print_result(result=result["prediction"], print_each_model=False)
                else:
                    logger.info(f"Text not available.")
            else:
                    logger.info(f"Text not available.")

        logger.info("*" * 70)

        print(results)

        total_tmp = [result["prediction"] for result in results if result["prediction"] is not None]
        total = [item["conclusion"] for item in total_tmp if "conclusion" in item]

        logger.info(f"Total news items: {len(total)}")
        logger.info(f"\033[91mNews detected as 'Fake': {total.count('Fake')}\033[0m")
        logger.info(f"\033[92mNews detected as 'True': {total.count('True')}\033[0m")
        logger.info("*" * 70)
    else:
        print_result(result=results, print_each_model=True)

def main():
    # Start counting time for the task to be completed
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Fake News Detector")

    # Arguments for text and link
    parser.add_argument("--text", type=str, help="News text")
    parser.add_argument("--link", type=str, help="News link")

    # Argument for automated web search
    parser.add_argument("--search_keyword", type=str, help="Keyword for automated web search (Fast Check Explorer)")

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
    elif args.search_keyword:
        handler = SocialMediaProcessor(type_process="search_keyword", logger=logger)
        input_item = args.search_keyword
    else:
        logger.error("You must provide text, link, image, social media account, URL, or search keyword.")
        sys.exit()

    # Run and process input to the script
    is_success, num_results, result = handler.process(input_item)

    # Supongamos que tienes un diccionario llamado 'resultados' con los nombres de los modelos como claves
    # y los resultados (True para Fake y False para No Fake) como valores.

    if not is_success:
        logger.error("No results found for the available models")
        sys.exit()
    else:
        show_results(result, num_results, input_item)
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        

if __name__ == "__main__":
    """
    Text Analysis: 
        python fake_news_detector.py --text "This is a news article about a recent event."

    URL Analysis:
        python fake_news_detector.py --link "https://www.example.com/fake-news-article"

    Automated Web Search:
        python fake_news_detector.py --search_keyword "fake news"
    """
    main()
