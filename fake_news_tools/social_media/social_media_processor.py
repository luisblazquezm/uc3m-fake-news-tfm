import requests
import time
import random
from bs4 import BeautifulSoup

from fake_news_tools.text.text_processor import TextProcessor
from fake_news_tools.social_media.utils import USER_AGENT_LIST, FAST_CHECK_API_KEY

REQUEST_MIN_TIME_WAIT = 15
REQUEST_MAX_TIME_WAIT = 30
NUM_ITEMS = 25
MAIN_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search?key={api_key}&query={query}&languageCode=en-US&pageSize={num_items}"

class SocialMediaProcessor: 
    # Add code for text analysis (Step 1 - NLP)

    def __init__(self, type_process, logger):
        self.__process = type_process
        self.__logger = logger
        self.__text_processor = TextProcessor(type_process="text", logger=logger)

    def process(self, content: str):
        result = {}
        num_results = -1

        if "search_keyword" == self.__process:
            result = self.__search_web(keyword=content)
            num_results = len(result)
        elif "analyze_url" == self.__process:
            result = self.__analyze_url(url=content)
            num_results = 1
        else:
            self.__logger.error("Option given is not available")
            return False, 0, {}

        if result is None:
            return False, 0, {}
        else:
            return True, num_results, result

    def __analyze_url(self, url: str):
        # Add code for social media analysis (Step 3 - Social Media Analysis)
        self.__logger.info(f"URL Analysis: {url}")

        # Wait some time to make another request
        time.sleep(random.randint(REQUEST_MIN_TIME_WAIT, REQUEST_MAX_TIME_WAIT))
        
        # Choose a random user agent
        user_agent = random.choice(USER_AGENT_LIST)

        try:
            #self.__logger.debug("Extracting content from '" + url + "'")
            response = requests.get(url, headers={'User-Agent': user_agent})
            response.raise_for_status()

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'lxml')

                article_element = soup.find('article')

                text = ""
                if article_element:
                    # Obtener el texto dentro del elemento <article>
                    news_content = article_element.get_text()
                    text = news_content.strip()
                    text = text.replace('\n', '').replace('\r', '').replace('  ', '')
                else:
                    # Find all paragraphs and join their text
                    paragraphs = soup.find_all('p')

                    if len(paragraphs) > 0:
                        article_text = " ".join(p.get_text().replace('\n', '').replace('\r', '').replace('  ', '') for p in paragraphs)
                        text = article_text.strip()
                    else: 
                        return None

                if text != "":
                    success, num_result, result =  self.__text_processor.process(text)
                    return result
                else:
                    self.__logger.error("No text extracted from the url given. Please, try with another link.")
                    return {}
                
            else:
                self.__logger.error("Error:", response.status_code)
                self.__logger.error(response.text)
                return None
                
        except requests.exceptions.RequestException as e:
            self.__logger.error("An error ocurred during request:", e)
            return None
        except Exception as e:
            self.__logger.error("An exception ocurred on URL extraction: ", e)
            return None
            
    def __search_web(self, keyword: str):
        # Add code for automated web search (Step 5 - Automated Web Search)
        self.__logger.info(f"Automated Web Search with Keyword: {keyword}")

        url = MAIN_URL.replace('{api_key}', FAST_CHECK_API_KEY).replace('{query}', keyword).replace('{num_items}', str(NUM_ITEMS))
        headers = {
        'Accept': 'application/json'
        }

        # Wait some time to make another request
        time.sleep(random.randint(REQUEST_MIN_TIME_WAIT, REQUEST_MAX_TIME_WAIT))
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            if response.status_code == 200:
                data = response.json()
            else:
                self.__logger.error("Error:", response.status_code)
                self.__logger.error(response.text)

            if len(data["claims"]) > 0:
                return [{"url": item["url"], "prediction": self.__analyze_url(url=item["url"])} for claim in data["claims"] for item in claim["claimReview"]]
            else:
                self.__logger.info("No results found for the given query '" + keyword + "'")
                return None
                
        except requests.exceptions.RequestException as e:
            self.__logger.error("An error ocurred during request:", e)
            return None
        except Exception as e:
            self.__logger.error("An exception ocurred on URL extraction: ", e)
            return None

