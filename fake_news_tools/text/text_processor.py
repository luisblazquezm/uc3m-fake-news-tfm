from collections import Counter

from fake_news_tools.text.models.model_abstraction import ModelAbstraction

class TextProcessor: 
    # Add code for text analysis (Step 1 - NLP)

    def __init__(self, type_process, logger):
        self.__process = type_process
        self.__logger = logger

    def process(self, text: str):

        # Text process
        result = self.__process_text(text=text)

        """try:
            self.__analyze_fact_check_api(text=text)
        except Exception as e:
            return -1

        try:
            self.__analyze_sentiment_api(text=text)
        except Exception as e:
            return -1"""

        return True, result

    def __process_text(self, text: str):

        # Apply different models
        final_result = { 'conclusion': "", 'values': {} }
        results = []
        for subclass in ModelAbstraction.__subclasses__():
            instance = subclass()  # Instantiate an object of the subclass
            prediction_types = instance.get_predictions()
            
            result = "Not available for this model"

            self.__logger.debug("Running model '" + instance.get_method() + "'")
            # Check if the model is available for title or text prediction
            if self.__process in prediction_types:
                result = instance.predict(data=text)  # Call predict method of the instance

            results.append({
                'method': instance.get_method(),
                'result': result,
                'score': []
            })

        # Prepare result
        final_result['values'] = {result['method']:result['result'] for result in results}

        # Get final conclusion result 'Fake' or 'Not fake'
        counter = Counter([result['result'] for result in results])
        most_common_value = counter.most_common(1)[0][0]
        final_result['conclusion'] = most_common_value

        return final_result

    def __analyze_fact_check_api(self, text: str):
        # Add code for fact-checking API integration (Step 6 - API Integration)
        api_key = "YOUR_FACT_CHECK_API_KEY"
        self.__logger("Fact-Checking API:")
        # Code for making the API request and getting results
    
    def __analyze_sentiment_api(self, text: str):
        # Add code for sentiment analysis API integration (Step 6 - API Integration)
        api_key = "YOUR_SENTIMENT_ANALYSIS_API_KEY"
        self.__logger("Sentiment Analysis API:")
        # Code for making the API request and getting results

