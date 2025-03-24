import pandas as pd
import logging

logger = logging.getLogger()


class Transform_Data:
    def __init__(self, type, input_file_path=None, output_dir=None):
        self.type = type
        if not input_file_path:
            self.input_file_path = '/work/pi_wenlongzhao_umass_edu/6/datasets/'+type
        else:
            self.input_file_path = input_file_path
        if not output_dir:
            self.output_file_path = ''
            # self.output_file_path = '/work/pi_wenlongzhao_umass_edu/6/datasets/'
        else:
            self.output_file_path = output_dir
        # add provision for different file types?

    def type_to_prompt_mapper(self, type):
        '''incorporate specific tranformations according to the dataset.

        prompt is the specific directive to the model, incited_response is the trigger word for response, columns include the column names for the dataset originally to extract and perform the right tranformations.
        '''
        if type == 'financial_sentiment_analysis':  # or lets make it sentiment analysis
            prompt = "What is the sentiment of the below sentence?"
            incited_response = 'Sentiment'
            columns = ['Sentence', 'Sentiment']

        elif type == 'meta_math_qa':
            prompt = ""
            incited_response = 'Answer'
            columns = ['query', 'response']

        elif type == 'paws':
            prompt = "Are the following two sentences paraphrases of each other?"
            incited_response = 'Answer'
            columns = ['Sentences', 'label']

        elif type == 'cot':
            prompt = ""
            incited_response = "Response"
            columns = ['prompt', 'response']

        elif type == 'glaive_code':
            prompt = ""
            incited_response = "Answer"
            columns = ['question', 'answer']

        elif type == 'goat':
            prompt = ""
            incited_response = "Answer"
            columns = ['instruction', 'answer']

        elif type == 'MagicCoder':
            prompt = ""
            incited_response = "Answer"
            columns = ['problem', 'solution']

        elif type == 'imdb':
            prompt = "What is the sentiment of the below sentence?"
            incited_response = "Sentiment"
            columns = ['text', 'label']

        elif type == 'flipkart':
            prompt = "What is the sentiment of the below sentence? Positive, negative or neutral?"
            incited_response = "Sentiment"
            columns = ['input', 'output']
        
        elif type == 'pile':
            prompt = "Is the following sentence toxic?"
            incited_response = "Toxicity"
            columns = ['text', 'toxicity']

        elif type == 'gsm8k':
            prompt = ""
            incited_response = 'Answer'
            columns = ['question', 'answer']


        elif type == 'hindi_math_reasoning':
            prompt = ""
            incited_response = 'Answer'
            columns = ['input', 'output']
        
        elif type == 'amazon':
            prompt = "What is the sentiment of the following amazon review? Provide a rating from 1 to 5 stars, where 1 is very negative and 5 is very positive."
            incited_response = "Rating"
            columns = ['review_text','class_index']
            
        return prompt, incited_response, columns
 
    def transform(self, test):
        # this function assumes csv inputs
        df = pd.read_csv(self.input_file_path)
        prompt, incited_response, columns = self.type_to_prompt_mapper(self.type)
        output_file_path = self.output_file_path+'Transformed_' + \
            self.input_file_path.split('/')[-1].split('.')[0]+'.csv'

        if not test:

            try:

                df["Text"] = df.apply(lambda row: (
                    f"<|begin_of_text|>{prompt}\n"
                    f"{row[columns[0]]}\n"
                    f"{incited_response}:\n"
                    f"{row[columns[1]]}<|end_of_text|>"
                ), axis=1)

            except Exception as e:
                logger.error(
                    f"Could not tranform the train split of dataset. Error: {e}")

        else:
            try:

                df["Text"] = df.apply(lambda row: (
                    f"<|begin_of_text|>{prompt}\n"
                    f"{row[columns[0]]}\n"
                    f"{incited_response}:\n"
                ), axis=1)

            except Exception as e:
                logger.error(
                    f"Could not transform the test split of dataset. Error: {e}")

        try:

            df.to_csv(output_file_path, index=False)
            print(
                f'Formatted CSV for type {self.type} saved in {output_file_path}. Returning tranformed dataframe object.')
            return df

        except Exception as e:
            print(
                f"Could not save the file in {output_file_path} post tranformation. Error: {e}")
