import openai
import requests
import os
import pandas as pd, numpy as np
import json
from typing import Any, Dict, List
import re
from json.decoder import JSONDecodeError
import ast
from ontology_generation.utilsLLM import utilsLLM


class OntologyGen():
    def __init__(self, model_name, deployment_name):

        super().__init__()
        self.model_name = model_name
        self.deployment_name = deployment_name

        self.utils = utilsLLM(model_name, deployment_name)

    def extract_concepts(self, data, topic_name, add_OAconcepts = False):
        temp = list(data['topic_first_word'].unique())
        all_topics = [topic.strip() for group in temp for topic in group.split(',')]

        print("Number of concepts before LLM refinement: ", len(all_topics))
        # print(all_topics)

        role = f"You are an ontology engineer, tasked with helping build an ontology in the domain of {topic_name}"
        prompt = f"Given a list of topics, delete topics that do not belong to the {topic_name} domain:\n"
        prompt += ', '.join(i for i in all_topics) + "\n\n DO NOT ADD NEW TOPICS. Return only the topics, no symbols or extra sentences"
        
        # print("\n")
        # print(prompt)
        
        concepts1 = self.utils.prompt_gpt(role, prompt)

        if isinstance(concepts1, str):
            # Split the string into a list
            concepts1 = concepts1.split(", ")
        
        print("Number of concepts after LLM refinement: ", len(concepts1))
        # print(concepts1)

        if add_OAconcepts == True:
            url = f"https://api.openalex.org/concepts?search={topic_name}"
            concept_request = requests.get(url).json()
            # res = concept_request['results']
            # res[0].get('related_concepts', [])
        
        
            concepts2 = [entry["display_name"] for entry in concept_request['results'][0].get('related_concepts', [])]
            print("Number of concepts related to domain as per OpenAlexDatabase: ", len(concepts2))
            taxonomy_topics = list(set(concepts1 + concepts2))
            taxonomy_topics_df = pd.DataFrame({'topics': taxonomy_topics})
            
            return taxonomy_topics_df
        
        else:
            taxonomy_topics = list(set(concepts1))
            taxonomy_topics_df = pd.DataFrame({'topics': taxonomy_topics})

            return taxonomy_topics_df
    

    def update_taxonomy(self, response, existing_taxonomy):
        try:
            # Assuming the response is a string representation of a dictionary
            # print(response)
            # print(type(response))
            updated_taxonomy = ast.literal_eval(response)
            
            for key, value in updated_taxonomy.items():
                if key in existing_taxonomy:
                    # Key already exists, append values to the existing key
                    if isinstance(existing_taxonomy[key], list):
                        existing_taxonomy[key].extend(value)
                    elif isinstance(existing_taxonomy[key], dict) and isinstance(value, dict):
                        # If both values are dictionaries, update recursively
                        existing_taxonomy[key] = self.process_response(json.dumps(value), existing_taxonomy[key])
                else:
                    # Key doesn't exist, create a new entry
                    existing_taxonomy[key] = value
            
            return existing_taxonomy
        except Exception as e:
            print(f"Error processing response: {e}")
            return existing_taxonomy

    def taxonomy_creation(self, role: str, df, topics_per_batch=20, eg=None, answer=None):
        total_topics = len(df)
        num_batches = total_topics // topics_per_batch + (total_topics % topics_per_batch > 0)
        taxonomy = {}

        for i in range(num_batches):
            if i == 0:
                prompt = "Develop a taxonomy based on the provided topics. Assign each topic to ONE category only. If a topic is unrelated to the domain, categorize it as irrelevant."
            else:
                prompt = '''Based on an existing taxonomy with the following categories:\n'''
                for category in taxonomy.keys():
                    prompt += f"{category}, "
                prompt += '''\nClassify the following topics into existing categories or create a new category if needed.\n'''

            for j in range(topics_per_batch):
                topic_index = i * topics_per_batch + j
                if topic_index < total_topics:
                    prompt += f"{df.loc[topic_index, 'topics']}, "
            
            prompt += '''\n\nThe taxonomy should accurately reflect the domain.\
    Additionally, if any topic is a subset of another, organize them hierarchically.\
    DO NOT include terms like "new category," line breaks, or special characters. Present the taxonomy in the following format, ensuring accurate formatting and parentheses:
    {
        "Category 1": ["topic 1", "topic 5","..."],
        "Category 2": [{"topic 3" : ["topic 6", "topic 8"]}, "topic 9","topic 11", "..."],
        "Category 3": ["topic 2", "topic 4","..."]
    }'''
            flag = True if (eg is not None) and (answer is not None) else False
            res = self.utils.prompt_gpt(role, prompt, with_ex=flag, eg=eg, answer=answer)
            taxonomy = self.update_taxonomy(res, taxonomy)

        return taxonomy

    def prompt_extract(self, role, prompt, with_ex = False, eg = None, answer = None):
        flag = True if (eg is not None) and (answer is not None) else False
        response = self.utils.prompt_gpt(role, prompt, with_ex=flag)
        data = self.utils.extract_and_evaluate_dict(response)

        if isinstance(data, str):
            return ast.literal_eval(data)
        else:
            return data

    # def reorganize_taxonomy(self, taxonomy, main_categories):
    #     # Initialize new_taxonomy with existing categories from the taxonomy
    #     new_taxonomy = {cat: taxonomy.get(cat, {'superTopicOf': {}}) for cat in main_categories if cat in taxonomy}
        
    #     # Add missing categories
    #     for cat in main_categories:
    #         if cat not in new_taxonomy:
    #             new_taxonomy[cat] = {'superTopicOf': {}}
        
    #     for key, value in taxonomy.items():
    #         if key in main_categories:
    #             continue  # Skip if the key is already a main category
            
    #         categorized = False
    #         for main_cat in main_categories:
    #             if key.startswith(main_cat):
    #                 if key == main_cat:
    #                     new_taxonomy[main_cat] = value
    #                 else:
    #                     new_taxonomy[key] = value
    #                     if main_cat in new_taxonomy:
    #                         del new_taxonomy[main_cat]
    #                 categorized = True
    #                 break
            
    #         if not categorized:
    #             new_taxonomy[main_categories[0]]['superTopicOf'][key] = value

    #     return new_taxonomy


    def reorganize_taxonomy(self, taxonomy, main_categories):
        excluded_categories = {'Removed', 'Irrelevant'}
        
        def clean_taxonomy(tax):
            if isinstance(tax, dict):
                return {k: clean_taxonomy(v) for k, v in tax.items() if k not in excluded_categories}
            elif isinstance(tax, list):
                return [clean_taxonomy(item) for item in tax if not (isinstance(item, dict) and any(k in excluded_categories for k in item))]
            else:
                return tax

        # Clean the input taxonomy
        cleaned_taxonomy = clean_taxonomy(taxonomy)
        
        # Initialize new_taxonomy with existing categories from the cleaned taxonomy
        new_taxonomy = {cat: cleaned_taxonomy.get(cat, {'superTopicOf': {}}) 
                        for cat in main_categories 
                        if cat in cleaned_taxonomy}
        
        # Add missing categories
        for cat in main_categories:
            if cat not in new_taxonomy:
                new_taxonomy[cat] = {'superTopicOf': {}}
        
        def categorize(key, value):
            for main_cat in main_categories:
                if key.startswith(main_cat):
                    if key == main_cat:
                        new_taxonomy[main_cat] = value
                    else:
                        new_taxonomy[key] = value
                        if main_cat in new_taxonomy:
                            del new_taxonomy[main_cat]
                    return True
            return False

        def process_item(key, value):
            if not categorize(key, value):
                if isinstance(value, dict) and 'superTopicOf' in value:
                    new_taxonomy[main_categories[0]]['superTopicOf'][key] = value
                else:
                    new_taxonomy[main_categories[0]]['superTopicOf'][key] = {'superTopicOf': value}

        for key, value in cleaned_taxonomy.items():
            if key not in main_categories:
                process_item(key, value)

        return new_taxonomy
    
    def taxonomy_json(self, role, taxonomy):

        prompt = '''Given a taxonomy of concepts''' + str(taxonomy) + '''

            Your task is to return it in the following format: 

            Desired format: 
            {
            "name": "Category 1",
            "children": [
                {
                "name": "Category 3",
                "children": [
                    {"name": "Topic 1"},
                    {"name": "Topic 2"},
                    {"name": "Topic 3"},
                    ...
                ]
                },
                {
                "name": "Category 4",
                "children": [
                    {"name": "Topic 4"},
                    {"name": "Topic 5"},
                    {"name": "Topic 6"},
                    ...
                ]
                },
                ...
            ]
            },
            { 
            "name": "Category 2",
            "children": []
            }

            '''
        json_data = self.utils.prompt_gpt(role, prompt, with_ex = False)
        filename = "output/taxonomy.json"

        with open(filename, 'w') as f: 
            json.dump(json_data, f)

    def extract_topics(self, data):
        topics = set()

        def traverse_topic(topic):
            if isinstance(topic, str):
                topics.add(topic)
            elif isinstance(topic, list):
                for sub_topic in topic:
                    traverse_topic(sub_topic)
            elif isinstance(topic, dict):
                for value in topic.values():
                    traverse_topic(value)

        traverse_topic(data)

        return topics
