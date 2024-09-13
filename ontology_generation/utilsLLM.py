import openai
import requests
import os
import pandas as pd, numpy as np
import json
from typing import Any, Dict, List
import re
from json.decoder import JSONDecodeError
import ast

openai.api_type = "azure"
openai.base_url = "https://texttospeech.openai.azure.com/openai/deployments/gpt_chat_test_preview/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ["OPENAI_API_KEY"]

class utilsLLM():
    def __init__(self, model_name, deployment_name):
        self.model_name = model_name
        self.deployment_name = deployment_name

    def process_response(self, response, dtax):
        if self.is_json(response):
            # Response is JSON, process normally
            data = json.loads(response)
            
            for d in data:
                # Extract key and values
                key = list(d.keys())[0]
                values = d[key]
                
                if key in dtax:
                    # Append values if key exists
                    dtax[key].extend(values)
                else:
                    # Add new key-value 
                    dtax[key] = values
                    
        else:
            # Response is plain text
            pattern = r'"([^"]+)"\: \[([^\]]+)'
            matches = re.findall(pattern, response)
            
            for topic, text_values in matches:
                # Convert string to list
                values = text_values.strip()[1:-1].replace('"', '').split(",")
                # Remove whitespace
                values = [v.strip() for v in values]
                
                if topic in dtax:
                    # Append values if key exists
                    dtax[topic].extend(values)
                else:
                    # Add new key-value 
                    dtax[topic] = values
                
        return dtax

    def is_json(self, text):
        try: 
            json.loads(text)
            return True
        except JSONDecodeError:
            return False

    def prompt_gpt(self, role, prompt, with_ex=False, eg=None, answer=None):
        try:

            # Create a list to hold the messages
            messages = [{"role": "system", "content": role}]
            
            # Add eg and answer if include_eg_answer is True
            if with_ex:
                messages.extend([
                    {'role': 'user', 'content': eg},
                    {'role': 'assistant', 'content': answer}
                ])

            messages.append({"role": "user", "content": prompt})

            # Get response from ChatGPT
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
           
        except Exception as e:
            print(f"Error while accessing ChatGPT")
            print(e)
            return None
        
        return answer
        
    def extract_and_evaluate_dict(self, dictd):
        # Extract content between triple backticks
        match = re.search(r'```json\n(.+?)\n```', dictd, re.DOTALL)

        if match:
            python_dict_str = match.group(1)
            try:
                # Safely evaluate the Python dictionary string
                ddata = ast.literal_eval(python_dict_str)
                if isinstance(ddata, (list, dict)):
                    return ddata
                elif isinstance(ddata, set):
                    return list(ddata)
                else:
                    print("The extracted content is not a list or dictionary. It is: ", type(ddata))
                    return dictd
            except (SyntaxError, ValueError) as e:
                print(f"Error evaluating Python dictionary: {e}")
                return dictd
        else:
            return dictd


