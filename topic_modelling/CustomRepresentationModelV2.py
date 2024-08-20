from bertopic.representation._base import BaseRepresentation
import openai
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import time
import copy
import numpy as np
from typing import Tuple, List, Dict, Union, Any


openai.api_key = "OPENAI_API_KEY"
openai.api_base = ""
openai.api_type = ''
openai.api_version = '2023-05-15'
deployment_name='name'


class CustomRepresentationModelV2(BaseRepresentation):
    def __init__(self, nr_docs=10, messages=None, verbose=False, temperature=1, top_p=1, max_tokens=1000):
        """
        Init function of the Representation model. Messages variable is set to default value if not defined. Use the words [KEYWORDS] and [DOCUMENTS] to include keywords of a topic and representative 
        documents to the messages object. nr_docs is the number of documents ChatGPT gets as an input if [DOCUMENTS] is included in the messages object.
        temperature and top_p are parameters of the ChatGPT API to handle the randomness of the answers (see https://platform.openai.com/docs/api-reference/completions/create). Max_tokens is the maximum tokens of the answer.
        If the answer includes more tokens it get cut.
        """
        self.nr_docs = nr_docs
        self.topic_labels = None
        self.verbose = verbose
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repr_docs_dict = {}
        self.topic_model = None
        self.topics = None
        self.probs = None

        if messages is None:
            self.messages = [
                {"role": "system", "content":
                 "You are a topic representation creator model"},
                {"role": "user", "content":
                 """I have a topic that contains the following documents: [DOCUMENTS]
                The topic is described by the following keywords: [KEYWORDS]
                Based on the information above, extract a short topic label in the following format: topic: <topic label>
                """
                 }
            ]

        else:
            self.messages = messages
        self.backup_messages = copy.deepcopy(self.messages)

    def extract_topics(self, topic_model: Any, documents: List[str], c_tf_idf: Any, topics: Dict[int, Any]) -> Dict[int, Any]:
        """
        Extracts topics based on given documents and a topic model.

        Args:
        - topic_model: The BERTopic model or other topic model used.
        - documents: A list of documents.
        - c_tf_idf: The c-TF-IDF matrix.
        - topics: Candidate topics as calculated with c-TF-IDF.

        Returns:
        - Dictionary containing updated topics.
        """
        self.topic_model = topic_model

        self.repr_docs_dict = self.create_representative_docs(
            documents=documents, max_docs=self.nr_docs)

        updated_topics = {}
        self._verbose_print('Get ChatGPT representations')

        for topic, docs in self.repr_docs_dict.items():
            messages = self._update_messages(
                docs, topic, topics, self.messages)
            self._verbose_print(f'Topic {topic}:\n{messages}')
            label, successful = self._process_topic_with_chatgpt(
                topic, messages, topics)

            if successful:
                updated_topics[topic] = [(label, 1)] + [("", 0)
                                                        for _ in range(9)]
            else:
                updated_topics[topic] = topics[topic]

        self.topic_labels = updated_topics
        return updated_topics

    def _verbose_print(self, message: str) -> None:
        """
        Prints a message if verbose mode is enabled.

        Args:
        - message: The message to print.
        """
        if self.verbose:
            print(message)

    def _process_topic_with_chatgpt(self, topic: int, messages: List[str], topics: Dict[int, Any]) -> Tuple[Union[str, None], bool]:
        """
        Processes a topic using ChatGPT.

        Args:
        - topic: The topic number.
        - messages: List of messages corresponding to the topic.
        - topics: Candidate topics as calculated with c-TF-IDF.

        Returns:
        - Tuple containing the label (or None if unsuccessful) and a boolean indicating success.
        """
        label = self._try_chat_completion(messages)
        if label:
            return label, True
        else:
            print(
                f'Error while accessing ChatGPT for topic {topic}... Trying again after 20 seconds...')
            time.sleep(20)
            label = self._try_chat_completion(
                messages, handle_invalid_request=True, topic=topic, topics=topics)
            return (label, True) if label else (None, False)

    def _try_chat_completion(self, messages: List[str], handle_invalid_request: bool = False, topic: Union[int, None] = None, topics: Union[Dict[int, Any], None] = None) -> Union[str, None]:
        """
        Attempts to fetch a topic label using ChatCompletion.

        Args:
        - messages: List of messages to be processed.
        - handle_invalid_request: Whether to handle an invalid request by adjusting the message.
        - topic: The topic number (used for verbose output and error handling).
        - topics: Candidate topics as calculated with c-TF-IDF.

        Returns:
        - The topic label as a string if successful, otherwise None.
        """
        try:
            response = self._create_chat_completion(messages)
            label = self._fetch_label_from_response(response)
            self._verbose_print(
                f'Topic {topic} with {response["usage"]["total_tokens"]} tokens as input length')
            return label

        except openai.InvalidRequestError as e:
            print(f'Invalid request for topic {topic}:\n{messages}')
            if handle_invalid_request:
                messages = self._update_messages(
                    ['No Documents'], topic, topics, self.backup_messages)
                self._verbose_print(f'Topic {topic}:\n{messages}')
                return self._try_chat_completion(messages)
            else:
                return None

        except Exception as e:
            print(f'Error for topic {topic}: {e}')
            return None

    def _create_chat_completion(self, messages: List[str]) -> Dict[str, Any]:
        """
        Initiates a ChatCompletion request with OpenAI.

        Args:
        - messages: List of messages to be processed.

        Returns:
        - Response dictionary from the ChatCompletion request.
        """
        return openai.ChatCompletion.create(
            engine=deployment_name,  # Make sure 'deployment_name' is accessible
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )

    def _fetch_label_from_response(self, response: Dict[str, Any]) -> str:
        """
        Fetches the topic label from the ChatCompletion response.

        Args:
        - response: The ChatCompletion response dictionary.

        Returns:
        - The topic label as a string.
        """
        return response['choices'][0]["message"]['content'].strip().replace("topic: ", "").replace("Topic: ", "")

    def _update_messages(self, docs, topic, topics, messages):
        """
        Replace [KEYWORDS] and [DOCUMENTS] with the real keywords and documents for each topic.
        """

        # Create list of keywords for the topic
        keywords = list(zip(*topics[topic]))[0]

        # Copy the self.messages to not overwrite it (deepcopy because it contains dicts in the list)
        return_messages = copy.deepcopy(messages)

        # Replace [KEYWORDS] and [DOCUMENTS] with the specific keywords and documents for the topic
        for i in range(len(return_messages)):
            if "[KEYWORDS]" in return_messages[i]['content']:
                return_messages[i]['content'] = return_messages[i]['content'].replace(
                    "[KEYWORDS]", ", ".join(keywords))

            if "[DOCUMENTS]" in return_messages[i]['content']:
                return_messages[i]['content'] = self._replace_documents(
                    return_messages[i]['content'], docs)

        return return_messages

    def create_representative_docs_list(self, documents_copy, max_docs=5):
        """
        Create a representative document list based on topics and a list of probabilities.

        Args:
            documents_copy (DataFrame): Copy of the original documents DataFrame.
            max_docs (int, optional): Maximum number of documents.

        Returns:
            dict: Dictionary of representative documents.
        """
        documents_copy["Probs"] = self.probs
        topics, probs = self.topics, self.probs
        repr_docs_dict = {}

        for topic_name, topic_group in documents_copy.groupby('Topic'):
            # Getting the indices of the documents within the current topic group
            topic_indices = topic_group.index

            # Extracting the probabilities for the documents in the current topic group
            topic_probs = probs[topic_indices]

            # Sorting the indices based on the probabilities in descending order
            sorted_indices = topic_indices[np.argsort(topic_probs)[::-1]]

            # Determining the number of documents to extract, ensuring it doesn't exceed the specified maximum or the available documents in the topic group
            num_docs_to_extract = min(max_docs, len(topic_group))

            # Getting the top N indices based on the sorted probabilities
            top_N_indices = sorted_indices[:num_docs_to_extract]

            # Extracting the text of the top N documents using the top N indices
            top_N_texts = documents_copy.loc[top_N_indices, 'Document'].tolist(
            )

            # Storing the representative documents for the current topic in the dictionary
            repr_docs_dict[topic_name] = top_N_texts

        return repr_docs_dict

    def create_representative_docs_list_from_matrix(self, documents_copy, max_docs=5):
        """
        Create a representative document list based on topics and a matrix of probabilities.

        Args:
            documents_copy (DataFrame): Copy of the original documents DataFrame.
            max_docs (int, optional): Maximum number of documents.

        Returns:
            dict: Dictionary of representative documents.
        """
        topics, probs = self.topics, self.probs
        repr_docs_dict = {}

        for topic_name, topic_group in documents_copy.groupby('Topic'):
            # Using the topic name as the index for the probability matrix
            topic_index = topic_name

            # Extracting the probabilities for the current topic from the matrix
            topic_probs = probs[:, topic_index]

            # Getting the IDs of the documents within the current topic group
            ids_in_topic_group = topic_group['ID'].tolist()

            # Filtering the probabilities to only include those of the documents in the current topic group
            filtered_probs = topic_probs[ids_in_topic_group]

            # Normalizing the probabilities so they sum up to 1
            normalized_probs = filtered_probs / np.sum(filtered_probs)

            # Determining the number of documents to sample, ensuring it doesn't exceed the available documents in the topic group
            actual_max_docs = min(max_docs, len(ids_in_topic_group))

            # Randomly sampling document indices based on their normalized probabilities
            sampled_indices = np.random.choice(np.arange(
                len(ids_in_topic_group)), size=actual_max_docs, replace=False, p=normalized_probs)

            # Using the sampled indices to get the actual document IDs
            sampled_ids = np.array(ids_in_topic_group)[sampled_indices]

            # Extracting the documents from the topic group using the sampled document IDs
            matching_docs = topic_group[topic_group['ID'].isin(sampled_ids)]

            # Getting the text of the sampled documents
            repr_docs = matching_docs['Document'].tolist()

            # Storing the representative documents for the current topic in the dictionary
            repr_docs_dict[topic_name] = repr_docs

        return repr_docs_dict

    def create_representative_docs(self, documents, max_docs=5):
        """
        Create representative documents based on the type of probability provided.

        Args:
            documents (DataFrame): Original documents DataFrame.
            max_docs (int, optional): Maximum number of documents.

        Returns:
            dict: Dictionary of representative documents.
        """
        self.topics = self.topic_model.topics_
        self.probs = self.topic_model.probabilities_

        documents_copy = documents.copy()

        if isinstance(self.probs, list) or (isinstance(self.probs, np.ndarray) and self.probs.ndim == 1):
            return self.create_representative_docs_list(documents_copy, max_docs)
        elif isinstance(self.probs, np.ndarray) and self.probs.ndim == 2:
            return self.create_representative_docs_list_from_matrix(documents_copy, max_docs)
        else:
            raise ValueError(
                "The probs is neither a list nor a valid numpy array.")

    @staticmethod
    def _replace_documents(prompt, docs):
        """
        Replace [DOCUMENTS] with formated real documents
        """
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc[:255]}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt

    def get_repr_docs(self, documents: List[str] = None):
        """ Returns the a dict of representative documents for each topic.
            Returns:
            - dict: A dictionary containing the representative documents.
        """

        if self.repr_docs_dict:
            return self.repr_docs_dict
        else:
            repr_docs_dict = self.create_representative_docs(
                documents=documents, max_docs=self.nr_docs)
            return repr_docs_dict

    def get_topic_description(self, prompt=None):
        """ Generates topic descriptions based on the representative documents.

        Parameters:
        - prompt (str): A custom prompt for the ChatGPT model.

        Returns:
        - dict: A dictionary containing topic descriptions.
        """

        # Check if representative documents exist
        if not self.repr_docs_dict:
            print("Error: Representative documents are not initialized.")
            return {}

        # Set the default prompt if none is provided
        if not prompt:
            prompt = ("The following documents are representative of a specific topic. "
                      "Based on their content, can you provide a concise summary or "
                      "description for this topic?")

        # Add space for the documents
        prompt = prompt + "\n```{}```"

        topic_descr_dict = {}

        # Iterate over the representative documents and get descriptions from ChatGPT
        for key, docs in self.repr_docs_dict.items():
            current_repr_docs = '\n'.join(docs)

            # Form the full prompt
            full_prompt = prompt.format(current_repr_docs)

            try:
                # Get response from ChatGPT
                response = openai.ChatCompletion.create(
                    engine=deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a topic summarizer model"},
                        {"role": "user", "content": full_prompt}
                    ]
                )

                # Extract the content from the ChatGPT response and store in the dictionary
                topic_descr_dict[key] = response['choices'][0]["message"]['content'].strip(
                )

            except Exception as e:
                print(f"Error while accessing ChatGPT for topic {key}: {e}")
                # Optionally: Store a default message in case of error
                topic_descr_dict[key] = "Description unavailable due to an error."

        return topic_descr_dict
