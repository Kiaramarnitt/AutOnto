from ..OpenAlex_data_gathering.OpenAlex import OpenAlex
from topic_modelling.BERTopicModel import BERTopicModel
from ontology_generation.OntologyGen import OntologyGen
from ontology_generation.OntologyEncap import OntologyEncap
from ontology_generation.Evaluation import Evaluation
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from rdflib import Namespace, Graph
from sentence_transformers import SentenceTransformer

def validate_folder_path(folder_path):
    """
    Validate if the folder exists and contains CSV files.

    Parameters:
    - folder_path (str): Path to the folder.

    Returns:
    - bool: True if the folder exists and contains CSV files, False otherwise.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return False
    elif not any(filename.endswith('.csv') for filename in os.listdir(folder_path)):
        print(f"Warning: Folder '{folder_path}' does not contain any CSV files.")
        return False
    else:
        return True
    
def validate_input_data(df):
    """
    Validate input DataFrame.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame.

    Returns:
    - bool: True if input data is valid, False otherwise.
    """
    # Check if DataFrame has required columns
    required_columns = ['abstract', 'type', 'language', 'title']
    if not all(col in df.columns for col in required_columns):
        print("Error: DataFrame is missing required columns.")
        return False
    
    # Perform additional validation as needed

    return True

def run_model(folder_path, topic_name, messages, subsample_percentage):
    try:
        if os.path.exists(folder_path):
            if validate_folder_path(folder_path):
                # Select the CSV file containing the word "cleaned" in its filename
                csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv') and "cleaned" in filename]
                if csv_files:
                    # If there are files containing "cleaned", select the first one
                    csv_file = csv_files[0]
                    file_path = os.path.join(folder_path, csv_file)
                else:
                    raise ValueError("No CSV file with cleaned data")
            else:
                print("Folder does not contain any CSV files. Running openalex.get_data...")
                # Run openalex.get_data
                openalex = OpenAlex()
                openalex.get_data(topic_name, "topics", "data")
                # Attempt to load the CSV file again
                csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv') and "cleaned" in filename]
                if not csv_files:
                    raise ValueError("No CSV file containing 'cleaned' in its filename found.")
        else:
            raise ValueError("Folder path does not exist.")

        if csv_files:
            # Load the selected CSV file
            df = pd.read_csv(file_path)
            # Perform further processing with df
            # Clean the abstract column in the dataframe
            if validate_input_data(df):

                # Filter the dataframe based on certain conditions
                subset = df[(df['abstract'].str.len() > 10) & 
                        (df['type'] == 'article') & 
                        (df['language'] == 'en') & 
                        (df['title'].str.len() > 10)]

                # Calculate the number of rows for the subsample
                subsample_size = int(len(subset) * (subsample_percentage / 100))

                # Randomly sample the data
                subsample = subset.sample(n=subsample_size, random_state=42)

                # Split the title and abstract documents into lists
                title_docs = subsample["title"].to_list()
                abstract_docs = subsample["abstract"].to_list()

                # Model
                sentence_model = SentenceTransformer('all-mpnet-base-v2', device="cuda")
                embeddings = sentence_model.encode(title_docs, show_progress_bar=True)
                bert_base_model = BERTopicModel(embeddings=embeddings, reduce_outliers=True, messages=messages)
                results, freq, topic_dict, topics, topic_model = bert_base_model.train_model(title_docs)

                # Add the Topic and Topic Words to the subsample dataframe
                subsample["Topic"] = topics
                topic_dict_words = {x: [i[0] for i in topic_dict[x]] for x in topic_dict}
                topic_dict_first_word = {x: topic_dict[x][0][0] for x in topic_dict}
                subsample['topic_words'] = subsample['Topic'].map(topic_dict_words)
                subsample['topic_first_word'] = subsample['Topic'].map(topic_dict_first_word)

                subsample.to_csv("../output/subsample.csv")

                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                topic_model.save("../output/", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

                return subsample
    except Exception as e:
        print(f"An error occurred: {e}")

topic_name = "natural language processing"
excluded_words = "natural language processing, computer science, machine learning, artificial intelligence"
folder_path = "data/"
messages = [
            {"role": "system", "content": "You are a topic representation creator model for studies in the domain of {topic_name}. Your task is to determine the sub-domain of the research work based on its abstract. Each sub-domain name should not exceed more than 4 words. Your representations should be specific and focus on the most described object. Always provide a representation. Please do not use the following words delimited with triple backticks: '''natural language processing, computer science, machine learning, artificial intelligence'''"},
            {'role': 'user', 'content': """I have a topic that contains the following documents which are delimited with triple backticks:
        '''-  business world large companies that can achieve continuity in innovation gain a significant competitive advantage the sensitivity of these companies to follow and monitor news sources in e commerce social media and forums provides important information to businesses in decision making process large amount of data shared in these resources sentiment analysis can be made from people s comments about services and products users emotions can be extracted and important feedback can be obtained all of this is of course possible with accurate sentiment analysis this study new data sets were created for turkish english and arabic and for first time comparative sentiment analysis was performed from texts in three different languages addition a very comprehensive study was presented to researchers by comparing performances of both pre trained language models for turkish arabic and english as well as deep learning and machine learning models our paper will guide researchers working on sentiment analysis about which methods will be more successful in texts written in different languages which contain different types and spelling mistakes which factors will affect success and how much these factors will affect performance,
        - analysis also called opinion mining is field of study that analyzes people s opinions sentiments attitudes and emotions are important sentiment analysis since songs and mood are mutually dependent on each other on selected song it becomes easy find mood of listener future it can be used for recommendation the song lyric is a rich source of datasets containing words that are helpful analysis and classification of sentiments generated from it now a days observe a lot of inter sentential and intra sentential code mixing songs which has a varying impact on audience to study this impact created a telugu songs dataset which contained both telugu english code mixed and pure telugu songs in this paper classify songs based on its arousal as exciting or non exciting we develop a language identification tool and introduce code mixing features obtained from it as additional features system with these additional features attains 4 5 accuracy greater than traditional approaches on our dataset, 
        - this paper we propose a sentiment analysis model for the assessment of teacher performance in the classroom by tweets written by a pilot group of college students naive bayes nb is the technique to be applied to classify tweets based on the polar express emotion positive negative and neutral to carry out this process a dataset fits adding distinctive terms of context as possible features to support the classification process, 
        - analysis refers to classify emotion of a text whether positive or negative the studies conducted on sentiment analysis are generally based on english and other languages while there are limited studies on turkish in this study after constructing a dataset using a well known hotel reservation site booking com compare performances of different machine learning approaches we also apply dictionary based method sentitfidf which differs from traditional methods due to their logarithmic differential term frequency and term presence distribution usage the results are evaluated using area under of a receiver operating characteristic roc curve auc the results show that using document term matrix as input gives better classification results than tfidf matrix we also observe that best results are obtained using random forest classifier with an auc value of 89 on both positive and negative comments, 
        - the current era of computing the use of social networking sites like twitter and facebook is growing significantly over time people from different cultures and backgrounds share vast volumes of textual comments that show their viewpoints on several aspects of life and make them available to all for commenting monitoring real social media activities has now become a prime concern for politicians in understanding their social image this paper are going to analyse the tweets of various social media platforms regarding two prominent political leaders and classify them as positive negative or neutral using machine learning and deep learning methods we have proposed a deep learning approach for a better solution our proposed model has provided state of the art results using deep learning models'''
        It must be in the following format: <topic label>"""},
            {'role': 'assistant', 'content': 'Sentiment Analysis'},
            {"role": "user", "content": """I have a topic that contains the following documents which are delimited with triple backticks:
        '''[DOCUMENTS]'''
        REMEMBER to only use 1-4 words and to NOT use the following words delimited with triple backticks: '''{excluded_words}'''
        It must be in the following format: <topic label>"""}
]

print("Running model...")
df = run_model("data/", topic_name, messages, 5)
print("Model run completed.")

df = pd.read_csv("output/subsample.csv")

print("Initializing OntologyGen...")
ontogen = OntologyGen(model_name="gpt-4-1106-preview", deployment_name="gpt_chat_test_preview")
print("OntologyGen initialized.")

print("Extracting concepts...")
taxonomy_topics_df = ontogen.extract_concepts(df, topic_name)
print("Concept extraction completed.")

# Taxonomy creation

role = "You are an ontology engineer, tasked with helping build an ontology in the domain of {topic_name}"
eg = '''Create a taxonomy for NLP based on the following topics:
- Question Answering
- Semantic Similarity
- Chatbot
- Annotation
- Co-occurrence
Desired format: {
"Category 1": ["topic 1", "topic 5","..."],
"Category 2": [{"topic 3" : ["topic 6", "topic 8"]},"topic 9","topic 11", "..."],
"Category 3": ["topic 2", "topic 4","..."]
}'''
answer = '''{
     "Semantic Text Processing": [{"Semantic Similarity" : ["Concept Similarity", "Semantic Distance", "Sentence Similarity", "Word Similarity"]}],
     "Natural Language Interfaces": ["Question Answering", "Chatbot"],
     "Text Processing: ["Annotation", "Co-occurrence"]
}'''

print("Creating taxonomy...")
taxonomy = ontogen.taxonomy_creation(role, taxonomy_topics_df, topics_per_batch = 20, eg=eg, answer=answer)
print("Taxonomy creation completed.")

# Determining relations

prompt = "Given the following taxonomy:\n" + str(taxonomy) + '''
\n\Modify the taxonomy with the relations between the key and values with superTopicof, contributesTo. Use relation relatedEquivalent only if two topics are same'''

prompt += '''Return the taxonomy only in the following format. Check that all parenthesis are closed:
    {
     "Category 1": { "relation 1": ["topic 1", "topic 5","..."]},
     "Category 2": { "relation 1" : [{"topic 3": { "relation 2": ["topic 6", "topic 8"]}}, "topic 9","topic 11", "..."]},
}''' + '''\nDO NOT ADD FILLER WORDS LIKE "NEWLY ADDED"'''

eg = '''Given the following taxonomy: 
{
     "Semantic Text Processing": [{"Semantic Similarity" : ["Concept Similarity", "Semantic Distance", "Sentence Similarity", "Word Similarity"]}]},
     "Natural Language Interfaces": ["Question Answering", "Chatbot"],
     "Text Processing: ["Annotation", "Co-occurrence"]
}
Define the relations between the key and values with superTopicof, contributesTo. Use relation relatedEquivalent only if two topics are same.\n Desired format:
    {
     "Category 1": { "relation 1": ["topic 1", "topic 5","..."]},
     "Category 2": { "relation 1" : [{"topic 3": { "relation 2": ["topic 6", "topic 8"]}}, "topic 9","topic 11", "..."]},
     "Category 3": {"relation 2": ["topic 13"], "relation 1": ["topic 14", "topic 17"]}
}'''

# AlternativeLabelOf, IsParentOf, Includes, IsRelatedTo.
answer = '''{
  "Semantic Text Processing": {
    "IsParentOf": [
      {
        "Semantic Similarity": {
          "Includes": [
            "Concept Similarity",
            "Semantic Distance",
            "Sentence Similarity",
            "Word Similarity"
          ]
        }
      }
    ]
  },
  "Natural Language Interfaces": {
    "contributesTo": [
      "Question Answering",
      "Chatbot"
    ]
  },
  "Text Processing": {
    "superTopicof": [
      "Annotation",
      "Co-occurrence"
    ]
  }
}'''

print("Adding relations")
taxonomy = ontogen.prompt_extract(role, prompt)
print("Relation prompting completed.")

# Update taxonomy

prompt = '''You have a taxonomy for {topic_name} concepts. Your task is to:

1. Reorganize the taxonomy by introducing more levels and sub-hierarchies to better group related concepts.
2. Review the 'Irrelevant' category and identify any topics that are actually relevant to {topic_name} or related fields.
3. For relevant topics from 'Irrelevant', decide where to place them within the reorganized taxonomy structure (under existing categories, new subcategories, or new top-level categories if needed).
4. For remaining irrelevant topics, remove them from the {topic_name} taxonomy and introduce separate top-level dictionaries to categorize them into and ensure that the dictionaries don't contain themselves.

Original taxonomy:\n''' + str(taxonomy) + '''\nReturn ONLY the modified taxonomy'''

taxonomy = ontogen.prompt_extract(role, prompt)
print("Taxonomy update completed.")

# Manual organization
print("Reorganizing taxonomy...")
categories = [topic_name]

# Example 
# main_categories = [
#     'Natural Language Processing',
#     'Artificial Intelligence',
#     'Computer Science'
# ]


taxonomy = ontogen.reorganize_taxonomy(taxonomy, categories)
print("Taxonomy reorganization completed.")

# Export taxonomy as a json file - Prompt is hard coded
ontogen.taxonomy_json(role, taxonomy)
print("Taxonomy exported as JSON.")

# Extract topics from taxonomy for term typing
terms = ontogen.extract_topics(taxonomy)
role = "You are a topic summarizer model"
temp = '''You are a summarizer model tasked with writing one line descriptions for ontology creation for the {topic_name} domain. 
Describe the following topics in one line: ''' + ', '.join(terms) + '''
Return the response only in the dictionary format. USE proper QUOTES:
{
'topic1': 'description1',
'topic2': 'description2',
}
'''
print("Prompting for topic descriptions...")
topics_descriptions = ontogen.prompt_extract(role, prompt)
print("Term typing completed")

# Define namespaces
fh= Namespace('http://fraunhofer.de/example/')  # Namespace for our entities
schema = Namespace("http://schema.org/")  # Common vocabulary for attributes
wiki= Namespace('https://www.wikidata.org/wiki/')

ontoencap = OntologyEncap(fh, schema, wiki)
g = ontoencap.initiate_graph()

try:
    # Print or serialize RDF triples (turtle format)
    print("Processing data for ontology encoding")
    graph = ontoencap.process_data(taxonomy, topics_descriptions)
    turtle_data = graph.serialize(format="turtle")
    print(turtle_data)
except Exception as e:
    print("Error occurred during serialization:", e)

# Assuming 'data' is the provided data dictionary
graph.serialize(destination=f'output/taxonomy.ttl' , format="turtle")

# Evaluation
file_path = "data/CSO.3.3.ttl"
parent_topic_uri = "https://cso.kmi.open.ac.uk/topics/natural_language_processing"
deduplicated_topics = Evaluation.extract_concepts_and_deduplicate(file_path, parent_topic_uri)
cso_concepts = Evaluation.process_concepts(deduplicated_topics)

role = "You are an ontology engineer"
prompt = '''You are a model tasked with removing any duplicate topics from a list for ontology creation for the {topic_name} domain. 
The list: ''' + ', '.join(cso_concepts) + '''
Return the response only in the dictionary format. Ensure proper use of quotations:
{
'topic1',
'topic2
}
'''
cso_list = ontogen.prompt_extract(role, prompt)

concept_uri = "http://fraunhofer.de/example/Natural_Language_Processing"
graph = Graph()
graph.parse("output/taxonomy.ttl", format="ttl")  # Load your RDF data

descendants = Evaluation.get_descendants(concept_uri, graph)
concepts_onto = Evaluation.clean_concept_names(descendants)

print("Number of concepts in CSO: ", len(cso_list))
print("Number of concepts in OntoNLP: ", len(concepts_onto))

evalinst = Evaluation()
cso_list_processed = evalinst.preprocess_list(cso_list)
concepts_onto_processed = evalinst.preprocess_list(concepts_onto)

# Load a pre-trained SentenceTransformer model
whaleloops_model = SentenceTransformer("whaleloops/phrase-bert")

# Calculate metrics for preprocessed_list1
phrase_embeddings1 = whaleloops_model.encode(cso_list_processed)
reference_embedding = whaleloops_model.encode('natural-language-processing')
metrics1 = Evaluation.calculate_metrics(phrase_embeddings1, reference_embedding)

# Calculate metrics for preprocessed_list2
phrase_embeddings2 = whaleloops_model.encode(concepts_onto_processed)
metrics2 = Evaluation.calculate_metrics(phrase_embeddings2, reference_embedding)

# Create a DataFrame to store the metrics
# Create a list of dictionaries to store the metrics
data = [
    {
        "List": "CSO",
        "Number of Terms": len(cso_list_processed),
        **metrics1
    },
    {
        "List": "OntoNLP",
        "Number of Terms": len(concepts_onto_processed),
        **metrics2
    }
]

comparison = pd.DataFrame(data)

# Export the DataFrame to a CSV file
comparison.to_csv("output/metrics_comparison.csv", index=False)