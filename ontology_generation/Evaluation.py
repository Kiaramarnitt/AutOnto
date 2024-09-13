from rdflib import Graph
import pandas as pd, numpy as np
from typing import Any, Dict, List
from itertools import chain
import re
from json.decoder import JSONDecodeError

from sklearn.metrics.pairwise import cosine_similarity


class Evaluation():
    @staticmethod
    def extract_concepts_and_deduplicate(file_path, parent_topic_uri):
        """
        Extracts concepts from an RDF file and deduplicates them.

        Args:
        - file_path (str): Path to the RDF file.
        - parent_topic_uri (str): URI of the parent topic.

        Returns:
        - set: Deduplicated set of concept URIs.
        """
        # Initialize the graph
        g = Graph()
        g.parse(file_path, format="ttl")

        # SPARQL query to select child topics and their equivalents
        query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ns0: <http://cso.kmi.open.ac.uk/schema/cso#>

        SELECT ?childTopic ?equivalent
        WHERE {
          <%s> ns0:superTopicOf ?childTopic .
          OPTIONAL { ?childTopic ns0:relatedEquivalent ?equivalent . }
        }""" % parent_topic_uri

        # Execute the SPARQL query
        results = g.query(query)

        # Process the results to deduplicate
        concept_properties = {}
        equivalent_map = {}

        for row in results:
            child_topic = str(row.childTopic)
            equivalent = str(row.equivalent) if row.equivalent else None

            # If there's an equivalent, map it back to the child topic
            if equivalent:
                equivalent_map[equivalent] = child_topic
            elif child_topic not in equivalent_map:  # Child topic with no equivalent
                equivalent_map[child_topic] = child_topic

        # Deduplication: prefer the child topic if it appears as an equivalent elsewhere
        deduplicated_topics = set(equivalent_map.values())

        # This part is simplified; expand according to your needs
        return deduplicated_topics

    @staticmethod
    def process_concepts(onto_concepts):
        """
        Processes ontology concepts and their properties.

        Args:
        - onto_concepts (dict or set): Ontology concepts.

        Returns:
        - dict or list: Processed ontology concepts.
        """
        if isinstance(onto_concepts, dict):
            new_concept_properties = {}
            for concept, properties in onto_concepts.items():
                new_properties = {}
                for property_uri, values in properties.items():
                    # Skip 'relatedLink' property
                    if property_uri == 'http://schema.org/relatedLink':
                        continue
                    
                    property_name = property_uri.split('#')[-1]
                    new_values = [value.split('/topics/')[-1] for value in values]
                    new_properties[property_name] = new_values
                
                concept_name = concept.split('/topics/')[-1]
                new_concept_properties[concept_name] = new_properties

        elif isinstance(onto_concepts, set):
            new_concept_properties = []
            for concept in onto_concepts:
                concept_name = concept.split('/topics/')[-1]
                new_concept_properties.append(concept_name)

        return new_concept_properties

    @staticmethod
    def get_descendants(concept_uri, graph):
        """
        Retrieves all descendants of a concept URI using SPARQL.

        Args:
        - concept_uri (str): The URI of the concept.
        - graph (rdflib.Graph): The RDF graph containing the concept hierarchy.

        Returns:
        - set: Set of URIs representing the descendants of the concept.
        """
        # SPARQL query to select all descendants of the given concept
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?descendant WHERE {
            ?descendant rdfs:subClassOf* <%s> .
        }
        """ % concept_uri

        # Execute the query
        results = graph.query(query)

        # Extract descendant URIs
        descendants = set()
        for row in results:
            descendants.add(str(row['descendant']))

        return descendants

    @staticmethod
    def clean_concept_names(concept_uris):
        """
        Cleans up a list of concept URIs and extracts only the concept names.

        Args:
        - concept_uris (list): List of concept URIs.

        Returns:
        - list: List of cleaned concept names.
        """
        cleaned_names = []
        for uri in concept_uris:
            # Split the URI by '/' and get the last part
            name = uri.split('/')[-1]
            # Replace underscores with spaces
            name = name.replace('_', ' ')
            cleaned_names.append(name)
        return cleaned_names

    def preprocess_term(self, term):
        # Lowercase the term
        term = term.lower()
        # Remove special characters, symbols, and underscores
        term = re.sub(r'[^a-zA-Z\s]', '_', term)
        # Remove any leading or trailing whitespaces
        term = term.strip()
        return term

    def preprocess_list(self, word_list):
        preprocessed_list = [self.preprocess_term(term).replace(" ", "_") for term in word_list]
        return preprocessed_list
    
    def calculate_average_similarity(list1, list2, word_vectors):
        similarities = []
        for word1 in list1:
            for word2 in list2:
                try:
                    # Calculate cosine similarity
                    similarity = word_vectors.similarity(word1, word2)
                    similarities.append(similarity)
                except KeyError:
                    # If a word is not in the vocabulary, skip it
                    continue
        if similarities:
            # Return the average similarity
            return sum(similarities) / len(similarities)
        else:
            return 0
        
    def calculate_metrics(phrase_embeddings, reference_embedding=None):
        # Calculate pairwise similarity within the list
        pairwise_similarity_within_list = cosine_similarity(phrase_embeddings)
        
        # Calculate average similarity
        upper_triangular_similarity = np.triu(pairwise_similarity_within_list, k=1).flatten()
        upper_triangular_similarity = upper_triangular_similarity[~np.isnan(upper_triangular_similarity)]
        pairwise_average_similarity = np.mean(upper_triangular_similarity)
        
        # Calculate minimum similarity
        minimum_similarity = np.min(pairwise_similarity_within_list)
        
        # Calculate maximum similarity
        maximum_similarity = np.max(pairwise_similarity_within_list)
        
        # Calculate median similarity
        median_similarity = np.median(pairwise_similarity_within_list)
        
        # Calculate standard deviation of similarity
        std_similarity = np.std(pairwise_similarity_within_list)
        
        # Aggregate embeddings by averaging
        aggregate_embedding = np.mean(phrase_embeddings, axis=0)
        
        # Compute cosine similarity between the aggregate embedding and each individual embedding
        aggregate_similarity = cosine_similarity(phrase_embeddings, aggregate_embedding.reshape(1, -1))
        
        # Compute the mean aggregate similarity
        mean_aggregate_similarity = np.mean(aggregate_similarity)
        
        # Optionally compute cosine similarity between the aggregate embedding and a reference embedding
        if reference_embedding is not None:
            # Compute cosine similarity between aggregate embedding and reference embedding
            cosine_similarity_with_reference = cosine_similarity(aggregate_embedding.reshape(1, -1), reference_embedding.reshape(1, -1))
        else:
            cosine_similarity_with_reference = None
        
        return {
            "pairwise_average_similarity": pairwise_average_similarity,
            #"aggregate_similarity": aggregate_similarity.flatten(),
            "mean_aggregate_similarity": mean_aggregate_similarity,
            "cosine_similarity_with_reference embedding": cosine_similarity_with_reference[0][0],
            "minimum_similarity": minimum_similarity,
            "maximum_similarity": maximum_similarity,
            "median_similarity": median_similarity,
            "std_similarity": std_similarity
        }

