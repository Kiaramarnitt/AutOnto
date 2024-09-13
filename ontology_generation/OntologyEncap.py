import rdflib
from rdflib import Namespace
from rdflib import URIRef, BNode, Literal, Graph, Literal, RDF
from rdflib.namespace import CSVW, DC, DCAT, DCTERMS, DOAP, FOAF, ODRL2, ORG, OWL, PROF, PROV, RDF, RDFS, XSD
from urllib.parse import quote
import re

class OntologyEncap():

    def __init__(self, fh, schema, wiki) -> None:
        # Define namespaces
        self.fh= fh
        self.schema = schema
        self.wiki= wiki

    def initiate_graph(self):
    # Create a new RDF graph
        g = Graph()
        
        # Bind namespace prefixes
        g.bind("fh", self.fh)
        g.bind("schema", self.schema)
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("wiki", self.wiki)
        g.bind("xsd",XSD)
        g.bind("dc",DC)
        g.bind("dct",DCTERMS)
        return g

    def create_uri(self, namespace,string):
        name = string.replace('.', '_')# Replacing points with underscores
        name = re.sub(r'[^\w\s-]', '', name) # Removing characters that aren't words, whitespace, or dashes
        name = name.replace(' ', '_') # Replacing spaces with underscores
        name = name.replace('__', '_') # Replacing double undesrcore with single
        name = name.strip()
        uri_encoded = quote((namespace+f"{name}"), safe=':/#')
        uri = URIRef(uri_encoded)
        return uri if uri else URIRef(namespace)

    def create_class_uri(self, fh, label):
        """
        Create a URI for a class based on a label.
        """
        return self.create_uri(fh, label)

    def add_class(self, graph, class_uri, label=None, comment=None, parent_uri=None, relationship_uri=None):
        """
        Add triples to specify that the URI represents a class,
        set its rdfs:label, and optionally add its rdfs:comment.
        """
        graph.add((class_uri, RDF.type, RDFS.Class))  # Specify the type as Class
        if label:
            graph.add((class_uri, RDFS.label, Literal(label)))  # Set the label
        if comment:
            graph.add((class_uri, RDFS.comment, Literal(comment)))  # Set the comment
        # if parent_uri:
        #     graph.add((class_uri, RDFS.subClassOf, parent_uri))  # Specify subClassOf relationship

        if parent_uri and relationship_uri:
            graph.add((parent_uri, relationship_uri, class_uri))  # Specify relationship to parent


    def add_relationship(self, graph, parent_uri, relationship_uri, child_uri):
        """
        Add a triple to represent a relationship between a parent and child URI.
        """
        graph.add((parent_uri, relationship_uri, child_uri))


    def process_relationship_data(self, graph, fh, parent_uri, relationship_name, relationship_data, meta=None):
        """
        Process relationship data, which could be a list, dict, or scalar value.
        """
        relationship_uri = self.schema[relationship_name]

        if isinstance(relationship_data, list):
            for item in relationship_data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        child_uri = self.create_class_uri(fh, key)
                        self.add_class(graph, child_uri, key, meta.get(key) if meta else None, parent_uri, relationship_uri)
                        self.add_relationship(graph, parent_uri, relationship_uri, child_uri)
                        self.process_relationship_data(graph, fh, child_uri, self.schema.superTopicOf, value, meta)
                else:
                    child_uri = self.create_uri(fh, item)
                    self.add_class(graph, child_uri, item, meta.get(item) if meta else None, parent_uri, relationship_uri)

        elif isinstance(relationship_data, dict):
            self.process_relationships(graph, fh, parent_uri, relationship_data, meta)

        else:
            child_uri = self.create_uri(fh, relationship_data)
            self.add_class(graph, child_uri, relationship_data, meta.get(relationship_data) if meta else None, parent_uri, relationship_uri)


    def process_relationships(self, graph, fh, parent_uri, relationships, meta=None):
        """
        Process a dictionary of relationships for a given parent URI.
        """
        for relationship_name, relationship_data in relationships.items():
            self.process_relationship_data(graph, fh, parent_uri, relationship_name, relationship_data, meta)

    def process_data(self, data, meta=None):
        """
        Process the input data and create an RDF graph.
        """
        g = Graph()
        for category, relationships in data.items():
            # Create a new parent URI for each top-level dictionary
            parent_uri = self.create_class_uri(self.fh, category)
            self.add_class(g, parent_uri, category, meta.get(category) if meta else None)

            # Process relationships within the category
            self.process_relationships(g, self.fh, parent_uri, relationships, meta)

        return g