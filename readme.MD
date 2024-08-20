# AutOnto

Using LLMs and Topic Modelling to generated ontologies

## Structure

- **data**: Contains data retrieved and cleaned from OpenAlex API.
  
  - `data.csv`: Original data.
  - `cleaned_data.csv`: Processed and cleaned data.

- **data_collection**: Module for data retrieval and cleaning.
  
  - `OpenAlex.py`: Implementation of the OpenAlex class for fetching and processing data from the OpenAlex API.
  - `example.py`: Example usage of the module.
  - `visualization.ipynb`: Jupyter notebook for data exploration and visualization.

- **ontology_generation**: Functions for ontology generation.
  
  - `Evaluation.py`: Module for extracting topics from existing ontologies, cleaning them, and comparing using similarity metrics.
  - `OntologyEncap.py`: Module for encoding ontology to TTL format.
  - `OntologyGen.py`: Module for ontology generation, utilizing utilities from `utilsLLM`.
  - `utilsLLM.py`: Utility functions for prompting to GPT and cleaning responses.

## How to use

`run.py`: Shows how the pipeline can be used

`run.ipynb`: Jupyter notebook version of the above script

- Modify the search term in the following [line](https://github.com/yeahalti/AutoOnto/blob/fb6a304627f99e96498b6bcf53e24ba9b016ee66/run.py#L67) in `run.py` or alternatively, modify and run `example.py` in `data_collection`. 
- Set work_parameter as either [topics](https://docs.openalex.org/api-entities/topics) or [concepts](https://docs.openalex.org/api-entities/concepts). Note: While the `concepts` feature has been deprecated, this repository utilizes it as it was implemented when the feature was still active.

## To-do
  - Add models to determine concepts
  - Study about topic association
  
