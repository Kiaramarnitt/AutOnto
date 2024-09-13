# AutOnto

Using LLMs and Topic Modelling to generated ontologies

## Structure

- **data**: Contains data retrieved and cleaned from OpenAlex API.
  
  - `data.csv`: Original data.
  - `cleaned_data.csv`: Processed and cleaned data.

- **ontology_generation**: Functions for ontology generation.
  
  - `Evaluation.py`: Module for extracting topics from existing ontologies, cleaning them, and comparing using similarity metrics.
  - `OntologyEncap.py`: Module for encoding ontology to TTL format.
  - `OntologyGen.py`: Module for ontology generation, utilizing utilities from `utilsLLM`.
  - `utilsLLM.py`: Utility functions for prompting to GPT and cleaning responses.

## How to use

`run.py`: Shows how the pipeline can be used

`run.ipynb`: Jupyter notebook version of the above script

- Modify the search term in the following [line](https://git.informatik.fh-nuernberg.de/fe/autonto/-/blob/main/run.py?ref_type=heads#L120) in `run.py` or in `run.ipynb`
- Similarly, modify excluded terms in the line below (terms that are generic to the search term/of a higher domain.)
- Ensure `data\cleaned_data.csv` file exists in the same directory as `run.py` or `run.ipynb`.
- `cleaned_data.csv` file must have the following columns [`abstract`, `title`, `language`, `type`]. Alternatively, if your data doesn't have columns `language` or `type`, comment them out from this [line] (https://git.informatik.fh-nuernberg.de/fe/autonto/-/blob/main/run.py?ref_type=heads#L83)
- !Note: If the data doesn't have `abstract` or `title` then the user will have to comment multiple lines within the script.
  
