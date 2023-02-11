# Query-adaptive-ontology-guided-information-retrieval-for-biomedical-search

This is the code repository for the query adaptive ontology based information retrieval model combining multiple ontologies in bioinformatics in different ways depending on the query context

**Description:** As multiple existing ontologies in the biomedical field often contain complementary information, it is crucial to combine them effectively during the search. We propose a deep learnign algorithm to learn factors based on local and global context of a query to utilize ontologies with different weights that are adapative based on the query. [Report](https://drive.google.com/file/d/1ys0GlE6mLwL86WdqOH5Qpyq88WPfWe6C/view?usp=sharing)   

### Contents

`Code` : Code for data cleaning, data processing and learning weights/factors called "Expansion factor" and "Ontology factor".
`expansion_model_revised` : deep-learning method for learning one of the adaptive weights. 
`BM25 results` : search ranking results with the BM25 and BM25 + query adaptive ontology guided results.  
`LC_related_data` : word vocabulary, some of the data statistics

## License

By using this source code you agree to the license described in https://github.com/sahitilucky/Query-adaptive-ontology-guided-information-retrieval-for-biomedical-search/blob/master/LICENSE
