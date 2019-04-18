# tinysearch
Semantic Search Engine using BERT embeddings
This is a project done as a part of CSCE 636 (Neural Networks).  

Existing search engines use keyword matching or tf-idf based matching to map the query to the web-documentsand rank them. They also consider other factors such as pagerank, hubs-and-authority scores, knowledge graphs to make theresults more meaningful. However, the existing search enginesfail to capture the meaning of query when it becomes largeand complex. BERT, introduced by Google in 2018, provides embeddings for words as well as sentences. In this project, I have developed a semantics-oriented search engine using neural networks and BERT embeddings that can search for query and rank the documents in the order of the most meaningful to least-meaningful. The results shows improvement over one existing search engine for complex queries for given set of documents.


Install Dependencies:
1. pip install bert-serving-server from [here](https://github.com/hanxiao/bert-as-service)
2. pip install tensorflow
3. pip install tkinter
4. pip install keras

How to run:

3. Download the two folders (uncased and model) from this zipped file from [drive](https://drive.google.com/file/d/1qx5lKIJ-F0f-VLexNFybcQvkcgIUQUZr/view?usp=sharing).

4. Run the bert-serving server as follows:
  `bert-serving-start -model_dir=uncased_L-12_H-768_A-12/ -tuned_model_dir=model/ -ckpt_name=model.ckpt-78 -num_worker=1 -pooling_strategy=CLS_TOKEN -max_seq_len=125 -num_worker=4`
  
5. To train the model run:
`python generate_embeddings.py` and then `python train.py`. 
Generate embeddings fetches the embeddings of quora-question-pairs and saves them. The train file loads the embeddings and trains the neural network model.
6. To run the GUI type: `python gui_v4.py`

