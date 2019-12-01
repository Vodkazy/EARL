# EEARL : Leveraging Context Information for Joint Entity and Relation Linking

EEARL (Extended Entity and Relation Linker), a system for jointly linking entities and relations in a question to a knowledge graph. Based on EARL (ISWC2018) , We address the problem of joint entity and relation linking, and proposes an Extended Entity and Relation Linker (EEARL) in order to improve the accuracy of the linking. EEARL extracts the context information of the keywords to be linked and the attribute features of the entity/relation by using the word embedding model, and then calculates the similarity between the context vector and the attribute tag vector to re-score the candidate elements. 

## Processing
For EARL, we directly reused its Python code published in [Github](https://github.com/AskNowQA/EARL). We implemented EEARL by modifying the entity/relation disambiguation part of EARL’s code.
Specifically, we follow the process of EARL. Given a natural language question, we use the `SENNA `system to extract keyword phrases, and then use EARL's `entity/relation prediction model` to determine the entity/relation type of the keyword phrases, and then use `ElasticSearch ` to traverse the pre-processed dictionary to generate candidate element lists, and calculates scores based on the **connection density** and **context information** to obtain the best entity/relation mappings in the KB.

## Prerequisites

Install all dependencies required that are mentioned in dependencies.txt. 
```
    $ sudo pip install -r requirements.txt
```

## Data Preparation

Download files from [URI1](https://pan.baidu.com/s/19tS_ZG9pKf9GDPukGGhCQA) and [URI2](https://www.dropbox.com/s/flh1fjynqvdsj4p/lexvec.commoncrawl.300d.W.pos.vectors.gz?dl=1) and put them into local part and restore them as following structure：
```
├── EARL
├── data
├── download
|   ├── blooms
|       └── bloomｘｘｘ.file
|   ├── elasticsearchdump
|       └── dbｘｘｘ.file
|   ├── Glove
|       └── gloveｘｘｘ.file
|   ├── lexvec.commoncrawl.300d.W.pos.neg3.vectors
|   └── text
├── files
├── model
└── script
```
链接：https://pan.baidu.com/s/19tS_ZG9pKf9GDPukGGhCQA 密码：hyyf

## Run

### ElasticSearch

To import elasticsearch data one could install elasticdump https://www.npmjs.com/package/elasticdump
```
    npm install elasticdump -g
```
Then import the mapping and the actual data:
```
    elasticdump --input=dbentityindex11mapping.json  --output=http://localhost:9200/dbentityindex11 --type=mapping

    elasticdump --limit=10000 --input=dbentityindex11.json  --output=http://localhost:9200/dbentityindex11 --type=data
```
Run the demo:
```
    $ python Main.py
```

## Note
This code is based on the LC-QuAD dataset for experimental design. If you want to link a question separately, you can modify the code in the `Main.py` file.
One advice：The entire code start up very slowly, and if you wanna process these code more quickly, please change it to the server architecture if needed.
