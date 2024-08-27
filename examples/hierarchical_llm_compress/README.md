## Preparing the environment
Please follow https://graphstorm.readthedocs.io/en/latest/install/env-setup.html to setup your GraphStorm environment.
In addition, run the following scripts to install necessary python packages

```
pip install ipython
pip3 install deepspeed
sudo apt install python3-dotenv
```

You also need to put your huggingface account token into .env file.

## Preparing Amazon Review dataset
This folder contains the data processing script to process the raw Amazon Review dataset
downloaded from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/. We use domain Video
Games https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz
and put it under raw_data as an example.
```
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz \
-P raw_data/
```

To construct a graph for GNN training in GraphStorm, we need to preprocess the data
and convert them to the format supported by GraphStorm on a CPU machine.
```
python preprocess_amazon_review.py
```

Once the data are processed, run the following command to construct a graph
for PEFT LLM-GNNs in GraphStorm for node classification on level-3 product type.
The command takes `AR_Video_Games.json` that specifies the input data for graph
construction, constructs the graph, and saves the parition to `amazon_review`.

```
python -m graphstorm.gconstruct.construct_graph \
			--conf-file AR_Video_Games.json \
			--output-dir datasets/amazon_review_Video_Games/ \
			--graph-name amazon_review \
			--num-processes 16 --num-parts 1 \
			--skip-nonexist-edges --add-reverse-edges

```
