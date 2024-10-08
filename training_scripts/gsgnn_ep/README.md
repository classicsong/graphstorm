# Training script examples for edge classification/regression
This folder provides example yaml configurations for edge classification and regression training tasks.
The configurations include:

  * ``ml_ec.yaml`` defines an edge classification task on the ``(user, rating, movie)`` edges. The target label field is ``rate``. It uses a single-layer RGCN model as its graph encoder.

  * ``ml_er.yaml``defines an edge regression task on the ``(user, rating, movie)`` edges. The target label field is ``rate``. It uses a single-layer RGCN model as its graph encoder.

  * ``ml_ec_homogeneous.yaml`` defines an edge classification task for a homogeneous graph. The target label field is ``rate``. It uses a single-layer GraphSage model as its graph encoder.

  * ``ml_ec_text.yaml`` defines an edge classification task on the ``(user, rating, movie)`` edges. The target label field is ``rate``. It uses a single-layer RGCN model as its graph encoder. In addition, the training task will do **LM-GNN co-training**. A BERT model, i.e., ``bert-base-uncased``, is used to compute the text embeddings of ``movie`` nodes and ``user`` nodes on the fly. During training, GraphStorm will randomly select 10 nodes for each mini-batch to participate the gradient computation to tune the BERT models. For more detials, please refer to https://graphstorm.readthedocs.io/en/latest/advanced/language-models.html#fine-tune-lms-on-graph-data.

  * ``ml_lm_ec.yaml`` defines an edge classification task on the ``(user, rating, movie)`` edges. The target label field is ``rate``. It uses a language model, i.e., the BERT model, as its graph encoder. The training task will do **LM-Graph co-training**. The BERT model will compute the text embeddings of ``movie`` nodes and ``user`` nodes on the fly. During training, GraphStorm will randomly select 10 nodes for each mini-batch to participate the gradient computation to tune the BERT models. For more detials, please refer to https://graphstorm.readthedocs.io/en/latest/advanced/language-models.html#fine-tune-lms-on-graph-data.


The example inference configurations are in ``inference_scripts/ep_infer/README``.

The following example script shows how to launch a GraphStorm edge classification training task.
You may need to change the arguments according to your tasks.
For more detials please refer to https://graphstorm.readthedocs.io/en/latest/cli/model-training-inference/index.html.

```
python3 -m graphstorm.run.gs_edge_classification \
    --workspace graphstorm/training_scripts/gsgnn_ep/ \
    --num-trainers 4 \
    --num-servers 1 \
    --num-samplers 0 \
    --part-config /data/movielen_100k_multi_label_ec/movie-lens-100k.json \
    --ip-config ip_list.txt \
    --ssh-port 2222 \
    --cf ml_ec.yaml \
    --save-model-path /data/gsgnn_ec/
```

The script loads a paritioned graph from ``/data/movielen_100k_multi_label_ec/movie-lens-100k.json`` and saves the trained model in ``/data/gsgnn_ec/``.

Note: All example movielens graph data are generated by ``tests/end2end-tests/create_data.sh``.
