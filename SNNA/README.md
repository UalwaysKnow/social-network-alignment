Includding two models:SNNAu and SNNAo.I use 'foursquare-twitter' as example training data.
1.Get input files:use deepwalk(100 dimension) to train embeddings and do preprocessing before training,details are written in 'example_prepocessing'.
2.Two models share the 10 input files,which are in 'example embeddings',output files will be set in each 'code' folders.
3.After preprocessing,start training with the method written in each 'code' folders.

Two models share the same parameters,renewing the following parameters when training data is changed:
--input_netl_nodes --input_netr_nodes --input_netl_weight --input_netr_weight --input_netl_anchors --input_netr_anchors --input_train_anchors --input_test_anchors --input_test_netl_embeddings --input_test_netr_embeddings --left_nodes_num --right_nodes_num --train_anchors_num --test_anchors_num --epoch_all
