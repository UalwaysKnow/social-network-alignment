Use 'foursquare-twit' as example,node:5313-5120 train:1288 test:323
In 'example embeddings','anchors_train.txt' and 'anchors_test.txt' are from 'example_source'.For each network,models need 4 files,use foursquare as an example. 
Preprocessing£º
1.deepwalk:
input:four_edges.txt; output:four.embeddings
2.sort.py:
input:four.embeddings; output:four_sorted.txt
3.extract.py:
input:anchors_train.txt,anchors_test.txt,four_sorted.txt
output:four-align-train.txt;four-align-test.txt
4.sampling-weight.py:
input:four_edges.txt; output:four_weight.txt
