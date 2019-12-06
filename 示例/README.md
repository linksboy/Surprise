| [Movielens 100k](http://grouplens.org/datasets/movielens/100k)                                                                         |   Precision |   Recall |       F1 |
|:---------------------------------------------------------------------------------------------------------------------------------------|------------:|---------:|---------:|
| [SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)      |    0.575716 | 0.123359 | 0.203182 |
| [SVD++](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)  |    0.567515 | 0.122138 | 0.201015 |
| [NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)      |    0.594291 | 0.130879 | 0.214516 |
| [Slope One](http://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne)                 |    0.575468 | 0.123564 | 0.203445 |
| [k-NN](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)                        |    0.581531 | 0.128212 | 0.210103 |
| [Centered k-NN](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans)           |    0.585171 | 0.129571 | 0.212163 |
| [k-NN Baseline](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)            |    0.583439 | 0.128667 | 0.210837 |
| [Co-Clustering](http://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering) |    0.58328  | 0.12734  | 0.209042 |
| [Baseline](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)   |    0.579463 | 0.124081 | 0.204394 |
| [Random](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor)    |    0.588388 | 0.127879 | 0.210097  


加入不同k值
| [Movielens 100k](http://grouplens.org/datasets/movielens/100k)   |   Precision |   Recall |       F1 |
|:-----------------------------------------------------------------|------------:|---------:|---------:|
| SVD@5                                                            |    0.573418 | 0.125694 | 0.206191 |
| SVD@10                                                           |    0.582344 | 0.127848 | 0.209666 |
| SVD@15                                                           |    0.586691 | 0.125171 | 0.206323 |
| SVD@20                                                           |    0.581142 | 0.124835 | 0.205523 |
| SVD@25                                                           |    0.577589 | 0.125571 | 0.206293 |
| SVDpp@5                                                          |    0.579339 | 0.124177 | 0.204518 |
| SVDpp@10                                                         |    0.575574 | 0.126139 | 0.206929 |
| SVDpp@15                                                         |    0.57485  | 0.125247 | 0.205681 |
| SVDpp@20                                                         |    0.591004 | 0.125525 | 0.20707  |
| SVDpp@25                                                         |    0.576988 | 0.125585 | 0.206274 |
| NMF@5                                                            |    0.575221 | 0.123994 | 0.204012 |
| NMF@10                                                           |    0.583422 | 0.125693 | 0.206827 |
| NMF@15                                                           |    0.570608 | 0.126817 | 0.207514 |
| NMF@20                                                           |    0.582909 | 0.126829 | 0.20833  |
| NMF@25                                                           |    0.58245  | 0.124142 | 0.204662 |
| SlopeOne@5                                                       |    0.574249 | 0.122802 | 0.202336 |
| SlopeOne@10                                                      |    0.582891 | 0.124817 | 0.205606 |
| SlopeOne@15                                                      |    0.577554 | 0.124612 | 0.204994 |
| SlopeOne@20                                                      |    0.57158  | 0.122005 | 0.201088 |
| SlopeOne@25                                                      |    0.575451 | 0.122118 | 0.20148  |
| KNNBasic@5                                                       |    0.58837  | 0.127342 | 0.209369 |
| KNNBasic@10                                                      |    0.576493 | 0.123425 | 0.203319 |
| KNNBasic@15                                                      |    0.590597 | 0.128991 | 0.211737 |
| KNNBasic@20                                                      |    0.584217 | 0.131379 | 0.214518 |
| KNNBasic@25                                                      |    0.570573 | 0.125657 | 0.205956 |
| KNNWithMeans@5                                                   |    0.572641 | 0.123186 | 0.202756 |
| KNNWithMeans@10                                                  |    0.593814 | 0.126939 | 0.209165 |
| KNNWithMeans@15                                                  |    0.579745 | 0.126416 | 0.20757  |
| KNNWithMeans@20                                                  |    0.595811 | 0.131052 | 0.214847 |
| KNNWithMeans@25                                                  |    0.583369 | 0.127564 | 0.20935  |
| KNNBaseline@5                                                    |    0.576264 | 0.124418 | 0.204651 |
| KNNBaseline@10                                                   |    0.577784 | 0.122359 | 0.20195  |
| KNNBaseline@15                                                   |    0.573012 | 0.122444 | 0.201773 |
| KNNBaseline@20                                                   |    0.578208 | 0.1271   | 0.208392 |
| KNNBaseline@25                                                   |    0.571828 | 0.124468 | 0.204437 |
| CoClustering@5                                                   |    0.56824  | 0.124307 | 0.20399  |
| CoClustering@10                                                  |    0.582432 | 0.128989 | 0.211204 |
| CoClustering@15                                                  |    0.586444 | 0.128631 | 0.210985 |
| CoClustering@20                                                  |    0.584783 | 0.123666 | 0.204158 |
| CoClustering@25                                                  |    0.592877 | 0.127225 | 0.209495 |
| BaselineOnly@5                                                   |    0.587911 | 0.129877 | 0.212754 |
| BaselineOnly@10                                                  |    0.58427  | 0.127394 | 0.209179 |
| BaselineOnly@15                                                  |    0.583899 | 0.13     | 0.212654 |
| BaselineOnly@20                                                  |    0.574673 | 0.123854 | 0.203788 |
| BaselineOnly@25                                                  |    0.570166 | 0.122532 | 0.201714 |
| NormalPredictor@5                                                |    0.565465 | 0.119614 | 0.197459 |
| NormalPredictor@10                                               |    0.585525 | 0.126894 | 0.208585 |
| NormalPredictor@15                                               |    0.57865  | 0.124773 | 0.205282 |
| NormalPredictor@20                                               |    0.574072 | 0.125251 | 0.205636 |
| NormalPredictor@25                                               |    0.583704 | 0.127712 | 0.209571 |
