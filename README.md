# Movie Recommender Engine
Recommend users movies by their interested movie types using matrix factorization. 

## Requirements
  * NumPy >= 1.12.0
  * keras >= 2.1.0
  * TensorFlow >= 1.4

## File description
  * `train.py` includes model architecture and training parameters.
  * `infer.py` is for inference.
  * The directory `model` includes pretrained matrix factorization model that performed best in my experiements.

## Experiments
  * Compare the results of different latent dimension parameters.
![](https://imgur.com/z0kYO4X.png)

  * Compare the effect of adding `bias` parameter
![](https://imgur.com/fkHM54N.png)