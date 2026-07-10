# LLIE-with-U-Net-GANs-using-DISTS-loss

Our replication of the recent paper ['Bridging Robustness and Efficiency: Real-Time Low-Light Enhancement via Attention U-Net GAN'](https://arxiv.org/pdf/2601.06518), and experimentation with an alternate loss (DISTS), that we feel might perform better.

## Dataset

Download the Sony subset of the SID dataset from Kaggle:

```bash
curl -L -o /path/to/folder/sid-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/marcorosato/sid-dataset
```
