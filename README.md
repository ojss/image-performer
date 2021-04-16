# Image Performer (pytorch)

An implementation of the image transformer with the attention layers replaced with performers[1].

Inspired by code from @sahajgarg and @lucidrains.

```bash
python train_transformer.py --performer --config transformer_dmol_performer.yml --img64 "../imagenet64/unzip/" --doc "performer-6l-imgnet"
```


References:
[1]: Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane,A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser,L.,  et al.Rethinking attention with performers.arXiv preprintarXiv:2009.14794(2020). https://arxiv.org/abs/2009.14794

