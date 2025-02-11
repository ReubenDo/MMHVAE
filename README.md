#  Unified Cross-Modal Image Synthesis with Hierarchical Mixture of Product-of-Experts

Public PyTorch implementation for our paper [Unified Cross-Modal Image Synthesis withHierarchical Mixture of Product-of-Experts](https://arxiv.org/pdf/2410.19378), 
currently under review. 

If you find this code useful for your research, please cite the following paper:

```
@article{dorent2024unified,
  title={Unified Cross-Modal Image Synthesis with Hierarchical Mixture of Product-of-Experts},
  author={Dorent, Reuben and Haouchine, Nazim and Golby, Alexandra and Frisken, Sarah and Kapur, Tina and Wells, William},
  journal={arXiv preprint arXiv:2410.19378},
  year={2024}
}
```

## Method Overview
We propose a deep mixture of multimodal hierarchical variational auto-encoders called MMHVAE that synthesizes missing images from observed images in different modalities. MMHVAE’s design focuses on tackling four challenges: 
1. creating a complex latent representation of multimodal data to generate high-resolution images
2. encouraging the variational distributions to estimate the missing information needed for cross-modal image synthesis
3. learning to fuse multimodal information in the context of missing data
4. leveraging dataset-level information to handle incomplete data sets at training time. 

*Example of multimodal synthesis*
<p align="center">
  <img src="figs/synthesis_results.gif">
</p>



## Virtual Environment Setup

TODO
  
