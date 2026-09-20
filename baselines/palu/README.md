# Palu: Compression KV-Cache with Low-Rank Decomposition

> SparseEngine vendors the runtime subset only. The upstream benchmark code is
> intentionally omitted; use `../../benchmark/` for repository-owned runs.
[[Paper](https://arxiv.org/abs/2407.21118)]

<div align='center'>
<img width="100%" alt="image" src="img/palu_idea.png"> 
</div>

## Updates
- [2024.11.03]: We updated our [arXiv](https://arxiv.org/pdf/2407.21118) with latency evaluation on quantization integration. Check it out!
- [2024.08.01]:🚀 Palu ver. 1 is released.

## TL;DR
Palu is a KV-Cache compression framework that utilizes low-rank projection to compress the hidden dimension of KV-Cache, thereby reducing memory footprint and increasing speed.

## Abstract
Palu is a pioneer KV-Cache compression framework that reduce the hidden dimenssion of KV-Cache via low-rank projection.
Different from [MLA in DeepSeek-V2](https://arxiv.org/abs/2405.04434) that requires a large-scale training from scratch, Palu works with existing LLMs such as Llama3, Mistral, in a post-training manner.
To achieve this, Palu decomposes the linear layers into low-rank matrices, caches the smaller intermediate states, and reconstructs the full keys and values on the fly. To improve accuracy, compression rate, and efficiency, Palu further encompasses (1) a medium-grained low-rank decomposition scheme, (2) an efficient rank search algorithm, (3) matrix fusion for quantization friendliness enhancements, and (4) co-designed GPU kernels. 

Our extensive experiments with popular LLMs show that Palu can compress KV-Cache by more than 91.25% while maintaining a significantly better accuracy (up to 1.19 lower perplexity) than state-of-the-art KV-Cache quantization methods at a similar or even higher memory usage. For more details, please refer to our [paper](https://arxiv.org/abs/2407.21118).

## Todo Lists
- [ ] Upgrade `transformers>=4.43.3`, for Llama3.1 support
- [ ] Update reconstruction kernel, with quantization integrated.
- [ ] Support FlashAttention or FlashInfer to enhance competatiblity
  

## Installation
1. Clone the repository (Make sure you have Git, Conda installed on your system)
```
git clone --recurse-submodules https://github.com/shadowpa0327/Palu.git
cd Palu
```

2. Prepare environment
```
conda create -n Palu python=3.10
conda activate Palu
pip install -r requirements.txt
```

3. Install the runtime third-party library
```
pip install -e 3rdparty/fast-hadamard-transform
```

## Usage
### Rank Search and Compression
We provide a script `compress.py` to perform the rank search and low-rank decomposition to generate the low-rank projection matrices for compressing KV-Cache. Here, we perform the decomposition with proposed `G-LRD` methods with group size equal to 4 as an example. 
```bash
python compress.py \
--model_id=/Path/To/Pretrained/Model \
--calib_dataset wikitext2 \
--param_ratio_target 0.7 \
--search_method fisher_uniform \
--head_group_size 4 \
--dump_huggingface_model \
--use_cache 
```

After executing the above command, a compressed models with decomposed low-rank projection matrices will be dumped into the `{MODEL_NAME}-ratio-{TARGET_RATIO}_gs-{GROUP_SIZE}-{SEARCH_METHOD}-{DECOMPOSE_METHODS}` directory. Here, the dumped models is stored via the huggingface transformers format. 

## Reference
If you find this work useful, please consider citing our paper:
```
@misc{chang2024palucompressingkvcachelowrank,
      title={Palu: Compressing KV-Cache with Low-Rank Projection}, 
      author={Chi-Chih Chang and Wei-Cheng Lin and Chien-Yu Lin and Chong-Yan Chen and Yu-Fang Hu and Pei-Shuo Wang and Ning-Chi Huang and Luis Ceze and Kai-Chiang Wu},
      year={2024},
      eprint={2407.21118},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.21118}, 
}
```
