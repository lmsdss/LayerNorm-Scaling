# The Curse of Depth in Large Language Models

We present the Curse of Depth, a phenomenon in Large Language Models (LLMs) where deeper layers contribute less effectively to training due to the widespread use of Pre-Layer Normalization (Pre-LN). Our analysis identifies this issue as a key bottleneck in LLM optimization and proposes LayerNorm Scaling as a solution to mitigate its impact.

<div align="center">
  <img src="scaling.png" alt="Image 2" style="width: 900px; margin: 0 auto;">
</div>


## Abstract

In this paper, we introduce the Curse of Depth, a concept that highlights, explains, and addresses the recent observation in modern Large Language Models (LLMs) where nearly half of the layers are less effective than expected. We first confirm the wide existence of this phenomenon across the most popular families of LLMs such as Llama, Mistral, DeepSeek, and Qwen. Our analysis, theoretically and empirically, identifies that the underlying reason for the ineffectiveness of deep layers in LLMs is the widespread usage of Pre-Layer Normalization (Pre-LN). While Pre-LN stabilizes the training of Transformer LLMs, its output variance exponentially grows with the model depth, which undesirably causes the derivative of the deep Transformer blocks to be an identity matrix, and therefore barely contributes to the training. To resolve this training pitfall, we propose LayerNorm Scaling, which scales the variance of output of the layer normalization inversely by the square root of its depth. This simple modification mitigates the output variance explosion of deeper Transformer layers, improving their contribution. Our experimental results, spanning model sizes from 130M to 1B, demonstrate that LayerNorm Scaling significantly enhances LLM pre-training performance compared to Pre-LN. Moreover, this improvement seamlessly carries over to supervised fine-tuning. All these gains can be attributed to the fact that LayerNorm Scaling enables deeper layers to contribute more effectively during training.

## Quick Start

### Install experiment dependencies

You can configure the environment using the following command lines:

```bash
conda create -n cod python=3.9 -y
conda activate cod
pip install -r exp_requirements.txt
```

### Training Examples
We provide scripts to train models of different sizes using Pre-LN, Post-LN, Mix-LN, and LayerNorm Scaling (CoD).

Train a 130M Model:
```bash
bash run_130m.sh pre      3   # Pre-LN
bash run_130m.sh post     3   # Post-LN
bash run_130m.sh post_pre 3   # Mix-LN
bash run_130m.sh cod      3   # LayerNorm Scaling (CoD)

(Note: 3 represents the number of Post-LN layers in Mix-LN.)
```


Train a 250M Mode:
```bash
bash run_250m.sh pre      6   # Pre-LN
bash run_250m.sh post     6   # Post-LN
bash run_250m.sh post_pre 6   # Mix-LN
bash run_250m.sh cod      6   # LayerNorm Scaling (CoD)

(Note: 6 represents the number of Post-LN layers in Mix-LN.)
```

Train a 350M Mode:
```bash
bash run_350m.sh pre      6   # Pre-LN
bash run_350m.sh post     6   # Post-LN
bash run_350m.sh post_pre 6   # Mix-LN
bash run_350m.sh cod      6   # LayerNorm Scaling (CoD)
```

Train a 1B Mode:
```bash
bash run_1b.sh pre        6   # Pre-LN
bash run_1b.sh post       6   # Post-LN
bash run_1b.sh post_pre   6   # Mix-LN
bash run_1b.sh cod        6   # LayerNorm Scaling (CoD)
```

### Performance Drop
Calculate the performance drop after removing different layers. We use [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) to obtain evaluation results. Please refer to its installation instructions to configure `lm_eval``.
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Then, you can run the following command to remove different layers and save the weights to a new model. The performance drop will be calculated based on the new model:
```bash
# LLaMA2-7B, Remove Layer 1
python layer_remove.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --layer_index 1 \
    --save_path ./llama_7b_removed_1
```


### Acknowledgement
This repository is built upon the [Mix-LN](https://github.com/pixeli99/MixLN/tree/main) repositories. Thanks for their great work!
