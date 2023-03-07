{%hackmd SybccZ6XD %}
# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

###### tags: `paper`

## ABSTRACT

- Image classification tasks
Previous: CNN
Paper: pure transformer

- How to do that
    - Transformer

- Experiment
    - Pre-train and transfer to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.)

## INTRODUCTION

- Step
    - split an image into patches
    - embeddings
    - Transformer
    - MLP
- When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size
    - Transformers lack some of the inductive biases such as translation equivariance and locality 如果每個patch單獨做self-attention呢?
- Trained in larger datasets, excellent result
    - large scale training trumps inductive bias.

## METHOD

follow the original Transformer (Vaswani et al., 2017) as closely as possible

### VISION TRANSFORMER (VIT)

![](https://i.imgur.com/OJwZzwi.png)

### Reshape and Unroll
The input of transformer is a sequence, so reshape the image
![](https://i.imgur.com/CeJEfbL.png)

### Linear Projection and Embedding

$z_0 = [x_{class}; x_p^1E; x_p^2E; ...; x_p^NE] + E_{pos}$
N = 9 in this example
$E\in \mathbb{R}^{(P^2C)\times D}$
$E_{pos}\in \mathbb{R}^{(N+1)\times D}$
![](https://i.imgur.com/Uu0JCAP.png)

Embedding code
```python=
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
x + = self.pos_embedding(:,:(n + 1))
```

class token
![](https://i.imgur.com/HRHu9I1.png)
similar to BERT (initial zero)
![](https://i.imgur.com/QuzUK8d.png)

### Transformer Encoder

![](https://i.imgur.com/dqZbyIA.png)

[Transformer](https://hackmd.io/11069bzHTUyQU1ClW9_TMQ)

### MLP

Fully connected and activation function

### FINE-TUNING AND HIGHER RESOLUTION

Pre-train on large datasets, and fine-tune to smaller task.
Step
- Pre-train
    - predition head: MLP with one hidden layer
    ![](https://i.imgur.com/FL0P5vh.png)
- Remove pre-trained predition head
- Attach a zero-initialized D x K feedforward layer
![](https://i.imgur.com/lfvlx7a.png)

## EXPERIMENTS

### SETUP

==Datasets.==

Pre-train
- ILSVRC-2012 ImageNet dataset: 1k classes and 1.3M images
- ImageNet-21k: 21k classes and 14M images
- JFT: 18k classes and 303M high resolution images

benchmark tasks: (Prepross by Big transfer (BiT): General visual representation learning.)
- ImageNet on the original validation labels and the cleaned-up ReaL labels
- CIFAR-10/100
- Oxford-IIIT Pets
- Oxford Flowers-102

19-task VTAB classification suite:
- Natural：tasks like the above, Pets, CIFAR
- Specialized：medical and satellite imagery
- Structured：tasks that require geometric understanding like localization

==Model Variants.==
Layers: How many encoder block
Hidden size D: The dim of output of linear projection
MLP size: 
Heads: How many head in the Multi-Head Attention
![](https://i.imgur.com/uFPvBii.png)

==Training & Fine-tuning.==

Optimization
Adam: $\beta_1 = 0.9, \beta_2 = 0.999, batch size = 4096, high weight decay of 0.1$

Fine-tune
SGD with momentum: $batch size = 512$

==Metrics.==

results
- few-shot accuracy: solving a regularized least-squares regression problem that maps the (frozen) representation of a subset of training images to $\{-1,1\}^K$ target vectors
- fine-tuning accuracy: after fine-tuning

:::warning
補充 (shot)
n-shot: 1 class have n samples 
:::

### COMPARISON TO STATE OF THE ART

![](https://i.imgur.com/L4OgBf8.png)
![](https://i.imgur.com/vkSKYBP.png)

### PRE-TRAINING DATA REQUIREMENTS

The comparison between small datasets and larger datasets when pre-trained.
- large ViT models perform worse than BiT
- large ViT models shine when pre-trained on larger datasets.
![](https://i.imgur.com/VO1Isoe.png)
The number in the above picture means P
![](https://i.imgur.com/PRHaLtL.png)


The comparison between different size of subset
- use early-stopping, and report the best validation accuracy achieved during training
![](https://i.imgur.com/2c8Brn0.png)

### SCALING STUDY

![](https://i.imgur.com/kvbHu7t.png)

### INSPECTING VISION TRANSFORMER

### SELF-SUPERVISION


## my result

randomcrop and resize
0.9737 3:03:00
resize
0.9823 2:58:00

me resize
0.9874 2:20:00