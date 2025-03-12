**1. What should you do if the two models have different tokenizers?**

If the two models have different tokenizers, you should ensure that both models use a consistent tokenization scheme. Otherwise, tokens may misalign, causing unintentional spliting of words or index errors. 

Essentially, the tokenizers splits the text/prompt into seperate chunks to process. If 2 models have different tokenizers, their outputs can't be aligned and compared objectively. The 2 models need to have a common tokenizer implemented and a way to translate each model's own tokenization into this common tokenization.

This can be done through either manually aligning tokenization outputs by mapping tokens from one tokenizer to another (which is usually infeasible for large prompts), using a shared vocabulary by retraining or fine-tuning one model with the other’s tokenizer, or detokenizing and then retokenizing the text with the common tokenizer so that both models process the same input representation.

**2. Do you think contrastive decoding is used in practice?**
In just this interview, the basic model of contrastive decoding took a few minutes to compute a "very very brief docstring". While contrastive decoding has the benefit of utilizing the comparison of 2 different models, it ultimately does use *2* models, making it computational expensive. 
As of now with faster models such as GPT-4o and Claude 3, contrastive decoding doesn't seem like it's being used in practice. However, since contrastive decoding appeals to a need for quality, its comparitive work may be used more in the future. 