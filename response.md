**1. What should you do if the two models have different tokenizers?**

If the two models have different tokenizers, you should ensure that both models use a consistent tokenization scheme. Otherwise, tokens may misalign, causing unintentional spliting of workds or index errors. 
A consistent scheme can be done by selecting a common tokenizer and either:

- Manually aligning tokenization outputs by mapping tokens from one tokenizer to another.
- Using a shared vocabulary by retraining or fine-tuning one model with the other’s tokenizer.
- Detokenizing and then retokenizing the text so that both models process the same input representation.

**2. Do you think contrastive decoding is used in practice?**

Contrastive decoding is an interesting approach that improves text generation by comparing outputs from a strong and a weak model, filtering out generic or low-quality responses. While it’s promising, it’s not yet mainstream in large-scale applications due to computational costs and integration complexity. However, it has potential in research settings, creative text generation, and areas where reducing repetitive or degenerate outputs is important. As more efficient implementations emerge, contrastive decoding could see broader adoption in real-world NLP systems.