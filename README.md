# LLM Glossary

<div align="center">

**Your essential reference for Large Language Models, AI, NLP, and Machine Learning terminology**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Contributions](https://img.shields.io/github/contributors/holasoymalva/llm-glossary.svg)](https://github.com/holasoymalva/llm-glossary/graphs/contributors)

[Explore Terms](#glossary) • [Contributing](#contributing) • [Resources](#additional-resources)

</div>

---

## Overview

The LLM Glossary is a comprehensive, community-driven reference guide for understanding the rapidly evolving landscape of Large Language Models and artificial intelligence. Whether you're a developer building AI applications, a researcher exploring cutting-edge techniques, or an enthusiast learning about generative AI, this glossary provides clear, concise definitions for the concepts that matter.

### Why This Glossary?

- **Always Current**: Community-maintained to keep pace with the fast-moving AI field
- **Practical Focus**: Definitions written for practitioners, not just academics
- **Cross-Referenced**: Terms link to related concepts for deeper understanding
- **Resource-Rich**: Each entry includes links to papers, tutorials, and implementations

## Quick Start

Browse the glossary by category:

- [Core Concepts](#core-concepts) - Foundation terms everyone should know
- [Model Architectures](#model-architectures) - Transformer variants and neural network designs
- [Training & Fine-tuning](#training--fine-tuning) - Methods for optimizing models
- [Inference & Deployment](#inference--deployment) - Production considerations
- [Evaluation & Benchmarks](#evaluation--benchmarks) - Measuring model performance
- [Applications & Use Cases](#applications--use-cases) - Real-world implementations

## Glossary

### Core Concepts

#### Large Language Model (LLM)
A neural network trained on massive text datasets to understand and generate human-like text. LLMs use deep learning architectures (typically Transformers) with billions to trillions of parameters to capture patterns in language.

**Key characteristics:**
- Trained on diverse internet-scale data
- Capable of few-shot and zero-shot learning
- General-purpose language understanding

**Examples**: GPT-4, Claude, Gemini, LLaMA

**Resources**:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Original Transformer paper)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3 paper)

---

#### Token
The fundamental unit of text processed by language models. Tokens are typically subword pieces that represent common character sequences.

**Common tokenization methods:**
- **Byte-Pair Encoding (BPE)**: Merges frequently occurring character pairs
- **WordPiece**: Used by BERT and similar models
- **SentencePiece**: Language-agnostic tokenization

**Example**:
```
Input: "Tokenization is important"
Tokens: ["Token", "ization", " is", " important"]
```

**Related**: Context Window, Vocabulary

---

#### Prompt Engineering
The practice of designing inputs (prompts) to effectively communicate with and guide LLMs toward desired outputs.

**Key techniques:**
- **Zero-shot**: Task description without examples
- **Few-shot**: Including example input-output pairs
- **Chain-of-Thought (CoT)**: Encouraging step-by-step reasoning
- **System Prompts**: Setting model behavior and constraints

**Example**:
```
# Zero-shot
"Translate this to French: Hello, world!"

# Few-shot
"Translate to French:
English: Hello → French: Bonjour
English: Thank you → French: Merci
English: Good morning → French: ?"
```

**Resources**:
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

#### Context Window
The maximum number of tokens an LLM can process in a single interaction, including both input and output.

**Considerations:**
- Larger windows enable processing longer documents
- Computational cost scales quadratically with window size
- Recent models support 128K+ tokens (≈100K words)

**Examples**:
- GPT-4 Turbo: 128K tokens
- Claude 3: 200K tokens
- Gemini 1.5 Pro: 1M tokens

---

### Model Architectures

#### Transformer
The foundational neural network architecture for modern LLMs, introduced in 2017. Uses self-attention mechanisms to process sequences in parallel.

**Key components:**
- **Self-Attention**: Weighs importance of different tokens
- **Feed-Forward Networks**: Transforms representations
- **Positional Encoding**: Captures token position information

**Variants**:
- **Encoder-only**: BERT, RoBERTa (classification tasks)
- **Decoder-only**: GPT, LLaMA (text generation)
- **Encoder-Decoder**: T5, BART (translation, summarization)

---

#### Attention Mechanism
A technique that allows models to focus on relevant parts of the input when processing each token.

**Types**:
- **Self-Attention**: Tokens attend to other tokens in same sequence
- **Cross-Attention**: Tokens attend to separate sequence (e.g., encoder outputs)
- **Multi-Head Attention**: Parallel attention computations

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where Q=Query, K=Key, V=Value, d_k=dimension

---

#### Mixture of Experts (MoE)
An architecture that uses multiple specialized "expert" networks, activating only relevant experts for each input.

**Benefits**:
- Increases model capacity without proportional compute cost
- Each expert can specialize in different domains
- More efficient than dense models at scale

**Examples**: GPT-4, Mixtral, Switch Transformers

---

### Training & Fine-tuning

#### Pre-training
The initial phase where models learn general language understanding from large unlabeled datasets.

**Objectives**:
- **Causal Language Modeling**: Predict next token (GPT-style)
- **Masked Language Modeling**: Predict masked tokens (BERT-style)
- **Denoising**: Reconstruct corrupted text (T5-style)

**Scale**: Trillions of tokens, thousands of GPU-hours

---

#### Fine-tuning
Adapting a pre-trained model to specific tasks or domains using smaller, task-specific datasets.

**Approaches**:
- **Full Fine-tuning**: Update all model parameters
- **Parameter-Efficient Fine-tuning (PEFT)**: Update subset of parameters
- **Instruction Tuning**: Train on task instructions
- **RLHF**: Reinforcement Learning from Human Feedback

---

#### Low-Rank Adaptation (LoRA)
An efficient fine-tuning technique that adds small, trainable rank decomposition matrices to model layers while keeping original weights frozen.

**Benefits**:
- Reduces trainable parameters by 10,000x
- Memory-efficient: multiple adapters can share base model
- Fast training and switching between tasks

**Formula**:
```
W' = W + BA
```
Where W is frozen, B and A are low-rank trainable matrices

**Resources**:
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

#### Reinforcement Learning from Human Feedback (RLHF)
Training approach that uses human preferences to align model outputs with desired behaviors.

**Process**:
1. Collect human comparisons of model outputs
2. Train reward model to predict human preferences
3. Use PPO/DPO to optimize policy against reward model

**Impact**: Critical for models like ChatGPT, Claude, Gemini

---

### Inference & Deployment

#### Quantization
Reducing model precision (e.g., from 32-bit to 8-bit or 4-bit) to decrease memory usage and increase inference speed.

**Methods**:
- **Post-Training Quantization (PTQ)**: Quantize after training
- **Quantization-Aware Training (QAT)**: Train with quantization in mind
- **GPTQ**: Optimal quantization for generative models
- **GGUF**: Efficient format for local inference

**Trade-offs**: Lower precision can reduce quality but enables deployment on consumer hardware

---

#### Temperature
A hyperparameter controlling randomness in text generation.

**Scale**:
- **Low (0.1-0.5)**: Deterministic, focused outputs
- **Medium (0.7-0.8)**: Balanced creativity and coherence
- **High (1.0+)**: More random and creative outputs

**Implementation**: Divides logits before softmax

---

#### Top-k and Top-p Sampling
Techniques to constrain token selection during generation.

**Top-k**: Sample from k most probable tokens
**Top-p (Nucleus)**: Sample from tokens comprising top p probability mass

**Best practice**: Often use together with temperature for quality control

---

#### Model Serving
Infrastructure and techniques for deploying LLMs in production.

**Frameworks**:
- **vLLM**: High-throughput serving with PagedAttention
- **Text Generation Inference (TGI)**: HuggingFace's serving solution
- **TensorRT-LLM**: NVIDIA's optimized serving
- **Ollama**: Local model serving

**Optimizations**: Batching, KV-cache, speculative decoding

---

### Evaluation & Benchmarks

#### Perplexity
A measurement of how well a probability model predicts a sample. Lower perplexity indicates better model performance.

**Formula**:
```
PPL = exp(-1/N Σ log P(token_i))
```

**Note**: Best for comparing models, not absolute quality assessment

---

#### MMLU (Massive Multitask Language Understanding)
Benchmark measuring knowledge across 57 subjects including STEM, humanities, and social sciences.

**Evaluation**: Multiple-choice questions, reports accuracy percentage

---

#### HumanEval
Coding benchmark measuring ability to generate functionally correct Python code from docstrings.

**Metric**: pass@k - percentage of problems solved with k attempts

---

### Applications & Use Cases

#### Retrieval-Augmented Generation (RAG)
Architecture combining LLMs with external knowledge retrieval to ground responses in factual information.

**Architecture**:
1. **Retrieval**: Find relevant documents using vector search
2. **Augmentation**: Add retrieved context to prompt
3. **Generation**: LLM generates response using context

**Benefits**: Reduces hallucinations, enables up-to-date information, domain-specific knowledge

**Tools**: LangChain, LlamaIndex, Haystack

---

#### Function Calling / Tool Use
Capability allowing LLMs to invoke external functions or APIs with structured parameters.

**Use cases**:
- Database queries
- API integrations
- Calculator functions
- Web searches

**Example**:
```json
{
  "name": "get_weather",
  "parameters": {
    "location": "San Francisco",
    "unit": "celsius"
  }
}
```

---

#### Agents
Autonomous systems that use LLMs for reasoning and decision-making to accomplish complex tasks.

**Components**:
- **Planning**: Break down goals into steps
- **Memory**: Maintain context across interactions
- **Tools**: Access to external capabilities
- **Reflection**: Self-evaluation and improvement

**Frameworks**: AutoGPT, BabyAGI, LangGraph, CrewAI

---

#### Embeddings
Dense vector representations of text that capture semantic meaning.

**Applications**:
- Semantic search
- Clustering and classification
- Recommendation systems
- RAG retrieval

**Models**: OpenAI text-embedding-ada-002, Cohere Embed, Sentence-BERT

---

## Contributing

We welcome contributions from the community! Here's how you can help:

### Adding New Terms

1. Fork the repository
2. Create a new branch (`git checkout -b add-new-term`)
3. Add your term following the established format:
   ```markdown
   #### Term Name
   Clear, concise definition (1-2 sentences)
   
   **Key points**:
   - Important detail 1
   - Important detail 2
   
   **Examples/Resources**: Links to papers or implementations
   ```
4. Submit a pull request

### Guidelines

- **Clarity First**: Definitions should be understandable to practitioners
- **Cite Sources**: Link to original papers or authoritative resources
- **Stay Current**: Update outdated information
- **Be Concise**: Respect readers' time
- **Cross-Reference**: Link related terms

## Additional Resources

### Learning Paths

**For Beginners**:
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)

**For Practitioners**:
- [Hugging Face Course](https://huggingface.co/learn)
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/)

**For Researchers**:
- [Papers with Code](https://paperswithcode.com/)
- [Arxiv Sanity](https://arxiv-sanity-lite.com/)

### Community

- [Hugging Face Discord](https://discord.gg/huggingface)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [EleutherAI Discord](https://discord.gg/eleutherai)

### Tools & Frameworks

- **Training**: PyTorch, JAX, DeepSpeed, Megatron-LM
- **Inference**: vLLM, TGI, llama.cpp, Ollama
- **Applications**: LangChain, LlamaIndex, Semantic Kernel
- **Evaluation**: lm-evaluation-harness, HELM

## Roadmap

- [ ] Add interactive search functionality
- [ ] Create visual diagrams for complex concepts
- [ ] Develop multilingual versions
- [ ] Build companion API for programmatic access
- [ ] Add video explanations for key terms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with contributions from developers, researchers, and AI enthusiasts worldwide. Special thanks to:

- The open-source AI community
- Papers with Code for inspiration
- All our contributors

---

<div align="center">

**Star this repo if you find it helpful! ⭐**

Made with ❤️ by the AI community

[Report Issue](https://github.com/holasoymalva/llm-glossary/issues) • [Request Feature](https://github.com/holasoymalva/llm-glossary/issues) • [Discuss](https://github.com/holasoymalva/llm-glossary/discussions)

</div>
