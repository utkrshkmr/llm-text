# References and Further Reading {#app:refs}

This appendix collects the works cited across the book: the papers behind the models, algorithms and
datasets, and the systems papers and open-source projects behind the Systems Lens material. Entries
are sorted alphabetically by first author.

## Staying current {#staying-current .unnumbered}

The field moves quickly. Good primary sources include arXiv (cs.CL, cs.LG, cs.DC), the proceedings
of MLSys, OSDI/SOSP, ISCA, NeurIPS, ICML and ICLR, and the documentation and design notes of the
open-source systems discussed in this book: PyTorch, vLLM, SGLang, TensorRT-LLM, Megatron-LM,
DeepSpeed, veRL, OpenRLHF, TRL, JAX and Triton.

## Cited works {#cited-works .unnumbered}

::: enumerate
Abbas, A., Tirumala, K., Simig, D., Ganguli, S., Morcos, A. S. (2023). SemDeDup: Data-Efficient
Learning at Web-Scale through Semantic Deduplication. arXiv:2303.09540.

Abdin, M., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your
Phone. arXiv:2404.14219.

Abts, D., Ross, J., Sparling, J., et al. (2020). Think Fast: A Tensor Streaming Processor (TSP) for
Accelerating Deep Learning Workloads. ISCA.

Achiam, J. (2018). Spinning Up in Deep Reinforcement Learning. OpenAI.

Agache, A., Brooker, M., Florescu, A., et al. (2020). Firecracker: Lightweight Virtualization for
Serverless Applications. NSDI.

Agarwal, R., Vieillard, N., Zhou, Y., et al. (2024). On-Policy Distillation of Language Models:
Learning from Self-Generated Mistakes (GKD). ICLR.

Agrawal, A. et al. (2024). Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.
OSDI.

Ahmadian, A., Cremer, C., Galle, M., et al. (2024). Back to Basics: Revisiting REINFORCE Style
Optimization for Learning from Human Feedback in LLMs. ACL.

Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., Sanghai, S. (2023). GQA:
Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP.

Alayrac, J.-B., Donahue, J., Luc, P., et al. (2022). Flamingo: a Visual Language Model for Few-Shot
Learning. NeurIPS.

Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the Study of SMS Spam
Filtering: New Collection and Results. ACM Symposium on Document Engineering (DocEng).

Andriushchenko, M., D'Angelo, F., Varre, A., Flammarion, N. (2023). Why Do We Need Weight Decay in
Modern Deep Learning? arXiv:2310.04415.

Ansel, J., Yang, E., He, H., et al. (2024). PyTorch 2: Faster Machine Learning Through Dynamic
Python Bytecode Transformation and Graph Compilation. ASPLOS.

Arora, S., Eyuboglu, S., Timalsina, A., et al. (2023). Zoology: Measuring and Improving Recall in
Efficient Language Models. arXiv:2312.04927.

Arora, S., Eyuboglu, S., Zhang, M., et al. (2024). Simple Linear Attention Language Models Balance
the Recall-Throughput Tradeoff. ICML.

Austin, J., Douglas, S., Frostig, R., et al. (2025). How to Scale Your Model: A Systems View of LLMs
on TPUs. Google DeepMind (online book).

Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M., Munos, R. (2023). A
General Theoretical Paradigm to Understand Learning from Human Preferences. arXiv:2310.12036
(AISTATS 2024).

Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450.

Badertdinov, I., Golubev, A., Nekrashevich, M., et al. (2025). SWE-rebench: An Automated Pipeline
for Task Collection and Decontaminated Evaluation of Software Engineering Agents. arXiv:2505.20411.

Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align
and Translate. ICLR.

Bahri, Y., Dyer, E., Kaplan, J., Lee, J., Sharma, U. (2021). Explaining Neural Scaling Laws.
arXiv:2102.06701.

Bai, J., Bai, S., Yang, S., et al. (2023). Qwen-VL: A Versatile Vision-Language Model for
Understanding, Localization, Text Reading, and Beyond. arXiv:2308.12966.

Bai, Y., Jones, A., Ndousse, K., et al. (2022). Training a Helpful and Harmless Assistant with
Reinforcement Learning from Human Feedback. arXiv:2204.05862.

Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback.
arXiv:2212.08073.

Baydin, A. G., Pearlmutter, B. A., Radul, A. A., Siskind, J. M. (2018). Automatic Differentiation in
Machine Learning: a Survey. JMLR.

Beirami, A., Agarwal, A., Berant, J., et al. (2024). Theoretical Guarantees on the Best-of-n
Alignment Policy. arXiv:2401.01879.

Beltagy, I., Peters, M. E., Cohan, A. (2020). Longformer: The Long-Document Transformer.
arXiv:2004.05150.

Besiroglu, T., Erdil, E., Barnett, M., You, J. (2024). Chinchilla Scaling: A Replication Attempt.
arXiv:2404.10102.

Bianchi, F., Suzgun, M., Attanasio, G., Röttger, P., Jurafsky, D., Hashimoto, T., Zou, J. (2024).
Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow
Instructions. ICLR.

Blelloch, G. E. (1990). Prefix Sums and Their Applications. Technical Report CMU-CS-90-190.

Bloom, B. H. (1970). Space/Time Trade-offs in Hash Coding with Allowable Errors. Communications of
the ACM.

Bradbury, J., Frostig, R., Hawkins, P., et al. (2018). JAX: Composable Transformations of
Python+NumPy Programs. <http://github.com/google/jax>

Bradley, R. A., and Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of
Paired Comparisons. Biometrika.

Broder, A. Z. (1997). On the Resemblance and Containment of Documents. Compression and Complexity of
Sequences.

Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. NeurIPS.

Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). Medusa: Simple LLM
Inference Acceleration Framework with Multiple Decoding Heads. ICML.

Cannon, L. E. (1969). A Cellular Computer to Implement the Kalman Filter Algorithm. PhD thesis,
Montana State University.

Chainlit contributors (2023). Chainlit: Build conversational AI applications in Python.
<https://github.com/Chainlit/chainlit>.

Chameleon Team (2024). Chameleon: Mixed-Modal Early-Fusion Foundation Models. arXiv:2405.09818.

Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating
Large Language Model Decoding with Speculative Sampling. arXiv:2302.01318.

Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2024). Punica: Multi-Tenant LoRA
Serving. MLSys.

Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code.
arXiv:2107.03374.

Chen, S., Wong, S., Chen, L., Tian, Y. (2023). Extending Context Window of Large Language Models via
Positional Interpolation. arXiv:2306.15595.

Chen, T., Xu, B., Zhang, C., Guestrin, C. (2016). Training Deep Nets with Sublinear Memory Cost.
arXiv:1604.06174.

Chen, X., Liang, C., Huang, D., et al. (2023). Symbolic Discovery of Optimization Algorithms (Lion).
NeurIPS.

Chen, Y.-H., Emer, J., Sze, V. (2016). Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow
for Convolutional Neural Networks. ISCA.

Chen, Z., Deng, Y., Yuan, H., Ji, K., Gu, Q. (2024). Self-Play Fine-Tuning Converts Weak Language
Models to Strong Language Models. ICML.

Chiang, W.-L., Zheng, L., Sheng, Y., et al. (2024). Chatbot Arena: An Open Platform for Evaluating
LLMs by Human Preference. ICML.

Child, R., Gray, S., Radford, A., Sutskever, I. (2019). Generating Long Sequences with Sparse
Transformers. arXiv:1904.10509.

Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical
Machine Translation. EMNLP.

Chollet, F. (2019). On the Measure of Intelligence. arXiv:1911.01547.

Choromanski, K., Likhosherstov, V., Dohan, D., et al. (2021). Rethinking Attention with Performers.
ICLR.

Chowdhery, A., Narang, S., Devlin, J., et al. (2022). PaLM: Scaling Language Modeling with Pathways.
JMLR 2023 (arXiv:2204.02311).

Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., Amodei, D. (2017). Deep Reinforcement
Learning from Human Preferences. NeurIPS.

Chung, H. W., Constant, N., Garcia, X., et al. (2023). UniMax: Fairer and More Effective Language
Sampling for Large-Scale Multilingual Pretraining. ICLR.

Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training Verifiers to Solve Math Word
Problems. arXiv:2110.14168.

Cohen, J. M., Kaur, S., Li, Y., Kolter, J. Z., Talwalkar, A. (2021). Gradient Descent on Neural
Networks Typically Occurs at the Edge of Stability. ICLR.

Cui, G., Yuan, L., Ding, N., et al. (2023). UltraFeedback: Boosting Language Models with
High-quality Feedback. arXiv:2310.01377 (ICML 2024).

D'Oosterlinck, K., Xu, W., Develder, C., Demeester, T., Singh, A., Potts, C., Kiela, D., Mehri, S.
(2024). Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in
Alignment. arXiv:2408.06266.

Dai, D., Deng, C., Zhao, C., et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in
Mixture-of-Experts Language Models. ACL.

Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
arXiv:2307.08691 (ICLR 2024).

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness. NeurIPS.

Dao, T., Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through
Structured State Space Duality. ICML.

Dauphin, Y. N., Fan, A., Auli, M., Grangier, D. (2017). Language Modeling with Gated Convolutional
Networks. ICML.

DeepSeek-AI (2024). DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.
arXiv:2401.02954.

DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
Model. arXiv:2405.04434.

DeepSeek-AI (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.

DeepSeek-AI (2025). DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. Technical
report.

DeepSeek-AI (2025). DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning.
arXiv:2511.22570.

DeepSeek-AI, Guo, D., Yang, D., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in
LLMs via Reinforcement Learning. arXiv:2501.12948.

Degris, T., White, M., Sutton, R. S. (2012). Off-Policy Actor-Critic. ICML.

Dehghani, M., Djolonga, J., Mustafa, B., et al. (2023). Scaling Vision Transformers to 22 Billion
Parameters. ICML.

Dettmers, T., & Zettlemoyer, L. (2023). The case for 4-bit precision: k-bit Inference Scaling Laws.
ICML.

Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix
Multiplication for Transformers at Scale. NeurIPS.

Dettmers, T., Lewis, M., Shleifer, S., Zettlemoyer, L. (2022). 8-bit Optimizers via Block-wise
Quantization. ICLR.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of
Quantized LLMs. NeurIPS.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding. NAACL.

Ding, N., Chen, Y., Xu, B., et al. (2023). Enhancing Chat Language Models by Scaling High-quality
Instructional Conversations (UltraChat). EMNLP.

Dodge, J., Sap, M., Marasovic, A., et al. (2021). Documenting Large Webtext Corpora: A Case Study on
the Colossal Clean Crawled Corpus. EMNLP.

Dong, H., Xiong, W., Goyal, D., et al. (2023). RAFT: Reward rAnked FineTuning for Generative
Foundation Model Alignment. TMLR.

Dong, Y., Ruan, C. F., Cai, Y., Lai, R., Xu, Z., Zhao, Y., & Chen, T. (2024). XGrammar: Flexible and
Efficient Structured Generation Engine for Large Language Models. arXiv:2411.15100.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale. ICLR.

Dósa, G. (2007). The Tight Bound of First Fit Decreasing Bin-Packing Algorithm Is FFD(I) \<= 11/9
OPT(I) + 6/9. ESCAPE, LNCS 4614.

Dubois, Y., Galambosi, B., Liang, P., Hashimoto, T. B. (2024). Length-Controlled AlpacaEval: A
Simple Way to Debias Automatic Evaluators. arXiv:2404.04475.

Dubois, Y., Li, X., Taori, R., et al. (2023). AlpacaFarm: A Simulation Framework for Methods that
Learn from Human Feedback. NeurIPS.

Duchi, J., Hazan, E., Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and
Stochastic Optimization. JMLR.

Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics.

Eldan, R., Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent
English? arXiv:2305.07759.

Elhoushi, M., et al. (2024). LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.
ACL.

Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., Madry, A. (2020).
Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. ICLR.

Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., Kiela, D. (2024). KTO: Model Alignment as
Prospect Theoretic Optimization. ICML.

Fedus, W., Zoph, B., Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models
with Simple and Efficient Sparsity. JMLR.

Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate Post-Training
Quantization for Generative Pre-trained Transformers. ICLR.

Frantar, E., Castro, R. L., Chen, J., Hoefler, T., & Alistarh, D. (2025). MARLIN: Mixed-Precision
Auto-Regressive Parallel Inference on Large Language Models. PPoPP.

Fu, W., Gao, J., Shen, X., et al. (2025). AReaL: A Large-Scale Asynchronous Reinforcement Learning
System for Language Reasoning. arXiv:2505.24298.

Fu, Y., Bailis, P., Stoica, I., & Zhang, H. (2024). Break the Sequential Dependency of LLM Inference
Using Lookahead Decoding. ICML.

Gage, P. (1994). A New Algorithm for Data Compression. The C Users Journal.

Gale, T., Narayanan, D., Young, C., Zaharia, M. (2023). MegaBlocks: Efficient Sparse Training with
Mixture-of-Experts. MLSys.

Ganguli, D., Lovitt, L., Kernion, J., et al. (2022). Red Teaming Language Models to Reduce Harms:
Methods, Scaling Behaviors, and Lessons Learned. arXiv:2209.07858.

Gao, L., Biderman, S., Black, S., et al. (2020). The Pile: An 800GB Dataset of Diverse Text for
Language Modeling. arXiv:2101.00027.

Gao, L., Schulman, J., Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. ICML.

GDPO (2026). Group reward-Decoupled Normalization Policy Optimization for multi-reward RL. arXiv
preprint.

Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size. arXiv:2408.00118.

Glorot, X., Bengio, Y. (2010). Understanding the Difficulty of Training Deep Feedforward Neural
Networks. AISTATS.

Gokaslan, A., Cohen, V. (2019). OpenWebText Corpus. <http://Skylion007.github.io/OpenWebTextCorpus>.

Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.

Greensmith, E., Bartlett, P. L., Baxter, J. (2004). Variance Reduction Techniques for Gradient
Estimates in Reinforcement Learning. JMLR.

Griewank, A., Walther, A. (2000). Algorithm 799: Revolve: An Implementation of Checkpointing for the
Reverse or Adjoint Mode of Computational Differentiation. ACM Transactions on Mathematical Software.

Gu, A., Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
arXiv:2312.00752.

Gu, A., Goel, K., Re, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces.
ICLR.

Gu, Y., Dong, L., Wei, F., Huang, M. (2024). MiniLLM: Knowledge Distillation of Large Language
Models. ICLR.

Guha, E., Marten, R., Keh, S., et al. (2025). OpenThoughts: Data Recipes for Reasoning Models.
arXiv:2506.04178.

Gulcehre, C., Le Paine, T., Srinivasan, S., et al. (2023). Reinforced Self-Training (ReST) for
Language Modeling. arXiv:2308.08998.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks.
ICML.

Gupta, V., Koren, T., Singer, Y. (2018). Shampoo: Preconditioned Stochastic Tensor Optimization.
ICML.

Hashimoto, T. (2021). Model Performance Scaling with Multiple Data Sources. ICML.

Hägele, A., Bakouch, E., Kosson, A., et al. (2024). Scaling Laws and Compute-Optimal Training beyond
Fixed Training Durations. NeurIPS.

He, H., and Thinking Machines Lab (2025). Defeating Nondeterminism in LLM Inference. Thinking
Machines Lab: Connectionism (blog).

He, K., Zhang, X., Ren, S., Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
Performance on ImageNet Classification. ICCV.

He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.

Hendrycks, D., Burns, C., Basart, S., et al. (2021). Measuring Massive Multitask Language
Understanding. ICLR.

Hendrycks, D., Burns, C., Kadavath, S., et al. (2021). Measuring Mathematical Problem Solving With
the MATH Dataset. NeurIPS Datasets and Benchmarks.

Hendrycks, D., Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415.

Hestness, J., Narang, S., Ardalani, N., et al. (2017). Deep Learning Scaling is Predictable,
Empirically. arXiv:1712.00409.

Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the Knowledge in a Neural Network. NeurIPS Deep
Learning Workshop. arXiv:1503.02531.

Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language
Models. NeurIPS.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2020). The Curious Case of Neural Text
Degeneration. ICLR.

Hong, J., Lee, N., Thorne, J. (2024). ORPO: Monolithic Preference Optimization without Reference
Model. EMNLP.

Houlsby, N., Giurgiu, A., Jastrzebski, S., et al. (2019). Parameter-Efficient Transfer Learning for
NLP. ICML.

Hsu, P.-L., Dai, Y., Kothapalli, V., et al. (2024). Liger Kernel: Efficient Triton Kernels for LLM
Training. arXiv:2410.10989.

Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
ICLR.

Hu, J., Wu, X., Zhu, Z., et al. (2024). OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF
Framework. arXiv:2405.11143.

Hu, S., Tu, Y., Han, X., et al. (2024). MiniCPM: Unveiling the Potential of Small Language Models
with Scalable Training Strategies. arXiv:2404.06395.

Huang, J., Chen, X., Mishra, S., et al. (2024). Large Language Models Cannot Self-Correct Reasoning
Yet. ICLR.

Huang, S., Noukhovitch, M., Hosseini, A., Rasul, K., Wang, W., Tunstall, L. (2024). The N+
Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization. arXiv:2403.17031.

Huang, Y., Cheng, Y., Bapna, A., et al. (2019). GPipe: Efficient Training of Giant Neural Networks
using Pipeline Parallelism. NeurIPS.

Huber, P. J. (1964). Robust Estimation of a Location Parameter. Annals of Mathematical Statistics.

Hunter, D. R. (2004). MM Algorithms for Generalized Bradley-Terry Models. The Annals of Statistics.

Inan, H., Upasani, K., Chi, J., et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for
Human-AI Conversations. arXiv:2312.06674.

Ivanov, A., Dryden, N., Ben-Nun, T., Li, S., Hoefler, T. (2021). Data Movement Is All You Need: A
Case Study on Optimizing Transformers. MLSys.

Jacobs, S. A., Tanaka, M., Zhang, C., et al. (2023). DeepSpeed Ulysses: System Optimizations for
Enabling Training of Extreme Long Sequence Transformer Models. arXiv:2309.14509.

Jiang, A. Q., Sablayrolles, A., Roux, A., et al. (2024). Mixtral of Experts. arXiv:2401.04088.

Jimenez, C. E., Yang, J., Wettig, A., et al. (2024). SWE-bench: Can Language Models Resolve
Real-World GitHub Issues? ICLR.

Johnson, D. S. (1973). Near-Optimal Bin Packing Algorithms. PhD thesis, MIT.

Jordan, K., Jin, Y., Boza, V., et al. (2024). Muon: An optimizer for hidden layers in neural
networks. Blog post, kellerjordan.github.io.

Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2017). Bag of Tricks for Efficient Text
Classification. EACL.

Jouppi, N. P., Kurian, G., Li, S., et al. (2023). TPU v4: An Optically Reconfigurable Supercomputer
for Machine Learning with Hardware Support for Embeddings. ISCA.

Jouppi, N. P., Young, C., Patil, N., et al. (2017). In-Datacenter Performance Analysis of a Tensor
Processing Unit. ISCA.

Kahneman, D., Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. Econometrica.

Kakade, S. (2002). A Natural Policy Gradient. NeurIPS.

Kalamkar, D., et al. (2019). A Study of BFLOAT16 for Deep Learning Training. arXiv:1905.12322.

Kandpal, N., Lester, B., Raffel, C., et al. (2025). The Common Pile v0.1: An 8TB Dataset of Public
Domain and Openly Licensed Text. arXiv:2506.05209.

Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural Language Models.
arXiv:2001.08361.

Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F. (2020). Transformers are RNNs: Fast
Autoregressive Transformers with Linear Attention. ICML.

Kazemnejad, A., Padhi, I., Natesan Ramamurthy, K., Das, P., Reddy, S. (2023). The Impact of
Positional Encoding on Length Generalization in Transformers. NeurIPS.

Kim, D., Kim, Y., Song, W., Kim, H., Kim, Y., Kim, S., Park, C. (2024). sDPO: Don't Use Your Data
All at Once. arXiv:2403.19270.

Kimi Team (2025). Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv:2501.12599.

Kingma, D. P., Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.

Kocetkov, D., Li, R., Ben Allal, L., et al. (2022). The Stack: 3 TB of Permissively Licensed Source
Code. arXiv:2211.15533.

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. (2022). Large Language Models are
Zero-Shot Reasoners. NeurIPS.

Komatsuzaki, A., Puigcerver, J., Lee-Thorp, J., et al. (2023). Sparse Upcycling: Training
Mixture-of-Experts from Dense Checkpoints. ICLR.

Kool, W., van Hoof, H., Welling, M. (2019). Buy 4 REINFORCE Samples, Get a Baseline for Free! ICLR
Workshop (DeepRLStructPred).

Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., Catanzaro, B. (2022).
Reducing Activation Recomputation in Large Transformer Models. arXiv:2205.05198 (MLSys 2023).

Krell, M. M., Kosec, M., Perez, S. P., & Fitzgibbon, A. (2021). Efficient Sequence Packing without
Cross-contamination: Accelerating Large Language Models without Impacting Performance.
arXiv:2107.02027.

Kung, H. T. (1982). Why Systolic Architectures? IEEE Computer.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica,
I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP.

Lambert, N. (2024). Reinforcement Learning from Human Feedback. <https://rlhfbook.com>.

Lambert, N., Pyatkin, V., Morrison, J., et al. (2024). RewardBench: Evaluating Reward Models for
Language Modeling. arXiv:2403.13787.

Lee, H., Phatale, S., Mansoor, H., et al. (2024). RLAIF vs. RLHF: Scaling Reinforcement Learning
from Human Feedback with AI Feedback. ICML.

Lee, K., Ippolito, D., Nystrom, A., et al. (2022). Deduplicating Training Data Makes Language Models
Better. ACL.

Lepikhin, D., Lee, H., Xu, Y., et al. (2021). GShard: Scaling Giant Models with Conditional
Computation and Automatic Sharding. ICLR.

Leskovec, J., Rajaraman, A., Ullman, J. D. (2014). Mining of Massive Datasets (2nd ed.), Chapter 3:
Finding Similar Items. Cambridge University Press.

Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt
Tuning. EMNLP.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative
Decoding. ICML.

Li, B., Zhang, Y., Guo, D., et al. (2024). LLaVA-OneVision: Easy Visual Task Transfer.
arXiv:2408.03326.

Li, H., et al. (2025). Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large
Language Model Pretraining. arXiv preprint.

Li, J., Fang, A., Smyrnis, G., et al. (2024). DataComp-LM: In Search of the Next Generation of
Training Sets for Language Models. NeurIPS Datasets and Benchmarks.

Li, J., Li, D., Savarese, S., Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with
Frozen Image Encoders and Large Language Models. ICML.

Li, S., Zhao, Y., Varma, R., et al. (2020). PyTorch Distributed: Experiences on Accelerating Data
Parallel Training. VLDB.

Li, X. L., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL.

Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., Hashimoto, T. B.
(2023). AlpacaEval: An Automatic Evaluator of Instruction-following Models. GitHub repository.

Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024). EAGLE: Speculative Sampling Requires Rethinking
Feature Uncertainty. ICML.

Liang, W., Liu, T., Wright, L., et al. (2024). TorchTitan: One-stop PyTorch Native Solution for
Production Ready LLM Pre-training. arXiv:2410.06511.

Lie, S. (2023). Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design
for Deep Learning. IEEE Micro.

Lieber, O., Lenz, B., Bata, H., et al. (2024). Jamba: A Hybrid Transformer-Mamba Language Model.
arXiv:2403.19887.

Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). Let's Verify Step by Step. arXiv:2305.20050
(source of the MATH-500 split).

Lin, B. Y., Deng, Y., Chandu, K., et al. (2024). WildBench: Benchmarking LLMs with Challenging Tasks
from Real Users in the Wild. arXiv:2406.04770.

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., & Han,
S. (2024). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys.

Little, J. D. C. (1961). A Proof for the Queuing Formula L = lambda W. Operations Research.

Liu, H., Li, C., Li, Y., Lee, Y. J. (2024). Improved Baselines with Visual Instruction Tuning. CVPR.

Liu, H., Li, C., Wu, Q., Lee, Y. J. (2023). Visual Instruction Tuning. NeurIPS.

Liu, H., Tam, D., Muqeeth, M., et al. (2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and
Cheaper than In-Context Learning ((IA)\^3). NeurIPS.

Liu, H., Zaharia, M., Abbeel, P. (2023). Ring Attention with Blockwise Transformers for
Near-Infinite Context. arXiv:2310.01889 (ICLR 2024).

Liu, Q., Zheng, X., Muennighoff, N., et al. (2024). RegMix: Data Mixture as Regression for Language
Model Pre-training. arXiv:2407.01492.

Liu, S.-Y., Wang, C.-Y., Yin, H., et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. ICML.

Liu, T., Zhao, Y., Joshi, R., Khalman, M., Saleh, M., Liu, P. J., Liu, J. (2023). Statistical
Rejection Sampling Improves Preference Optimization. arXiv:2309.06657 (ICLR 2024).

Liu, Z., Chen, C., Li, W., et al. (2025). Understanding R1-Zero-Like Training: A Critical
Perspective (Dr. GRPO). arXiv:2503.20783.

Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., & Hu, X. (2024). KIVI: A
Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML.

Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.

Lozhkov, A., Li, R., Ben Allal, L., et al. (2024). StarCoder 2 and The Stack v2: The Next
Generation. arXiv:2402.19173.

Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative Refinement with
Self-Feedback. NeurIPS.

Magnusson, I., Bhagia, A., Hofmann, V., et al. (2023). Paloma: A Benchmark for Evaluating Language
Model Fit. arXiv:2312.10523.

Mazeika, M., Phan, L., Yin, X., et al. (2024). HarmBench: A Standardized Evaluation Framework for
Automated Red Teaming and Robust Refusal. ICML.

McCandlish, S., Kaplan, J., Amodei, D., OpenAI Dota Team (2018). An Empirical Model of Large-Batch
Training. arXiv:1812.06162.

Meng, L., Yang, J., Tian, R., et al. (2024). DeepStack: Deeply Stacking Visual Tokens is
Surprisingly Simple and Effective for LMMs. arXiv:2406.04334.

Meng, Y., Xia, M., Chen, D. (2024). SimPO: Simple Preference Optimization with a Reference-Free
Reward. NeurIPS.

Meta AI (2025). The Llama 4 Herd: The Beginning of a New Era of Natively Multimodal AI Innovation.
Meta AI blog.

Miao, X., et al. (2024). SpecInfer: Accelerating Large Language Model Serving with Tree-based
Speculative Inference and Verification. ASPLOS.

Micikevicius, P., et al. (2022). FP8 Formats for Deep Learning. arXiv:2209.05433.

Micikevicius, P., Narang, S., Alben, J., et al. (2018). Mixed Precision Training. ICLR.

Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). Efficient Estimation of Word Representations in
Vector Space. ICLR Workshop.

Milakov, M., & Gimelshein, N. (2018). Online normalizer calculation for softmax. arXiv:1805.02867.

MiniMax (2025). MiniMax-01: Scaling Foundation Models with Lightning Attention. arXiv:2501.08313.

MiniMax (2025). MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (CISPO).
arXiv:2506.13585.

Moore, R. C., Lewis, W. (2010). Intelligent Selection of Language Model Training Data. ACL.

Moritz, P., Nishihara, R., Wang, S., et al. (2018). Ray: A Distributed Framework for Emerging AI
Applications. OSDI.

Muennighoff, N., Rush, A. M., Barak, B., et al. (2023). Scaling Data-Constrained Language Models.
NeurIPS.

Muennighoff, N., Soldaini, L., Groeneveld, D., et al. (2024). OLMoE: Open Mixture-of-Experts
Language Models. arXiv:2409.02060.

Muennighoff, N., Yang, Z., Shi, W., et al. (2025). s1: Simple Test-Time Scaling. arXiv:2501.19393.

Munos, R., et al. (2023). Nash Learning from Human Feedback. arXiv:2312.00886 (ICML 2024).

Narang, S., Chung, H. W., Tay, Y., et al. (2021). Do Transformer Modifications Transfer Across
Implementations and Applications? EMNLP.

Narayanan, D., Harlap, A., Phanishayee, A., et al. (2019). PipeDream: Generalized Pipeline
Parallelism for DNN Training. SOSP.

Narayanan, D., Shoeybi, M., Casper, J., et al. (2021). Efficient Large-Scale Language Model Training
on GPU Clusters Using Megatron-LM. SC.

NVIDIA (2022). NVIDIA H100 Tensor Core GPU Architecture. Whitepaper.

NVIDIA (2025). Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models.
arXiv:2504.03624.

Olsson, C., et al. (2022). In-context Learning and Induction Heads. Transformer Circuits Thread.

OpenAI (2024). Learning to Reason with LLMs (OpenAI o1). OpenAI blog.

OpenAI (2024). Learning to Reason with LLMs. OpenAI blog post (o1 announcement).

Oren, Y., Meister, N., Chatterji, N., Ladhak, F., Hashimoto, T. B. (2024). Proving Test Set
Contamination in Black Box Language Models. ICLR.

Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with
Human Feedback. NeurIPS.

Pagnoni, A., et al. (2024). Byte Latent Transformer: Patches Scale Better Than Tokens.
arXiv:2412.09871.

Pang, R. Y., Yuan, W., Cho, K., He, H., Sukhbaatar, S., Weston, J. (2024). Iterative Reasoning
Preference Optimization. NeurIPS.

Park, R., Rafailov, R., Ermon, S., Finn, C. (2024). Disentangling Length from Quality in Direct
Preference Optimization. Findings of ACL.

Pascanu, R., Mikolov, T., Bengio, Y. (2013). On the Difficulty of Training Recurrent Neural
Networks. ICML.

Paster, K., Dos Santos, M., Azerbayev, Z., Ba, J. (2023). OpenWebMath: An Open Dataset of
High-Quality Mathematical Web Text. arXiv:2310.06786.

Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library.
NeurIPS.

Patarasuk, P., Yuan, X. (2009). Bandwidth Optimal All-reduce Algorithms for Clusters of
Workstations. Journal of Parallel and Distributed Computing 69(2).

Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, I., Maleki, S., & Bianchini, R. (2024).
Splitwise: Efficient Generative LLM Inference Using Phase Splitting. ISCA.

Pearce, T., Song, J. (2024). Reconciling Kaplan and Chinchilla Scaling Laws. arXiv:2406.12907.

Penedo, G., Kydlicek, H., Ben Allal, L., et al. (2024). The FineWeb Datasets: Decanting the Web for
the Finest Text Data at Scale. NeurIPS Datasets and Benchmarks.

Penedo, G., Malartic, Q., Hesslow, D., et al. (2023). The RefinedWeb Dataset for Falcon LLM:
Outperforming Curated Corpora with Web Data, and Web Data Only. NeurIPS Datasets and Benchmarks.

Peng, B., Quesnelle, J., Fan, H., Shippole, E. (2023). YaRN: Efficient Context Window Extension of
Large Language Models. arXiv:2309.00071.

Phan, L., Gatti, A., Han, Z., et al. (2025). Humanity's Last Exam. arXiv:2501.14249.

Piché, A., et al. (2025). PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence
Generation. arXiv preprint.

Pope, R. et al. (2023). Efficiently Scaling Transformer Inference. MLSys.

Porian, T., Wortsman, M., Jitsev, J., Schmidt, L., Carmon, Y. (2024). Resolving Discrepancies in
Compute-Optimal Scaling of Language Models. NeurIPS.

Prabhu, R. et al. (2025). vAttention: Dynamic Memory Management for Serving LLMs without
PagedAttention. ASPLOS.

Press, O., Smith, N. A., Lewis, M. (2022). Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation. ICLR.

Qi, P., Wan, X., Huang, G., Lin, M. (2024). Zero Bubble Pipeline Parallelism. ICLR.

Qin, R. et al. (2024). Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.
arXiv:2407.00079.

Qwen Team (2025). Qwen3-VL Technical Report. arXiv:2511.21631.

Rabe, M. N., Staats, C. (2021). Self-attention Does Not Need O(n\^2) Memory. arXiv:2112.05682.

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From
Natural Language Supervision. ICML.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. (2018). Improving Language Understanding by
Generative Pre-Training. OpenAI technical report.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are
Unsupervised Multitask Learners. OpenAI Technical Report.

Rae, J. W., Borgeaud, S., Cai, T., et al. (2021). Scaling Language Models: Methods, Analysis &
Insights from Training Gopher. arXiv:2112.11446.

Rafailov, R., Hejna, J., Park, R., Finn, C. (2024). From r to Q\*: Your Language Model is Secretly a
Q-Function. arXiv:2404.12358.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., Finn, C. (2023). Direct
Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS.

Raffel, C., Shazeer, N., Roberts, A., et al. (2020). Exploring the Limits of Transfer Learning with
a Unified Text-to-Text Transformer. JMLR.

Rajbhandari, S., Rasley, J., Ruwase, O., He, Y. (2020). ZeRO: Memory Optimizations Toward Training
Trillion Parameter Models. SC20.

Rajbhandari, S., Ruwase, O., Rasley, J., Smith, S., He, Y. (2021). ZeRO-Infinity: Breaking the GPU
Memory Wall for Extreme Scale Deep Learning. SC.

Rasley, J., Rajbhandari, S., Ruwase, O., He, Y. (2020). DeepSpeed: System Optimizations Enable
Training Deep Learning Models with Over 100 Billion Parameters. KDD.

Razin, N., Malladi, S., Bhaskar, A., Chen, D., Arora, S., Hu, W. (2024). Unintentional Unalignment:
Likelihood Displacement in Direct Preference Optimization. arXiv:2410.08847 (ICLR 2025).

Rein, D., Hou, B. L., Stickland, A. C., et al. (2023). GPQA: A Graduate-Level Google-Proof Q&A
Benchmark. arXiv:2311.12022.

Ren, J., Rajbhandari, S., Aminabadi, R. Y., et al. (2021). ZeRO-Offload: Democratizing Billion-Scale
Model Training. USENIX ATC.

Rogozhnikov, A. (2022). Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation.
ICLR.

Rouhani, B. D., et al. (2023). Microscaling Data Formats for Deep Learning. arXiv:2310.10537.

Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). Learning Representations by
Back-Propagating Errors. Nature.

Sanders, P., Speck, J., Träff, J. L. (2009). Two-Tree Algorithms for Full Bandwidth Broadcast,
Reduction and Scan. Parallel Computing.

Sap, M., Card, D., Gabriel, S., Choi, Y., Smith, N. A. (2019). The Risk of Racial Bias in Hate
Speech Detection. ACL.

Sardana, N., Frankle, J. (2023). Beyond Chinchilla-Optimal: Accounting for Inference in Language
Model Scaling Laws. arXiv:2401.00448.

Schaeffer, R., Miranda, B., Koyejo, S. (2023). Are Emergent Abilities of Large Language Models a
Mirage? NeurIPS.

Schlag, I., Irie, K., Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight
Programmers. ICML.

Schulman, J. (2020). Approximating KL Divergence. Blog post (joschu.net).

Schulman, J., Levine, S., Moritz, P., Jordan, M., Abbeel, P. (2015). Trust Region Policy
Optimization. ICML.

Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2016). High-Dimensional Continuous
Control Using Generalized Advantage Estimation. ICLR.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization
Algorithms. arXiv:1707.06347.

Sennrich, R., Haddow, B., Birch, A. (2016). Neural Machine Translation of Rare Words with Subword
Units. ACL.

Sergeev, A., Del Balso, M. (2018). Horovod: Fast and Easy Distributed Deep Learning in TensorFlow.
arXiv:1802.05799.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., & Dao, T. (2024). FlashAttention-3:
Fast and Accurate Attention with Asynchrony and Low-precision. NeurIPS.

Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models. arXiv:2402.03300.

Sharma, M., Tong, M., Korbak, T., et al. (2023). Towards Understanding Sycophancy in Language
Models. arXiv:2310.13548.

Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head is All You Need. arXiv:1911.02150.

Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.

Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated
Mixture-of-Experts Layer. ICLR.

Shazeer, N., Stern, M. (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. ICML.

Shen, G., Wang, Z., Delalleau, O., et al. (2024). NeMo-Aligner: Scalable Toolkit for Efficient Model
Alignment. arXiv:2405.01481.

Sheng, G., Zhang, C., Ye, Z., et al. (2025). HybridFlow: A Flexible and Efficient RLHF Framework
(veRL). EuroSys.

Sheng, Y. et al. (2024). Fairness in Serving Large Language Models. OSDI.

Sheng, Y., Cao, S., Li, D., et al. (2024). S-LoRA: Serving Thousands of Concurrent LoRA Adapters.
MLSys.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-LM:
Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv:1909.08053.

Singhal, P., Goyal, T., Xu, J., Durrett, G. (2023). A Long Way to Go: Investigating Length
Correlations in RLHF. arXiv:2310.03716.

Snell, C., Lee, J., Xu, K., and Kumar, A. (2024). Scaling LLM Test-Time Compute Optimally can be
More Effective than Scaling Model Parameters. arXiv:2408.03314.

So, D. R., Manke, W., Liu, H., et al. (2021). Primer: Searching for Efficient Transformers for
Language Modeling. NeurIPS.

Soldaini, L., Kinney, R., Bhagia, A., et al. (2024). Dolma: an Open Corpus of Three Trillion Tokens
for Language Model Pretraining Research. ACL.

Spector, B. F., Arora, S., Singhal, A., Fu, D. Y., & Re, C. (2024). ThunderKittens: Simple, Fast,
and Adorable AI Kernels. arXiv:2410.20399.

Stiennon, N., Ouyang, L., Wu, J., et al. (2020). Learning to Summarize from Human Feedback. NeurIPS.

Su, D., Kong, K., Lin, Y., et al. (2024). Nemotron-CC: Transforming Common Crawl into a Refined
Long-Horizon Pretraining Dataset. arXiv:2412.02595.

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021). RoFormer: Enhanced Transformer with
Rotary Position Embedding. arXiv:2104.09864.

Sun, Y., Dong, L., Huang, S., et al. (2023). Retentive Network: A Successor to Transformer for Large
Language Models. arXiv:2307.08621.

Sutskever, I., Martens, J., Dahl, G., Hinton, G. (2013). On the Importance of Initialization and
Momentum in Deep Learning. ICML.

Sutskever, I., Vinyals, O., Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks.
NeurIPS.

Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

Sutton, R. S., McAllester, D., Singh, S., Mansour, Y. (2000). Policy Gradient Methods for
Reinforcement Learning with Function Approximation. NeurIPS.

Tajwar, F., et al. (2026). Maximum Likelihood Reinforcement Learning. arXiv:2602.02710.

Tamkin, A., McCain, M., Handa, K., et al. (2024). Clio: Privacy-Preserving Insights into Real-World
AI Use. arXiv:2412.13678.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B.
(2023). Alpaca: A Strong, Replicable Instruction-Following Model. GitHub repository.

Team OLMo (2025). OLMo 3. Technical report, Allen Institute for AI.

Team OLMo, Walsh, P., Soldaini, L., et al. (2024). 2 OLMo 2 Furious. arXiv:2501.00656.

Tenstorrent (2024--2025). TT-Metalium and TT-NN documentation and source code.
<https://github.com/tenstorrent/tt-metal>

Thakur, R., Rabenseifner, R., Gropp, W. (2005). Optimization of Collective Communication Operations
in MPICH. International Journal of High Performance Computing Applications 19(1).

Tieleman, T., Hinton, G. (2012). RMSProp: Divide the gradient by a running average of its recent
magnitude. Neural Networks for Machine Learning (online notes).

Tillet, P., Kung, H. T., Cox, D. (2019). Triton: An Intermediate Language and Compiler for Tiled
Neural Network Computations. MAPL Workshop.

Touvron, H., Martin, L., Stone, K., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat
Models. arXiv:2307.09288.

van de Geijn, R. A., Watts, J. (1997). SUMMA: Scalable Universal Matrix Multiplication Algorithm.
Concurrency: Practice and Experience.

van den Oord, A., Vinyals, O., Kavukcuoglu, K. (2017). Neural Discrete Representation Learning.
NeurIPS.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L.,
Polosukhin, I. (2017). Attention Is All You Need. NeurIPS.

Vidgen, B., Scherrer, N., Kirk, H. R., et al. (2024). SimpleSafetyTests: A Test Suite for
Identifying Critical Safety Risks in Large Language Models. arXiv:2311.08370.

von Werra, L., Belkada, Y., Tunstall, L., et al. (2020-). TRL: Transformer Reinforcement Learning.
GitHub: huggingface/trl.

Vyas, N., Morwani, D., Zhao, R., et al. (2024). SOAP: Improving and Stabilizing Shampoo using Adam.
arXiv:2409.11321.

Wang, L., Yang, N., Huang, X., et al. (2024). Improving Text Embeddings with Large Language Models.
ACL.

Wang, P., Bai, S., Tan, S., et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception of
the World at Any Resolution. arXiv:2409.12191.

Wang, S., Wei, J., Sabne, A., et al. (2023). Overlap Communication with Dependent Computation via
Decomposition in Large Deep Learning Models. ASPLOS.

Wang, X., Wei, J., Schuurmans, D., et al. (2023). Self-Consistency Improves Chain of Thought
Reasoning in Language Models. ICLR.

Wang, Y., Ma, X., Zhang, G., et al. (2024). MMLU-Pro: A More Robust and Challenging Multi-Task
Language Understanding Benchmark. NeurIPS Datasets and Benchmarks.

Wei, J., et al. (2022). Emergent Abilities of Large Language Models. TMLR.

Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in
Large Language Models. NeurIPS.

Wenzek, G., Lachaux, M.-A., Conneau, A., et al. (2020). CCNet: Extracting High Quality Monolingual
Datasets from Web Crawl Data. LREC.

Willard, B. T., & Louf, R. (2023). Efficient Guided Generation for Large Language Models.
arXiv:2307.09702.

Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist
Reinforcement Learning. Machine Learning 8(3-4).

Williams, S., Waterman, A., Patterson, D. (2009). Roofline: An Insightful Visual Performance Model
for Multicore Architectures. Communications of the ACM.

Wu, S., et al. (2023). BloombergGPT: A Large Language Model for Finance. arXiv:2303.17564.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023). SmoothQuant: Accurate and
Efficient Post-Training Quantization for Large Language Models. ICML.

Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). Efficient Streaming Language Models with
Attention Sinks. ICLR.

Xie, S. M., Santurkar, S., Ma, T., Liang, P. (2023). Data Selection for Language Models via
Importance Resampling. NeurIPS.

Xiong, R., Yang, Y., He, D., et al. (2020). On Layer Normalization in the Transformer Architecture.
ICML.

Xu, H., Sharaf, A., Chen, Y., Tan, W., Shen, L., Van Durme, B., Murray, K., Kim, Y. J. (2024).
Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine
Translation. ICML.

Xu, S., Fu, W., Gao, J., Ye, W., Liu, W., Mei, Z., Wang, G., Yu, C., Wu, Y. (2024). Is DPO Superior
to PPO for LLM Alignment? A Comprehensive Study. ICML.

Xu, Y., Lee, H., Chen, D., et al. (2021). GSPMD: General and Scalable Parallelization for ML
Computation Graphs. arXiv:2105.04663.

Yang, A. et al. (2025). Qwen3 Technical Report. arXiv:2505.09388.

Yang, G., Hu, E. J., Babuschkin, I., et al. (2022). Tensor Programs V: Tuning Large Neural Networks
via Zero-Shot Hyperparameter Transfer. arXiv:2203.03466.

Yang, J., Lieret, K., Jimenez, C. E., et al. (2025). SWE-smith: Scaling Data for Software
Engineering Agents. arXiv:2504.21798.

Yang, S., Kautz, J., Hatamizadeh, A. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule.
ICLR.

Yang, S., Wang, B., Shen, Y., Panda, R., Kim, Y. (2024). Gated Linear Attention Transformers with
Hardware-Efficient Training. ICML.

Yang, S., Wang, B., Zhang, Y., Shen, Y., Kim, Y. (2024). Parallelizing Linear Transformers with the
Delta Rule over Sequence Length. NeurIPS.

Yao, F., et al. (2025). Your Efficient RL Framework Secretly Brings You Off-Policy RL Training
(truncated importance sampling, verl). Blog post.

Yao, S., Yu, D., Zhao, J., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large
Language Models. NeurIPS.

Yao, Z., Aminabadi, R. Y., Ruwase, O., et al. (2023). DeepSpeed-Chat: Easy, Fast and Affordable RLHF
Training of ChatGPT-like Models at All Scales. arXiv:2308.01320.

Ye, Z. et al. (2025). FlashInfer: Efficient and Customizable Attention Engine for LLM Inference
Serving. MLSys.

Young, J. W. (1974). A First Order Approximation to the Optimum Checkpoint Interval for Restarts.
Communications of the ACM.

Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., Chun, B.-G. (2022). Orca: A Distributed Serving System
for Transformer-Based Generative Models. OSDI.

Yu, L., Simig, D., Flaherty, C., Aghajanyan, A., Zettlemoyer, L., Lewis, M. (2023). MEGABYTE:
Predicting Million-byte Sequences with Multiscale Transformers. NeurIPS.

Yu, Q., Zhang, Z., Zhu, R., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at
Scale. arXiv:2503.14476.

Yuan, W., Pang, R. Y., Cho, K., Li, X., Sukhbaatar, S., Xu, J., Weston, J. (2024). Self-Rewarding
Language Models. ICML.

Zelikman, E., Wu, Y., Mu, J., Goodman, N. D. (2022). STaR: Bootstrapping Reasoning With Reasoning.
NeurIPS.

Zeng, Y., Yang, Y., Zhou, A., et al. (2024). AIR-Bench 2024: A Safety Benchmark Based on Risk
Categories from Regulations and Policies. arXiv:2407.17436.

Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L. (2023). Sigmoid Loss for Language Image
Pre-Training. ICCV.

Zhang, B., Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS.

Zhao, Y., Gu, A., Varma, R., et al. (2023). PyTorch FSDP: Experiences on Scaling Fully Sharded Data
Parallel. VLDB.

Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., Liu, P. J. (2023). SLiC-HF: Sequence
Likelihood Calibration with Human Feedback. arXiv:2305.10425.

Zheng, C., et al. (2025). Group Sequence Policy Optimization (GSPO). arXiv:2507.18071.

Zheng, C., Liu, S., Li, M., et al. (2025). Group Sequence Policy Optimization. arXiv:2507.18071.

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena. NeurIPS Datasets and Benchmarks.

Zheng, L., Yin, L., Xie, Z., et al. (2024). SGLang: Efficient Execution of Structured Language Model
Programs. NeurIPS.

Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., & Zhang, H. (2024). DistServe:
Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. OSDI.

Zhou, C., Liu, P., Xu, P., et al. (2023). LIMA: Less Is More for Alignment. NeurIPS.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., Irving,
G. (2019). Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593.

Zoph, B., Bello, I., Kumar, S., et al. (2022). ST-MoE: Designing Stable and Transferable Sparse
Expert Models. arXiv:2202.08906.

Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., Fredrikson, M. (2023). Universal and
Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043.
:::
