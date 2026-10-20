# Breakthrough Papers in Foundation Models After ChatGPT: Consolidated Recommendations

Merged from four independent lists:
- `muse_rec.md` (Muse)
- `codex_rec.md` (Codex)
- `claude_rec.md` (Claude)
- ChatGPT's answer (pasted into the conversation)

ChatGPT launched November 30, 2022. Links come from the Codex and ChatGPT lists where they gave them.

How big each list is: **Muse** about 25 papers, **Codex** 19, **ChatGPT** 14 distinct (15 entries, with Gemini 1.5 listed twice), **Claude** about 110. Claude's list is far larger, so it naturally has the most picks the others don't.

---

## 1. All Four Agents Agree (9)

| Paper | Org | Date | Consensus reason |
|-------|-----|------|------------------|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | Meta | Feb 2023 | Started the open-weights ecosystem; strong small models trained on more tokens. |
| [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | OpenAI | Mar 2023 | Set the frontier capability bar; multimodal (image + text) input. |
| [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | Stanford | May 2023 | Replaced the RLHF reward-model + PPO pipeline with one simple loss. |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | CMU / Princeton | Dec 2023 | First serious attention-free alternative for long sequences. |
| [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | Mistral AI | Jan 2024 | Brought efficient sparse MoE to strong open models. |
| [Gemini 1.5](https://arxiv.org/abs/2403.05530) | Google DeepMind | Feb 2024 | Million-token multimodal context. |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | Meta | Jul 2024 | 405B open-weight model with an end-to-end training report. |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | DeepSeek | Dec 2024 | Frontier-level MoE (671B total / 37B active) at low training cost (MLA, MTP, FP8). |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | DeepSeek | Jan 2025 | RL on verifiable rewards produces reasoning (R1-Zero); open o1-level model; distillation into small models. |

**Top-pick overlap:** all four agents name **GPT-4, LLaMA, and DeepSeek-R1/V3** among their headline picks. **DPO** is in the top tier for Muse, Claude, and ChatGPT.

---

## 2. Three of Four Agree (8)

| Paper | Date | Recommended by | Missing from |
|-------|------|----------------|--------------|
| [QLoRA](https://arxiv.org/abs/2305.14314) | May 2023 | Muse, Codex, Claude | ChatGPT |
| [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) | Apr 2023 | Muse, Codex, Claude | ChatGPT |
| [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) | Apr 2023 | Muse, Codex, Claude | ChatGPT |
| [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) | Sep 2023 | Muse, Codex, Claude | ChatGPT |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | May 2025 | Muse (together with Qwen2.5), Codex, Claude | ChatGPT |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Jul 2023 | Muse, Claude, ChatGPT | Codex |
| Mistral 7B | Oct 2023 | Muse, Claude, ChatGPT | Codex |
| OpenAI o1 ([System Card](https://arxiv.org/abs/2412.16720) / "Learning to Reason with LLMs") | Sep–Dec 2024 | Codex, Claude, ChatGPT | Muse brings up o1 only as the comparison point for R1, not as its own entry |

---

## 3. Two of Four Agree (9)

| Paper | Date | Recommended by |
|-------|------|----------------|
| Scalable Diffusion Models with Transformers (DiT) | Dec 2022 | Muse, Claude |
| Let's Verify Step by Step (process reward models) | May 2023 | Muse, Claude |
| DINOv2 | Apr 2023 | Muse, Claude |
| AWQ: Activation-aware Weight Quantization | Jun 2023 | Muse, Claude |
| Llama 2 | Jul 2023 | Muse, Claude |
| [Scaling Rectified Flow Transformers (Stable Diffusion 3)](https://arxiv.org/abs/2403.03206) | Mar 2024 | Codex, Claude |
| [AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) | May 2024 | Codex, Claude |
| DeepSeekMath (introduced GRPO) | Feb 2024 | ChatGPT, Claude |
| [Tülu 3: open post-training with RLVR](https://arxiv.org/abs/2411.15124) | Nov 2024 | ChatGPT, Claude |

---

## 4. Where Each Agent Differs

### Muse
- **Picks no one else made:** none. Every Muse paper appears in at least one other list. Its only addition is **Qwen2.5**, which it lists together with Qwen3.
- **Differences in emphasis and dates:**
  - Muse calls DeepSeek-R1 the paper that "established test-time compute as new scaling axis."
  - Muse dates LLaVA to Oct 2023 (the NeurIPS version) and vLLM to Jun 2023 (the blog post). The arXiv papers are Apr 2023 and Sep 2023.
  - Muse dates Mixtral to Dec 2023 (the model release). The paper is from Jan 2024.

### Codex
- **Picks no one else made:**
  - [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) (Dec 2023): one natively multimodal family spanning on-device to frontier sizes.
  - [Gemini 2.5](https://arxiv.org/abs/2507.06261) (2025): reasoning, long context, multimodality, and agentic capabilities in one frontier family.
- **Differences in framing:**
  - Codex puts DPO, QLoRA, and vLLM in a separate group of "enabling methods/systems."
  - It weights model and technical reports most heavily.
  - It's the only list that explicitly adds caveats about licensing and reproducibility.

### ChatGPT
- **Picks no one else made:**
  - [GPT-4o System Card](https://arxiv.org/abs/2410.21276) (Oct 2024): a single omnimodal model (text, image, audio) enabling low-latency, natural voice interaction, instead of a pipeline of separate models.
- **Differences in framing:**
  - ChatGPT ranks papers by "how much they changed the direction of research," not by benchmark gains.
  - It explicitly separates new algorithms from technical reports that show a recipe working at scale.
  - It leans heavily toward reasoning: it recommends DeepSeekMath → o1 → R1 as the chain to understand.
  - It skips vision-only models (SAM, DiT), serving (vLLM), and QLoRA.
  - It lists Gemini 1.5 twice (#7 and #15).

### Claude
Claude has the most picks no one else made, because it covers more areas. The most consequential ones:

- **Alignment and reasoning:**
  - Constitutional AI
  - Self-Instruct
  - LIMA
  - Kimi k1.5
  - s1: Simple Test-Time Scaling
  - Scaling Test-Time Compute Optimally (Snell et al.)
  - Tree of Thoughts
  - Quiet-STaR
- **Open models:**
  - phi-1 "Textbooks Are All You Need"
  - OLMo
  - DeepSeek-V2 (MLA)
  - Kimi K2 (MuonClip)
- **Architecture, efficiency, and inference:**
  - Speculative decoding
  - GQA
  - YaRN
  - StreamingLLM / attention sinks
  - Ring Attention
  - BitNet b1.58
  - Mamba-2
  - FlashAttention-3
  - Muon at scale
  - Native Sparse Attention
  - Gated Attention (NeurIPS 2025 Best Paper)
- **Agents and tool use:**
  - Toolformer
  - Reflexion
  - Generative Agents
  - Voyager
  - SWE-agent
- **Multimodal and generative:**
  - BLIP-2
  - Consistency Models
  - Sora tech report
  - Genie
  - VAR (NeurIPS 2024 Best Paper)
  - Chameleon
  - SAM 2
  - Movie Gen
- **Robotics:**
  - Diffusion Policy
  - RT-2
  - Open X-Embodiment
  - OpenVLA
  - π0
- **Scaling, data, and evaluation:**
  - Emergent Abilities Mirage
  - Scaling Data-Constrained LMs
  - Lost in the Middle
  - SWE-bench
  - GPQA
  - Chatbot Arena
  - FineWeb / DCLM
  - Humanity's Last Exam
- **Interpretability and safety:**
  - GCG
  - Representation Engineering
  - Towards / Scaling Monosemanticity
  - Sleeper Agents
  - Alignment Faking
  - Circuit Tracing
- **Science:**
  - GraphCast
  - GNoME
  - FunSearch
  - AlphaGeometry
  - Evo
  - ESM3

**Missing from Claude:**
- Gemini 1.0 and Gemini 2.5 (Codex)
- GPT-4o System Card (ChatGPT)
- Qwen2.5 (Muse)

---

## 5. Suggested Reading

### 5a. ChatGPT's reading order (kept as given; written for someone working through Stanford CS336)

1. **LLaMA**: the ingredients of a modern pretrained LM (data, architecture, scaling, training).
2. **DPO**: how preference data changes model behavior after pretraining.
3. **FlashAttention-2**: how GPU memory movement and parallelism affect real training performance.
4. **DeepSeekMath**: GRPO and the link between RL and mathematical reasoning.
5. **DeepSeek-R1**: reasoning-oriented RL, verifiable rewards, and the full training recipe.
6. **DeepSeek-V3**: MoE architecture, distributed training, and efficient frontier-scale engineering.

> **ChatGPT's strongest recommendation:** for a long-term foundation-model career, prioritize **DeepSeekMath, DeepSeek-R1, DPO, and FlashAttention-2**. Together they cover post-training algorithms, reasoning, and ML systems. Then read **Llama 3** and **DeepSeek-V3** to see how those techniques fit into large-scale model development.

### 5b. Combined reading path across all four lists

This builds on ChatGPT's order and adds the consensus papers.

1. **Foundations (pretraining and open models):** LLaMA → GPT-4 → Llama 3
2. **Post-training:** DPO → Tülu 3 (and optionally QLoRA for hands-on fine-tuning)
3. **Systems:** FlashAttention-2 → vLLM
4. **Reasoning:** Let's Verify Step by Step → DeepSeekMath (GRPO) → o1 System Card → DeepSeek-R1
5. **Efficient scaling and architecture:** Mixtral → DeepSeek-V3 → Mamba
6. **Multimodality and long context:** Gemini 1.5 → LLaVA → GPT-4o System Card
7. **Optional specialty areas** (from Sections 3–4): DiT/SD3 (generative vision), SAM/DINOv2 (vision backbones), AlphaFold 3 (science), and Claude's wider topic list.
