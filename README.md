## Basic Seq2Seq Language Translation with Luong (Dot) Attention

This project implements a **basic Seq2Seq (Sequence-to-Sequence) neural network** with **Luong (Dot) Attention** for English → French language translation.

---

## Model Overview

- Encoder–Decoder architecture
- Luong (Dot) Attention mechanism
- Designed for educational and experimental purposes
- Implemented without batching or heavy optimizations

> ⚠️ Note: Since batching and advanced optimizations are not used, training can be slow.  
> However, the model works well for small datasets.

---

## Dataset Preparation

To verify the model:

1. Create a text file containing paired **English–French sentence translations**.
2. Use a small dataset initially (recommended: **10–15 sentence pairs**).
3. Run the training script:

```bash
python seq_to_seq.py
