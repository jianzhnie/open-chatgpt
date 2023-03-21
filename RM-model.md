# Sections to train Reward Model (RM)

Trainer code based on huggingface. Compatible with deepspeed or accelerate

## Dataset

For now we only supports webgpt and summary dataset from OpenAI.

## Model

You can add new huggingface model as you want.


## Example1: 
Run training procedure

```bash
python trainer.py 
```

## Example2: 
Additional axis labeling, this outputs a 4 summary quality evaluation metrics, (score are normalized to 0-1 )

```bash
python summary_quality_trainer.py
```

The four summary are :

- overall
- accuracy
- coverage
- coherence
