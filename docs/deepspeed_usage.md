# How to Choose Which ZeRO Stage and Offloads To Use For Best Performance

So now you know there are all these different stages. How to decide which of them to use? This section will attempt to address this question.

## In general the following applies:

### Speed-wise (left is faster than right)

Stage 0 (DDP) > Stage 1 > Stage 2 > Stage 2 + offload > Stage 3 > Stage 3 + offloads

### GPU Memory usage-wise (right is more GPU memory efficient than left)

Stage 0 (DDP) < Stage 1 < Stage 2 < Stage 2 + offload < Stage 3 < Stage 3 + offloads

So when you want to get the fastest execution while fitting into minimal number of GPUs, here is the process you could follow. We start with the fastest approach and if running into GPU OOM we then go to the next slower approach, but which will use less GPU memory. And so on and so forth.

First of all set batch size to 1 (you can always use gradient accumulation for any desired effective batch size).

1. Enable --gradient_checkpointing 1 (HF Trainer) or directly model.gradient_checkpointing_enable() - if OOM then
2. Try ZeRO stage 2 first. if OOM then
3. Try ZeRO stage 2 + offload_optimizer - if OOM then
4. Switch to ZeRO stage 3 - if OOM then
5. Enable offload_param to cpu - if OOM then
6. Enable offload_optimizer to cpu - if OOM then
7. If you still can’t fit a batch size of 1 first check various default values and lower them if you can. For example, if you use generate and you don’t use a wide search beam make it narrower as it’d take a lot of memory.
8. Definitely use mixed half-precision over fp32 - so bf16 on Ampere and higher GPUs and fp16 on older gpu architectures.
9. If you still OOM you could add more hardware or enable ZeRO-Infinity - that is switch offloads offload_param and offload_optimizer to nvme. You need to make sure it’s a very fast nvme. As an anecdote I was able to infer BLOOM-176B on a tiny GPU using ZeRO-Infinity except it was extremely slow. But it worked!


You can, of course, work through these steps in reverse by starting with the most GPU memory efficient config and then going backwards. Or try bi-secting it.

Once you have your batch size 1 not leading to OOM, measure your effective throughput.

Next try to increase the batch size to as large as you can, since the higher the batch size the more efficient the GPUs are as they perform the best when matrices they multiply are huge.

Now the performance optimization game starts. You can turn off some offload features or step down in ZeRO stages and increase/decrease batch size and again measure your effective throughput. Rinse and repeat until satisfied.

Don’t spend forever on it, but if you’re about to start a 3 months training - do spend a few days on it to find the most effective throughput-wise setup. So that your training cost will be the lowest and you will finish training faster. In the current crazy-paced ML world, if it takes you an extra month to train something you are likely to miss a golden opportunity. Of course, this is only me sharing an observation and in no way I’m trying to rush you. Before beginning to train BLOOM-176B I spent 2 days on this process and was able to increase throughput from 90 to 150 TFLOPs! This effort saved us more than one month of training time.

These notes were written primarily for the training mode, but they should mostly apply for inference as well. For example, during inference Gradient Checkpointing is a no-op since it is only useful during training. Additionally, we found out that if you are doing a multi-GPU inference and not using DeepSpeed-Inference, Accelerate should provide a superior performance.
