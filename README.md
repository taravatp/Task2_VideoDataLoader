# Zebracat_task2

**question 1: Explain why did you design the video dataloader in this way?**

This dataloader enables us to extract a specified number of random frames from each video and their corresponding captions at each iteration.
During the implementation of this task, my primary concerns were how to preprocess the dataset appropriately. The is how I did that:
* Video frames are all of a fixed size, so there was no resizing needed. However, I normalized the pixel intensities to be between 0 and 1 to facilitate optimization. Furthermore, the shapes of video tensors are adjusted in a way to be fed into Conv3D layers.
* Preprocessing the captions was more challenging than frames. First of all, neural networks do not understand strings! I numericalized them to build a suitable representation. Also, captions have a variable length. This prohibits us from using large batches because PyTorch can not stack tensors with different sizes on each other! We could either overlook this problem and use batch sizes of 1 or devise a solution! I preferred implementing a padding strategy to work with larger batches, speeding up training and taking advantage of GPUs!


**question 2: What are the weaknesses of your video loader?**
* It would be better to create multiple sequences of each video that overlap. For example, the first sequence could be frame number one to frame number 5; the second could be frame number 2 to number 6, and so on. In this way, we could take the most advantage out of our available dataset. But in the implemented approach, we are just selecting N random frames from each video!
* This approach does not work with videos with multiple separate sentences as captions.
* We are padding all the captions in a batch to have similar shapes. However, Imagine that a sentence contains 20 words, and another sentence contains merely four words. In this scenario, the padding would lead to sparse representation for the shorter sequence and uselessly increase computation cost.

