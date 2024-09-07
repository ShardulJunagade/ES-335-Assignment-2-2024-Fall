# Assignment 2 

**Total marks: 10 (This assignment total to 20, we will overall scale by a factor of 0.5)**

## Task 1 : Ascending the Gradient Descent [6 marks]

Use the below dataset for Task 1: 
```py
np.random.seed(45)
num_samples = 40
    
# Generate data
x1 = np.random.uniform(-1, 1, num_samples)
f_x = 3*x1 + 4
eps = np.random.randn(num_samples)
y = f_x + eps
```

1. Use ```torch.autograd``` to compute the true gradient on above dataset using linear regression ($\theta_1x + \theta_0$). **[1 mark]**

2. Compute the stochastic gradient over the entire dataset. Show that the stochastic gradient is an unbiased estimator of the true gradient.  **[1 mark]**

3. Implement full-batch, mini-batch and stochastic gradient descent. Calculate the average number of iterations required for each method to get sufficiently close to the optimal solution, where "sufficiently close" means within a distance of $\epsilon$ (or $\epsilon$-neighborhood)  from the minimum value of the loss function. Visualize the convergence process for 15 epochs. Choose $\epsilon = 0.001$ for convergence criteria. Which optimization process takes a larger number of epochs to converge, and why? Show the contour plots for different epochs (or show an animation/GIF) for visualisation of optimisation process. Also, make a plot for Loss v/s epochs for all the methods. **[2 marks]**

4. Explore the article [here](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/#:~:text=Momentum%20is%20an%20extension%20to,spots%20of%20the%20search%20space.) on gradient descent with momentum. Implement gradient descent with momentum for the dataset. Visualize the convergence process for 15 steps. Compare the average number of steps taken with gradient descent (for variants full batch and stochastic) with momentum to that of vanilla gradient descent to converge to an $\epsilon$-neighborhood for both dataset. Choose $\epsilon = 0.001$. Write down your observations. Show the contour plots for different epochs for momentum implementation. Specifically, show all the vectors: gradient, current value of theta, momentum, etc. **[2 marks]**
     

## Task 2 : Super-Resolution using Random Fourier Features (RFF)  [4 Marks]

Begin by exploring the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/siren.ipynb) that introduces the application of Random Fourier Features (RFF) for image reconstruction. Demonstrate the following applications using the cropped image from the notebook:
    
1.  Superresolution: Perform superresolution on the image shown in notebook to enhance its resolution by factor 2. Show a qualitative comparison of original and reconstructed image. (i.e display original image and the image you created side by side) **[2 Marks]**
2. The above only helps us with a qualitative comparison. Let us now do a quantitative comparison. First, skim read this article: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution. This article is given as reference to understand super-resolution. You are not expected to implement any deep learning model. You are supposed to use RFF with linear regression for image reconstruction. **[2 Marks]**

    Follow the steps below:
    - Start with a 400x400 image (ground truth high resolution).
    - Resize it to a 200x200 image (input image)
    - Use RFF + Linear regression to increase the resolution to 400x400 (predicted high resolution image)
    - Compute the following metrics:
        - RMSE on predicted v/s ground truth high resolution image
        - Peak SNR on predicted v/s ground truth high resolution image


## Task 3 : Reconstructing using Random Fourier Features (RFF) [3 marks]

1. **Image Reconstruction** - Take an image of your liking. Apply RFF to complete the image with 10%, 20%, and so on up to 90% of its data missing randomly. Randomly remove portions of the data, train the model on the remaining data (Remove the missing part from the training data), and predict for the entire image. Display both the original and reconstructed images for each level of missing data. Additionally, calculate and report the Root Mean Squared Error (RMSE) and Peak Signal-to-Noise Ratio (PSNR) for each reconstruction. **[2 Marks]**

2. **Audio Reconstruction** - Select an audio sample of your choice, ideally around 5 seconds in length. Reconstruct the audio using Random Fourier Features (RFF) and Linear Regression. Display the reconstructed audio sample and calculate the Root Mean Squared Error (RMSE) and Signal-to-Noise Ratio (SNR) for the reconstruction. **[1 Mark]**

Note : Please notice that generally PSNR is used for images while SNR is used for audio signals.

## Task 4 : Image Reconstruction using Matrix Factorisation [4 Marks]

Use the [instructor's notebook](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/movie-recommendation-knn-mf.ipynb) on matrix factorisation, and solve the following questions. Here, ground truth pixel values are missing for particular regions within the image- you don't have access to them.

1. Use an image of your liking and reconstruct the image in the following two cases, where the missing region is-
    - Structured : A rectangular block of 30X30 is assumed missing from the image. 
    - Unstructured : A random subset of 900 (30X30) pixels is missing from the image. 

    Choose rank `r` yourself. Perform Gradient Descent till convergence, plot the selected regions, original and reconstructed images, compute the metrics mentioned in Q4 and write your observations. 
    Obtain the reconstruction using RFF + Linear regression and compare the two. **[2 Marks]**

2. Vary region size (NxN) for ```N = [20, 40, 60, 80]``` and perform Gradient Descent till convergence. Again, consider the two cases for your region as mentioned in Part (a). Demonstrate the variation in reconstruction quality by making appropriate plots and metrics. **[2 Marks]**

## Task 5 : Image Compression using Matrix Factorisation [3 Marks]
    
Here, ground truth pixel values are not missing and you have access to them. You want to explore the use of matrix factorisation in order to store them more efficiently. Consider an image patch of size (NxN) where N=50. We are trying to compress this "patch" (matrix) into two matrices, by using low-rank matrix factorization. Consider the following three cases

1. A patch with mainly a single color.
2. A patch with 2-3 different colors.
3. A patch with at least 5 different colors.

Vary the low-rank value as ```r = [5, 10, 25, 50]```  for each of the cases. Use Gradient Descent and plot the reconstructed patches over the original image (retaining all pixel values outside the patch, and using your learnt compressed matrix in place of the patch) to demonstrate difference in reconstruction quality. Write your observations. **[3 Marks]**

Here is a reference set of patches chosen for each of the 3 cases from left to right. You can chose an image of your liking and create similar patches for the cases. You can choose the image shown below as well.

<div style="display: flex;">
  <img src="sample_images/1colour.jpg" alt="Image 1" width="250"/>
  <img src="sample_images/2-3_colours.jpg" alt="Image 2" width="270"/>
  <img src="sample_images/multiple_colours.jpg" alt="Image 3" width="265"/>
</div>

<br>

---


### General Instructions

- Show your results in a Jupyter Notebook or an MD file. If you opt for using an MD file, you should also include the code (.py or .ipynb as a separate file.)
- This assignment is of 20 marks and will be scaled down to 10 marks.
- Please read the questions carefully and make sure you are addressing all the questions or else you may loose some marks.
- Ensure that your code is readable and commited before the deadline.
- Use #assignments channel on slack for doubts.

