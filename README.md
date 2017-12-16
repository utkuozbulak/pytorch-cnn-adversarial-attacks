# Convolutional Neural Network Adversarial Attacks

This repo contains following CNN adversarial attacks implemented in Pytorch: 

* Fast Gradient Sign, Untargeted [1]
* Fast Gradient Sign, Targeted [1]
* Gradient Ascent, Adversarial Images 
* Gradient Ascent, Fooling Images (Unrecognizable images predicted as classes with high confidence) [2]

It will also include more adverisarial attack techniques and defenses in the future as well.

The code uses pretrained AlexNet in the model zoo. You can simply change it with your model but don't forget to change target class parameters as well.

All images are pre-processed with mean and std of the ImageNet dataset before being fed to the model. None of the code uses GPU as these operations are quite fast (for a single image). You can make use of gpu with very little effort. The examples below include numbers in the brackets after the description, like *Mastiff (243)*, this number represents the class id in the ImageNet dataset.

I tried to comment on the code as much as possible, if you have any issues understanding it or porting it, don't hesitate to reach out. 

Below, are some sample results for each operation.

## Fast Gradient Sign - Untargeted


<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> Predicted as <strong>Zebra</strong> (340) <br/> Confidence: 0.94 </td>
			<td width="27%" align="center"> Predicted as <strong>Bow tie</strong> (457) <br/> Confidence: 0.95 </td>
			<td width="27%" align="center"> Predicted as <strong>Castle</strong> (483) <br/> Confidence: 0.99 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/pytorch-cnn-adversarial-attacks/master/input_images/eel.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-adversarial-attacks/master/generated/untargeted_adv_noise_from_390_to_397.png"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-adversarial-attacks/master/generated/untargeted_adv_img_from_390_to_397.png"> </td>
		</tr>
	</tbody>
</table>



## Fooling Image Generation
This operation is quite similar to generating class specific images, we start with a random image and continously update the image with targeted backpropagation (for a certain class) and stop when we achieve target confidence for that class. All of the below images are generated from pretrained AlexNet to fool it.


<table border=0 width="50px" >
	<tbody> 
    <tr>		<td width="27%" align="center"> Predicted as <strong>Zebra</strong> (340) <br/> Confidence: 0.94 </td>
			<td width="27%" align="center"> Predicted as <strong>Bow tie</strong> (457) <br/> Confidence: 0.95 </td>
			<td width="27%" align="center"> Predicted as <strong>Castle</strong> (483) <br/> Confidence: 0.99 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_340.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_457.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_483.jpg"> </td>
		</tr>
	</tbody>
</table>


## Disguised Fooling (Adversarial) Image Generation
For this operation we start with an image and perform gradient updates on the image for a specific class but with smaller learning rates so that the original image does not change too much. As it can be seen from samples, on some images it is almost impossible to recognize the difference between two images but on others it can clearly be observed that something is wrong. All of the examples below were created from and tested on AlexNet to fool it.


<table border=0 width="50px" >
	<tbody> 
		<tr>		<td width="27%" align="center"> Predicted as <strong>Eel</strong> (390) <br/> Confidence: 0.96 </td>
			<td width="27%" align="center"> Predicted as <strong>Apple</strong> (948) <br/> Confidence: 0.95 </td>
			<td width="27%" align="center"> Predicted as <strong>Snowbird</strong> (13) <br/> Confidence: 0.99 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/eel.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/apple.JPEG"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/input_images/bird.JPEG"> </td>
		</tr>
		<tr>		<td width="27%" align="center"> Predicted as <strong>Banjo</strong> (420) <br/> Confidence: 0.99 </td>
			<td width="27%" align="center"> Predicted as <strong>Abacus</strong> (457) <br/> Confidence: 0.99 </td>
			<td width="27%" align="center"> Predicted as <strong>Dumbell</strong> (543) <br/> Confidence: 1 </td>
		</tr>
		<tr>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_420.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_398.jpg"> </td>
			<td width="27%" align="center"> <img src="https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-visualizations/master/results/fooling_sample_class_543.jpg"> </td>
		</tr>
	</tbody>
</table>




## Requirements:
```
torch >= 0.2.0.post4
torchvision >= 0.1.9
numpy >= 1.13.0
opencv >= 3.1.0
```


## References:

[1] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. *Striving for Simplicity: The All Convolutional Net*, https://arxiv.org/abs/1412.6806

[2] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, A. Torralba. *Learning Deep Features for Discriminative Localization*, https://arxiv.org/abs/1512.04150

[3] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, https://arxiv.org/abs/1610.02391

[4] K. Simonyan, A. Vedaldi, A. Zisserman. *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*, https://arxiv.org/abs/1312.6034

[5] A. Mahendran, A. Vedaldi. *Understanding Deep Image Representations by Inverting Them*, https://arxiv.org/abs/1412.0035

[6] H. Noh, S. Hong, B. Han,  *Learning Deconvolution Network for Semantic Segmentation* https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf

[7] A. Nguyen, J. Yosinski, J. Clune.  *Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable  Images* https://arxiv.org/abs/1412.1897

[8] D. Smilkov, N. Thorat, N. Kim, F. Viégas, M. Wattenberg. *SmoothGrad: removing noise by adding noise* https://arxiv.org/abs/1706.03825

[9] D. Erhan, Y. Bengio, A. Courville, P. *Vincent. Visualizing Higher-Layer Features of a Deep Network* https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network

[10] A. Mordvintsev, C. Olah, M. Tyka. *Inceptionism: Going Deeper into Neural Networks* https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
