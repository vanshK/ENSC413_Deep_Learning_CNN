# ENSC413_Deep_Learning_CNN
Undergraduate Project on Deep Learning
Image Classification using CNN

Data:
Acknowledgements
https://github.com/Shenggan/BCCD_Dataset MIT License

Link to my Kaggle Kernel:
https://www.kaggle.com/vkochar/identify-blood-cell-subtypes-from-images-4fa0e1/notebook


Abstract

In this report, the techniques for classifying White Blood Cells based on Convolutional Neural Networks have been discussed. White Blood Cells play an important role in the proper functioning of the human immune system and the increment of a white blood cell type might indicate a server disease or illness. Therefore, it is important to classify white blood cells for hematologist to pin point the accurate reason for a disease. Deep Learning can be used to automate the process which a hematologist undertakes thus improving the efficiency and speed of the process. One possible solution is CNN with ReLu and Softmax activation function. CNN have been used for image classification in the past and provide an optimum solution to classifying White Blood Cells with an acceptable accuracy. Increasing the depth of a CNN network results in better accuracy with less fluctuation in the accuracy for validation and test set. 


Introduction
The blood stream consists of different types of blood cells including platelets, red blood cells, plasma and white blood cells. Each cell has a specific function and form a major component of the blood stream. The proportion of white blood cells compared to the other type of cells is marginal amounting to approximately 1%, however this amount should not be considered as a factor in indicating their impact on the human immune system, as these cells help in defending against viruses, bacteria ensuring the health of a person. An abnormal increase or decrease in the level of white blood cells may indicate a disease or illness, such as HIV/AIDS, Myelodysplastic syndrome, and Cancer of the blood.
Background
Blood smear images from a microscope provide important information for diagnosing and predicting diseases in hematological analysis. Blood samples are prepared and sent to a blood cell counter for calculating each type of cell. If hematologists find an unusual number of cells in any type, they will investigate further by looking into the microscopic blood smear, recount the number of cells and check their morphology in more detail. Any blood cells with irregular shapes or characteristics may trigger a presence of severe diseases.
In the practice of hematology, a sample of blood smeared image is examined to gather information to properly analyze, distinguish and recognize a particular disease. The process of analysis starts with determining the count of each blood cell in a blood sample by a hematologist, this is done by a blood cell counter which takes a prepared blood sample as a input. If the result of the count indicates an unusual quantity of a particular blood cell, thereby indicating a possibility of a disease, the hematologist further examines the blood smear under a microscope and check for any non-uniformity. [2]. Therefore, the practice of segregation of cell type based on blood smear is important for the hematologist for proper analysis of the disease.

Description of White Blood Cells
The White Blood Cells itself can be divided into two categories of cells Granulocytes and Agranulocytes. The Granulocytes comprise of eosinophil, neutrophil and basophil. While Agranulocytes comprise of Lymphocyte and monocyte. Eosinophils can be described as having a nucleus which is lobed, and having a globular, substantial granule having an orange color. The WBC with the highest proportion in the blood stream is Neutrophil, there granules are relatively compact in size and red in color with a surrounding cytoplasm which is blue in color therefore giving the nuclei a lavender color ,it has more than one lobed nuclei in it .Basophils can be described as cell structure containing non uniform dispersal of granules, their main function is providing appropriate reaction to the onset of antigens and allergies. Lymphocyte and Monocytes have evenly distributed texture nuclei. Monocytes are characterized by having a kidney shaped nucleus and lymphocytes have a circular nucleus. Therefore, the WBC’s can be differentiated on the basis of cytoplasmic features and nucleus shape and orientation. [2]
Motivation
Since White Blood cells play a vital role in the proper functioning of the immune system with each cell type responsible for a function the process of differentiating the cells given a blood sample based on deep learning is a possible solution to increase the efficiency and accelerate the process of classifying cell types. This can be accomplished by image classification techniques in Deep Learning such as CNN (Convolution Neural Networks).  The objective of the work is to build upon previously tested CNN architectures for image classification of Blood Cells and improve the accuracy achieved in classifying the images of Neutrophils, Eosinophil, Lymphocytes and Monocytes.

Related Work
There have been few studies related to classifying White Blood Cells using Deep Learning. The focus has been more on RBC classification primarily due to its relationship with malaria infection.  
In the research conducted by Macawile et al. white blood cell count and classification of WBC type was accomplished by utilizing Convolutional Neural Network models as the method and microscopic images of blood samples as the data. The models of CNN under consideration were ResNet-101, Alexnet and GoogleNet. The results indicated that AlexNet was the optimum choice as compared to the other two methods for this task since this model resulted in final accuracy of 96.63%, sensitivity of 89.18% and specificity of 97.85% achieved on the sample of 21 images of microscopic blood images. [3] This method achieved high accuracy, but data sample utilized for this method is relatively low which could potentially decrease the test accuracy if a large test set is used. 
	Research conducted by Anjali Gautam et al. implemented the classification of WBC’s on basis of Morphological features using Thresholding and Feature Extraction, the features included Area, Perimeter Eccentricity and Circularity of each WBC type (Neutrophil, Eosinophil, Basophil, Monocyte and Lymphocyte) there result indicated 73% accuracy based on 63 images of blood samples [4]. This method having utilized mathematical operations over CNN resulted in lower accuracy.
	RBC blood cell classification has been undertaken utilizing CNN which potentially can be applied to WBC classification, Research conducted by Mengjia Xu et al. utilized CNN architecture comprising of convolution layers, pooling layers, dropout layers and fully connected layers, utilizing RelU non-linear function as the activation function for the hidden layers and SoftMax for the output layer, along with cross entropy loss function.  This research utilized grayscale RBC images of count 434. This method resulted in a test accuracy of 89.28% [5]. 

Details of the project
Dataset 
The Dataset for the project was obtained from Kaggle Blood Cell images [6]. The Dataset was divided into two files, in total compromising of 125000 augmented images of blood cells in JPEG format. This was accompanied by the labels to differentiate the various cells in CSV format. The dataset contained images of Neutrophils, Monocyte, Lymphocyte and Eosinophil type of WBCs. Each type of Blood Cell compromised of 3000 images associated with it. The Dataset also contained 410 pre-augmented images of the cell images and supplementary labels, the cells in these images were segregated with the individual boxes thereby helping in differentiating the cells in the images.


