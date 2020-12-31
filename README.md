# Imbalanced Learning in Binary Classification
Imbalanced Learning Project

applied to the case of frauduelnt credit card transactions. dataset available at https://www.kaggle.com/mlg-ulb/creditcardfraud

focal loss function modified from https://github.com/Tony607/Focal_Loss_Keras/

#project :

### Imbalanced Learning

Imbalanced data involves cases where the different classes in your dataset are not balanced. In some circumcstances this is not a problem, for example when the sample accurately represents the true popoulation. However there are specific cases where it presents a problem, in particular when it is important to accurately classify an underrepresented class. In general this is important when the cost of missclassification is high, e.g. when the true population is more balanced than the sample. Also, as in our case, detecting fraud. While fraudulent transactions are rare, they are costly when they go undetected, because credit card companies must reimburse the customer for the fraudulent spending.

Imbalanced learning, then, involves techniques which are focused on overcoming the problems of imbalanced data, and accurately predicting to minimize the costs of misclassification in the true population. We examine several techniques, which can fall into two main types:

not that imbalanced learning only refers to the case of labeled data.
We apply the following methods to a convolution neural network. Focal Loss specifically is a loss function which is designed for convolution neural networks. While other methods we use are more flexible in application, we focus on the use with convolution neural networks in order to better evaluate th results.

### Sampling
* SMOTE (oversampling)
* NearMiss (undersampling)

### Loss Functions
* focal loss
* Balanced Cross Entropy (class weights, we use inverse class frequency)
* Asymmetric Loss for Cross Entropy
* Cross Entropy loss (control)



### Balanced Cross Entropy

provide class weights, so that misclassification and classification of one class is more important than the other.

### Asymmetric Cross Entropy

similar to balanced cross entropy but finer details. Asymmetric Loss refers to a class of loss functions where loss is calculated differently based on both the class as well as the correct vs incorrect classification. In cases where costs are unique, this can be easily reflected in the loss function. In our case, since costs are associated more with missclassification of fraud as real, this has higher weight than the others.

In essence, imbalanced learning involves accomodating different costs of misclassification. Since these costs are context specific, we also investigate approaches specific to our domain of application. Specifically we use two asymmetric loss functions: one which weights Type II error higher , and one which uses the amount in the transaction as a weight for the Type II error cost.



 
 ### Focal Loss
 Focal loss in a type of loss function desinged for convolution neural networks, modified from Cross Entropy Loss. The idea is that on successive passes, weights are decreased for observations which have already been predicted correctly. In effect, the relative weight of difficult-to-classify observations increases.
 
 ### Focal Loss
  * benefit: faster training by skipping overrepresented data ?? (https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-focal-loss-in-object-detection/)
  * this approach si speciically applide to CNN (?)
  ### Cross Entropy Loss
  penaliz wrong predictions more than right predictions
  ### Unequal Costs
  In some applications costs of misclassification may differ. In our case of email fraud this is readily apparent: misclassifying an transation as fraud causes some transactional costs and inconvenience to the consumer, but misclassifying a fraudulent transaction as real can cost the consumer or the credit card company thousands of dollars. For this reason, it seems prurient to consider a case where the amount is a function of our missclassification cost.
 
  $ E[classify as fraud | not fraud ] < $$ E[classify as real | fraud] $
 
 
  ### evaluation metrics
  F1 score and others commonly mentioned.
  https://keras.io/api/metrics/classification_metrics/
  recommended: 
  AUC
 
  also interesting for applied approach: total dollars lost?