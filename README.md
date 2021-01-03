# Imbalanced Learning in Binary Classification
> by Sherman Aline & She Zhongqi

## Motivation

Imbalanced data refers to a data which has large class imbalances. This is not a problem in itself. If missclassification costs are equal across classes, and the true population is reflected in the sample population there is no problem. However if either of these does not hold, it is desirable to modify our approach.

Imbalanced Learning, then, refers to methods which try to accomodate the cases when both those requirements do not hold. In essence, these methods devise ways of modifying learning costs to better reflect the real-world costs.

Consider the case of credit card fraud. Fraud is extremely rare, but can be very costly. One fraudulent transaction can mean a loss of thousands of dollars for the company. Without imbalanced learning though, this fact is not internalized into the classification model. One missclassified transaction out of thousands is still a near-100% accuracy.

We compare several common approaches to mitigating the problems of imbalanced data, which can be categorized into two types: modification to the loss function and modification to the sample. Both these approaches, in effect aim to increase the relative weight of missclassified samples in the minority class. We also investigate an approach which is specific to our application: considering the direct monetary cost of missclassification.


 We evaluate these methods in the application to credit card fraud detection, using the dataset available at https://www.kaggle.com/mlg-ulb/creditcardfraud.


##Model

We evaluate the following methods on a convultion neural network model. The CNN model was chosen for evaluation, primarily because Focal Loss is designed for this model. 

#### Loss-Based Methods
* Focal Loss (#focalloss)
* Cross Entropy loss (baseline)
* Balanced Cross Entropy (class weights, we use inverse class frequency)
* Cross Entropy with Monetary Weights
* Focal Loss with Monetary Weights
* Asymmetric Loss for Cross Entropy

#### Sampling-Based Methods
* SMOTE (oversampling)
* NearMiss (undersampling)

First, we provide an overview of each method and why it is useful for imbalanced data. Second, we explain which metrics we used to compare the methods. Finally, we give a recommendation.


### Focal Loss
 Focal Loss, recently designed by Facebook Research, is an extension of Cross Entrop Loss for Convolution Neural Networks. Focal Loss updates sample weights at the end of each epoch, lowering weight on samples which were classified successfully and increasing weight on those which were missclassified. 
 This loss approach was designed for use in Object Detection for Computer Vision.
 Another benefit of this model is that it can effectively train faster by skipping overrepresented data.
### Cross Entropy loss (baseline)
 penaliz wrong predictions more than right predictions

### Balanced Cross Entropy


### Asymmetric Cross Entropy

similar to balanced cross entropy but finer details. Asymmetric Loss refers to a class of loss functions where loss is calculated differently based on both the class as well as the correct vs incorrect classification. In cases where costs are unique, this can be easily reflected in the loss function. In our case, since costs are associated more with missclassification of fraud as real, this has higher weight than the others.

In essence, imbalanced learning involves accomodating different costs of misclassification. Since these costs are context specific, we also investigate approaches specific to our domain of application. Specifically we use two asymmetric loss functions: one which weights Type II error higher , and one which uses the amount in the transaction as a weight for the Type II error cost.


### Balanced Cross Entropy 

(class weights, we use inverse class frequency)

provide class weights, so that misclassification and classification of one class is more important than the other.


### Asymmetric Loss for Cross Entropy

  In some applications costs of misclassification may differ. In our case of email fraud this is readily apparent: misclassifying an transation as fraud causes some transactional costs and inconvenience to the consumer, but misclassifying a fraudulent transaction as real can cost the consumer or the credit card company thousands of dollars. For this reason, it seems prurient to consider a case where the amount is a function of our missclassification cost.
 
  $ E[classify as fraud | not fraud ] < $$ E[classify as real | fraud] $
### Cross Entropy with Monetary Weights

### Focal Loss with Monetary Weights


### SMOTE
SMOTE or Synthetic Minority Over-Sampling Technique, is an extension to oversampling. Where oversampling imputes repeats of existing observations, SMOTE generates new observations. These observations are generated using K-nearest neighbors. First, a K-mean is generated from some n observations from the minority class. Second, the closest point to the mean is calculated, and a new observation is randomly imputed somewhere on the line between the mean and the true observation.

This technique is repeated so "synthesize" the desired number of minority-class samples, so that the data is more balanced.

Because this method increases the total size of the training set, it is slower.
 
### Near Miss

The NearMiss is an under-sampling technique which uses distance to eliminate majority class  samples.

The NearMiss method corresponds to the NearMiss function in the Python library, and here are several versions of applying NearMiss Algorithm:

NearMiss-1: Select samples of the majority class for which average distances to the k closest instances of the minority class is smallest

NearMiss-2: Select samples of the majority class for which average distances to the k farthest instances of the minority class is smallest.

NearMiss-3: This is a two-stage algorithm. First, for each minority sample, retain their M nearest-neighbors samples; then, those majority samples with the largest average distance to N nearest-neighbors samples will be selected.

This method increases the speed of training, since the training set is made smaller.

## Evaluation

 We consider two different approaches in evaluating our models.
 * model based
 * application based 

In the first section we are concerned primarily with general model performance: speed and effectiveness in classifying our minority class. In the second section we focus on the application to credit card fraud, emphasizing the monetary cost and the impact of false-negatives.

### Model Based
#### Metrics
* AUC
*
*

#### Recommendations

###Application Based
####Metrics
*
*
*
#### Recommendations