# ml-prediction-IC50

*** Construct a machine learning model to predict IC50 values!***

### Data preprocessing


Some ways to improve the prediction results: Normalization to the gene expression matrix values. I would normalize the train set to have Z-score. I would apply the mean and the std of the train set to the test set in order to avoid the bias to the test set.

Better results with Random forest. However, I also had good results with decision tree. I also used Support Vector Regression. It had a bit worse results.
Of course, there are many methods to optimize the parameters of the models, like the n_estimator of Random Forest, which gives us better results.

Cross Validation to avoid overfitting: the algorithm is consistent. Low variation between the folds.

