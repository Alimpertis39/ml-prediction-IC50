# ml-prediction-IC50

*** Construct a machine learning model to predict IC50 values!***

### Data preprocessing
After loading all the data files (train, test , gene_expression_matrix, cell_line_info, smile file):
1. Remove rows with negatice IC50 values in order no to affect any result.
2. Remove duplicates by aggregating the duplicate lines to extract their mean IC50 value.
3. Extract the given given structures from the SMILES file. Therefore, keep only the drugs with known structure.
4. Extract the genes from the pickle file that I used for the gene expression profile matrix. I replace the NaN vaules with 0.
5. Remove the rows of the Cell line info file with UNCLASSIFIED and NaN TCGA value.

### Feature Engineering
1. Convert the given structures to Morgan Fingerprints.
2. After the data preprocessing of the gene expression profile matrix and the cell line info, the cell lines of the intersection of these two datasets are kept in order to used.
3. Then, merge these datasets to the train and test sets to integrate extra information and remove the NaN value rows.
4. The TCGA value, which is a categorical variable, is encoded with one-hot encoding, and therefore create binary columns for each one of the possible labels of the categorical variable.
5. Create a new categorical variable based on the
6. From the Morgan Fingerprints for each drug structure, take the total number of ones
7. Finally, apply PCA to reduce the number of columns, keeping the 90% of the variance that explained the original columns.
   * One feature engineering that improve the results would be normalization on the profile matrix data. Then the Z-score used the mean and the std of each column(expressed gene). Important: these mean and std of the train set would used on the test set in order to avoid bias on the test set* 

Some ways to improve the prediction results: Normalization to the gene expression matrix values. I would normalize the train set to have Z-score. I would apply the mean and the std of the train set to the test set in order to avoid the bias to the test set.

Better results with Random forest. However, I also had good results with decision tree. I also used Support Vector Regression. It had a bit worse results.
Of course, there are many methods to optimize the parameters of the models, like the n_estimator of Random Forest, which gives us better results.

Cross Validation to avoid overfitting: the algorithm is consistent. Low variation between the folds.

