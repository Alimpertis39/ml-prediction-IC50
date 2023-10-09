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
6. From the Morgan Fingerprints for each drug structure, take the total number of ones.
7. Finally, apply PCA to reduce the number of columns, keeping the 90% of the variance that explained the original columns.
   * One feature engineering that improve the results would be normalization on the profile matrix data. Then the Z-score used the mean and the std of each column(expressed gene). Important: these mean and std of the train set would used on the test set in order to avoid bias on the test set* 

### Model development
1. Random Forest (Multiple decision trees combined to make predictions)
2. SVR (Support Vector Regression)
   
   *Another way to improve the results would be to make optimization parameter, e.g. optimization of the n_estimator of random forest algorithm*
   *Apply Cross Validation with 5 folds to have a more robust model's performance. Important that the performance was almost similar, low variation among folds, so the algorith is consistent.*
   Better results using random forest algorithm.
   About RMSE of  prediction: 1.25, Mean Absolute Error (MAE): 0.99

   ### Interpretation
   1. Scatter plot with true and predicted values, and the x=y line to compare with the optimal prediction.
  






