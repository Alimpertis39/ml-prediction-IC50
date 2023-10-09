import numpy as np
from matplotlib import pyplot as plt
import copy
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def bioactivity_class_fun(ic50):
    if float(ic50) < 1:
        return 'active'
    elif float(ic50) > 5:
        return 'inactive'
    else:
        return 'intermediate'


def extract_genes_from_pickle(file):
    with open(file_pickle, 'rb') as file:
        gene_list = pickle.load(file)
    return gene_list


def extract_morgan_fp_from_smiles(file_smiles):
    molecules = []
    molecule_names = []
    molecules_objects = []
    dict_mol_name_obj = copy.deepcopy(dict())
    dict_mol_name_str = copy.deepcopy(dict())
    with open(file_smiles, 'r') as file_smile:
        for line in file_smile:
            parts = line.strip().split()
            smiles_string = parts[0]
            molecule_name = " ".join(parts[1:])
            molecules.append(smiles_string)
            molecule_names.append(molecule_name)
            mol = Chem.MolFromSmiles(smiles_string)
            molecules_objects.append(mol)
            if mol is not None:
                # Generate Morgan fingerprint
                radius = 2
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius)
                # Convert the fingerprint to a binary bitstring
                morgan_fp_bits = list(morgan_fp.ToBitString())
                morgan_fp_bits = [int(x) for x in morgan_fp_bits]

                # print("Morgan Fingerprint Bitstring: ", ''.join(morgan_fp_bits))
                # Convert the bitstring to a list of integers
                # morgan_fp_integers = [int(bit) for bit in morgan_fp_bits]
                dict_mol_name_str[molecule_name] = smiles_string
                dict_mol_name_obj[molecule_name] = morgan_fp_bits
    return dict_mol_name_obj


def remove_duplicates(df):
    # remove duplicate rows from train
    df_duplicates = df.copy()
    df_duplicates['Average_IC50'] = \
        df_duplicates.groupby(['drug', 'cell_line'])['IC50'].transform('mean')
    # Drop the original "IC50" column
    df_duplicates.drop(columns=['IC50'], inplace=True)
    # Rename the "Average_IC50" column to "IC50"
    df_duplicates.rename(columns={'Average_IC50': 'IC50'}, inplace=True)
    # Select the two columns and drop duplicate combinations
    df_without_duplicates = df_duplicates.drop_duplicates(
        subset=['drug', 'cell_line'])
    return df_without_duplicates


def remove_drugs(df, dict_mol_name_obj):
    morgan_fingerprints = df['drug'].map(dict_mol_name_obj)
    df_filtered_negatives_duplicates = df.copy()
    df_filtered_negatives_duplicates['morgan_fingerprint'] = morgan_fingerprints
    # missing_values = df_filtered_negatives_duplicates.isnull().sum()
    # print('Missing values: ', missing_values)
    # we dropped the rows without structure

    df_filtered_drugs = df_filtered_negatives_duplicates.dropna()
    return df_filtered_drugs


def integrate_profile_matrix(df, df_profile_matrix):
    df_filtered_profile_matrix_with_nans = df.merge(df_profile_matrix, on='cell_line',
                                                                           how='left')
    df_filtered_profile_matrix = df_filtered_profile_matrix_with_nans.dropna()
    return df_filtered_profile_matrix


file_train = 'gdsc_cell_line_ic50_train_fraction_0.9_id_997_seed_42.csv'
file_test = 'gdsc_cell_line_ic50_test_fraction_0.1_id_997_seed_42.csv'
file_gene_expression_profile_matrix = 'gdsc-rnaseq_gene-expression.csv'
file_pickle = '2128_genes.pkl'
file_info = 'Cell_lines_infos.csv'
file_smiles = 'gdsc.smi'

# Load the data from the pickle file and print the list of genes to focus on
genes_list = extract_genes_from_pickle(file_pickle)
print('The number of genes: ', len(genes_list))
##########################################################################################################################################

print('Start SMILES')
dict_mol_name_obj = extract_morgan_fp_from_smiles(file_smiles)

##########################################################################################################################################

df_train = pd.read_csv(file_train, sep=',')
rows = len(df_train.axes[0])
print('The rows of train at the beginning: ', rows)

df_test = pd.read_csv(file_test, sep=',')
rows = len(df_test.axes[0])
print('The rows of test at the beginning: ', rows)

df_profile_matrix = pd.read_csv(file_gene_expression_profile_matrix, sep=',')
rows = len(df_profile_matrix.axes[0])
print('The rows of profile matrix at the beginning: ', rows)

df_cell_lines_info = pd.read_csv(file_info, sep=',')
rows = len(df_cell_lines_info.axes[0])
print('The rows of cell info at the beginning: ', rows)

# remove rows with negatice IC50 values
df_train_filtered_negatives = df_train[df_train['IC50'] >= 0]
df_test_filtered_negatives = df_test[df_test['IC50'] >= 0]

rows = len(df_train_filtered_negatives.axes[0])
print('The rows of train after removing negatives: ', rows)
rows = len(df_test_filtered_negatives.axes[0])
print('The rows of test after removing negatives: ', rows)

# remove duplicate rows from train
df_train_filtered_negatives_duplicates = remove_duplicates(df_train_filtered_negatives)
rows = len(df_train_filtered_negatives_duplicates.axes[0])
print('The rows of train after dropping duplicates: ', rows)

# remove duplicate rows from test
df_test_filtered_negatives_duplicates = remove_duplicates(df_test_filtered_negatives)
rows = len(df_test_filtered_negatives_duplicates.axes[0])
print('The rows of train after dropping duplicates: ', rows)
###################################################################################################

print('-------------------------Morgan Fingerprints for train set---------------------')
df_train_filtered_drugs = remove_drugs(df_train_filtered_negatives_duplicates, dict_mol_name_obj)
print('The rows of train after filtering the drugs structures:  ', df_train_filtered_drugs.shape[0])


print('-------------------------Morgan Fingerprints for test set---------------------')
df_test_filtered_drugs = remove_drugs(df_test_filtered_negatives_duplicates, dict_mol_name_obj)
print('The rows of test after filtering the drugs structures:  ', df_test_filtered_drugs.shape[0])

###################################################################################################

# keep the genes provided in the pickle file
df_profile_matrix = df_profile_matrix.rename(columns={"Unnamed: 0": "cell_line"})
cell_line_names = df_profile_matrix['cell_line'].tolist()
# drop the columns(genes) that don't exist in the pickle file and insert again the cell line column
df_profile_matrix = df_profile_matrix.drop(columns=[col for col in df_profile_matrix.columns if col not in genes_list])
df_profile_matrix.insert(0, "cell_line", cell_line_names)

# drop duplicates cell lines because of different datasets and keep the TCGA for each cancer cell line
df_cell_lines_info.rename(columns={'Name': 'cell_line'}, inplace=True)
# drop call lines with Nan and unclassified TCGA value
df_cell_lines_info.dropna(inplace=True)
df_cell_lines_info = df_cell_lines_info.drop_duplicates(subset=['cell_line'])
df_cell_lines_info = df_cell_lines_info.loc[df_cell_lines_info['TCGA'] != 'UNCLASSIFIED']

cell_lines_to_use = df_cell_lines_info['cell_line'].unique().tolist()
df_profile_matrix = df_profile_matrix.loc[df_profile_matrix['cell_line'].isin(cell_lines_to_use)]
df_profile_matrix.fillna(0, inplace=True)
# this is another way
intersection = list(set(cell_line_names).intersection(set(cell_lines_to_use)))
print(len(intersection), ' should be equal ', len(df_profile_matrix.axes[0]))

# Cell line in the intersection
cell_line_final = df_profile_matrix['cell_line'].unique()

###############################################################################################
print('Start integrating cell info and profile matrix to our datasets')
print('The rows of train set: ', len(df_train_filtered_drugs.axes[0]))
df_train_filtered_profile_matrix = integrate_profile_matrix(df_train_filtered_drugs, df_profile_matrix)
print('The rows of train after filtering the profile matrix:  ', df_train_filtered_profile_matrix.shape[0])

df_train_filtered_profile_matrix_cell_info = df_train_filtered_profile_matrix.merge(
    df_cell_lines_info[['cell_line', 'TCGA']], on='cell_line', how='left')
# Use one-hot encoding for the categorical variables
one_hot_encoded = pd.get_dummies(df_train_filtered_profile_matrix_cell_info['TCGA'], prefix='TCGA').astype(int)
df_train_filtered_profile_matrix_cell_info = pd.concat([df_train_filtered_profile_matrix_cell_info, one_hot_encoded],
                                                       axis=1)
print('The rows of train after filtering the cell info ', df_train_filtered_profile_matrix_cell_info.shape[0])

print('The rows of test set: ', len(df_test_filtered_drugs.axes[0]))
df_test_filtered_profile_matrix = integrate_profile_matrix(df_test_filtered_drugs, df_profile_matrix)
print('The rows of test after filtering the profile matrix:  ', df_test_filtered_profile_matrix.shape[0])

df_test_filtered_profile_matrix_cell_info = df_test_filtered_profile_matrix.merge(
    df_cell_lines_info[['cell_line', 'TCGA']], on='cell_line', how='left')
# Use one-hot encoding for the categorical variables
one_hot_encoded = pd.get_dummies(df_test_filtered_profile_matrix_cell_info['TCGA'], prefix='TCGA').astype(int)
df_test_filtered_profile_matrix_cell_info = pd.concat([df_test_filtered_profile_matrix_cell_info, one_hot_encoded],
                                                      axis=1)
print('The rows of test after filtering the cell info ', df_test_filtered_profile_matrix_cell_info.shape[0])

# drop the first unnamed column of the datasets
df_train_filtered_profile_matrix_cell_info = df_train_filtered_profile_matrix_cell_info.drop(['Unnamed: 0'], axis=1)
df_test_filtered_profile_matrix_cell_info = df_test_filtered_profile_matrix_cell_info.drop(['Unnamed: 0'], axis=1)

# Add a new categorical variable depending on the target value that I apply one hot encoding to it
df_train_filtered_profile_matrix_cell_info['bioactivity_class'] = df_train_filtered_profile_matrix_cell_info[
    'IC50'].apply(bioactivity_class_fun)
bioactivity_class_counts = df_train_filtered_profile_matrix_cell_info['bioactivity_class'].value_counts()
categories = bioactivity_class_counts.index
counts = bioactivity_class_counts.values
plt.figure(figsize=(8, 6))
plt.bar(categories, counts)
plt.title('Frequencies of Bioactivity Classes')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
# Show the plot
plt.savefig('frequencies.png')
plt.close()


df_test_filtered_profile_matrix_cell_info['bioactivity_class'] = df_test_filtered_profile_matrix_cell_info[
    'IC50'].apply(bioactivity_class_fun)

################################################################################################################

# use the morgan fingerprint info using the total number of ones
df_train_filtered_profile_matrix_cell_info['morgan_fingerprint'] = df_train_filtered_profile_matrix_cell_info[
    'morgan_fingerprint'].apply(lambda x: np.sum(x))

df_test_filtered_profile_matrix_cell_info['morgan_fingerprint'] = df_test_filtered_profile_matrix_cell_info[
    'morgan_fingerprint'].apply(lambda x: np.sum(x))

print(df_train_filtered_profile_matrix_cell_info.head(25))
X = df_train_filtered_profile_matrix_cell_info.drop(
    columns=['IC50', 'drug', 'cell_line', 'bioactivity_class', 'TCGA'])  # Features
y = df_train_filtered_profile_matrix_cell_info['IC50']  # Target variable

# (90% training, 10% validation/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
print('train set rows and columns before pca\n')
print(X_train.shape[0])
print(X_train.shape[1])
# PCA transformer
pca = PCA(n_components=0.90, random_state=42)  # Retain 90% of the variance

X_train_pca = pca.fit_transform(X_train)
print('train set  rows and columns after pca\n')
print(X_train_pca.shape[0])
print(X_train_pca.shape[1])

X_test_pca = pca.transform(X_test)

print('finished pca')
rf_regressor = RandomForestRegressor(n_estimators=50, random_state=42)

rf_regressor.fit(X_train_pca, y_train)
y_pred = rf_regressor.predict(X_test_pca)

# svr_regressor = svr_regressor = SVR(kernel='linear', C=1.0)

# svr_regressor.fit(X_train_pca, y_train)

# y_pred = svr_regressor.predict(X_test_pca)

y_pred = rf_regressor.predict(X_test_pca)

# Perform k-fold cross-validation 
# k = 5
# cross_val_scores = cross_val_score(rf_regressor, X_train_pca, y_train, cv=k, scoring='r2')
# # cross_val_scores = cross_val_score(tree_regressor, X_train_pca, y_train, cv=k, scoring='r2')

# # Calculate the mean R-squared and standard deviation of R-squared
# mean_r2 = np.mean(cross_val_scores)
# std_r2 = np.std(cross_val_scores)

# print(f"CV - Mean R-squared: {mean_r2:.2f}")
# print(f"CV - Standard Deviation of R-squared: {std_r2:.2f}")


mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f"RMSE of  prediction: {rmse:.2f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, facecolors='none', edgecolors='orange', marker='o', label='True vs. Predicted')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)],
         linestyle='--', color='blue', label='Perfect Prediction')

plt.legend()
plt.title('True vs. Predicted Values')
plt.savefig('predicted_vs_true_values.png')
