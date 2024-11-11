import numpy as np
import tensorflow as tf
import pandas as pd

#https://towardsdatascience.com/all-you-need-to-know-about-pca-technique-in-machine-learning-443b0c2be9a1
#https://medium.com/@ansjin/dimensionality-reduction-using-pca-on-multivariate-timeseries-data-b5cc07238dc4

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca_train_data = None
        self.pca_val_data = None
        self.pca_test_data = None

    def fit_transform(self, df):
        train_tensor = tf.constant(df.values, dtype=tf.float32)

        # Compute covariance matrix
        cov_matrix = tf.matmul(tf.transpose(train_tensor), train_tensor)

        # Ensure symmetric covariance matrix
        cov_matrix = 0.5 * (cov_matrix + tf.transpose(cov_matrix))

        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = tf.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        eigenvalues_indices = tf.argsort(tf.math.real(eigenvalues), direction='DESCENDING')
        sorted_eigenvalues = tf.gather(eigenvalues, eigenvalues_indices)
        sorted_eigenvectors = tf.gather(eigenvectors, eigenvalues_indices, axis=1)

        # Select top k eigenvectors corresponding to the top k eigenvalues
        pca_matrix = sorted_eigenvectors[:, :self.n_components]

        pca_matrix_val = pca_matrix.numpy()  # Convert to NumPy array

        # Transform the data
        transformed_data = np.dot(df.values, pca_matrix_val)

        return transformed_data

    def convert_to_pca(self, train_df, val_df, test_df):
        self.pca_train_data = self.fit_transform(train_df)
        self.pca_val_data = self.fit_transform(val_df)
        self.pca_test_data = self.fit_transform(test_df)

        # Convert transformed data back to DataFrame
        self.pca_train_data = pd.DataFrame(self.pca_train_data, columns=['PCA Component 1', 'PCA Component 2'], index=train_df.index)
        self.pca_train_data['phoenix_memory_used_cm_sessionP_smf'] = train_df['phoenix_memory_used_cm_sessionP_smf']
        
        self.pca_val_data = pd.DataFrame(self.pca_val_data, columns=['PCA Component 1', 'PCA Component 2'], index=val_df.index)
        self.pca_val_data['phoenix_memory_used_cm_sessionP_smf'] = val_df['phoenix_memory_used_cm_sessionP_smf']

        self.pca_test_data = pd.DataFrame(self.pca_test_data, columns=['PCA Component 1', 'PCA Component 2'], index=test_df.index)
        self.pca_test_data['phoenix_memory_used_cm_sessionP_smf'] = test_df['phoenix_memory_used_cm_sessionP_smf']

        return self.pca_train_data, self.pca_val_data, self.pca_test_data

    
    # def print_mapping():
    #     # Print the mapping of original columns to principal components
    #     print("Mapping of Original Columns to Principal Components:")
    #     original_column_names = [column for column, index in column_indices.items()]
    #     for i in range(pca_matrix_val.shape[1]):
    #         principal_component = pca_matrix_val[:, i]
    #         column_contributions = [(original_column_names[j], principal_component[j]) for j in range(len(original_column_names))]
    #         sorted_contributions = sorted(column_contributions, key=lambda x: abs(x[1]), reverse=True)
    #         print(f"Principal Component {i+1}:")
    #         for column_name, contribution in sorted_contributions:
    #             print(f"\t{column_name}: {contribution}")
