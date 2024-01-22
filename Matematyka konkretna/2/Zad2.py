import numpy as np
import matplotlib.pyplot as plt
import cv2
image = cv2.imread('15.webp')

U_row, S_row, Vt_row = np.linalg.svd(image, full_matrices=False)
U_col, S_col, Vt_col = np.linalg.svd(image.T, full_matrices=False)


U_row_flat = U_row.reshape(-1, U_row.shape[-1])
U_col_flat = U_col.reshape(-1, U_col.shape[-1])

corr_matrix_row = np.corrcoef(U_row_flat, rowvar=False)
corr_matrix_col = np.corrcoef(U_col_flat, rowvar=False)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('wiersze')
plt.imshow(corr_matrix_row, cmap='viridis', aspect='auto')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('kolumny')
plt.imshow(corr_matrix_col, cmap='viridis', aspect='auto')
plt.colorbar()

plt.tight_layout()
plt.show()