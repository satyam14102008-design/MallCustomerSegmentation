# Mall Customer Segmentation — PyCharm Project

## Included
- `main.py`: standardization, PCA, K-Means with silhouette-based K selection, DBSCAN, cluster profiling, and output generation.
- `data/Mall_Customers.csv`: the supplied dataset.
- `requirements.txt`: required Python packages.
- `outputs/`: generated plots and CSV results will be saved here.

## Run in PyCharm
1. Extract this ZIP to a folder on your computer.
2. Open PyCharm and choose **Open**, then select the extracted `Mall_Customer_Segmentation_PyCharm` folder.
3. If prompted, create/select a Python virtual environment (Python 3.10+ recommended).
4. Open the PyCharm Terminal and run:

   `python -m pip install -r requirements.txt`

5. Right-click `main.py` and choose **Run 'main'**.
6. Check the `outputs` folder for the generated plots and CSV files.

## Notes
- The code expects the dataset at `data/Mall_Customers.csv`, which is included.
- K-Means tests K values from 2 to 8 and selects the highest silhouette score among those tested.
- DBSCAN uses starting parameters `eps=0.7` and `min_samples=5`; tune these if the results are not useful.
- PCA is used to visualize the clusters in two dimensions. Clustering and silhouette-based K selection use standardized original numeric features.
