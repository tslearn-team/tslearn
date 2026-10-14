# -*- coding: utf-8 -*-
"""
Nearest Neighbors Classification (Chinatown dataset)
====================================================

This example demonstrates the use of k-nearest neighbors based classifiers for time series
:class:`~tslearn.neighbors.KNeighborsTimeSeriesClassifier` and examines the impact 
of different distance metrics as model parameters on the `Chinatown` time series dataset .

We compare the predictive performance of classifiers [1] fitted with three different 
metrics: Dynamic Time Warping (DTW )[2], Euclidean distance and Symbolic Aggregate 
approXimation distance (SAX-MINDIST) [3] across several values of k.

[1] `Wikipedia entry for the k-nearest neighbors algorithm
<https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_

[2] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
for spoken word recognition". IEEE Transactions on Acoustics, Speech, and
Signal Processing, 26(1), 43-49 (1978).

[3] J. Lin, E. Keogh, L. Wei and S. Lonardi, "Experiencing SAX: a novel
symbolic representation of time series". Data Mining and Knowledge Discovery,
15(2), 107-144 (2007).

"""

# sphinx_gallery_start_ignore
import warnings
warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

# Authors: Romain Tavenard, Anna Bobasheva
# License: BSD 3 clause

##############################################################################
# Load the dataset
# ----------------
#
# In this example we use the `Chinatown dataset from the UCR/UEA archive
# <https://www.timeseriesclassification.com/description.php?Dataset=Chinatown>`_ .
#
# The dataset consists of pedestrian counts recorded at the Chinatown location in 
# Melbourne, Australia. Each time series represents the number of pedestrians 
# detected during a day, weekday or weekend. 
#
# The dataset is scaled to the [0, 1] range to ensure that time series
# are on a comparable scale before applying distance-based classification.
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax

loader = UCR_UEA_datasets()
# sphinx_gallery_start_ignore
if "__file__" not in locals():
    # runs by sphinx-gallery
    import os
    loader._data_dir = os.path.join(
        os.path.dirname(os.path.realpath(os.getcwd())), '..', "datasets"
    )
# sphinx_gallery_end_ignore
X_train, y_train, X_test, y_test = loader.load_dataset("Chinatown")

scaler = TimeSeriesScalerMinMax() 
X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

##############################################################################
# Nearest neighbor classification
# --------------------------------------
#
# We train multiple k-nearest neighbors classifiers for time series with different
# configurations:
#
# * *Distance metrics*: 
#
#   * *dtw* - distance metrics which can handle temporal shifts
#   * *euclidean* - standard point-wise distance measurement
#   * *sax* - distance metric which works on symbolic representations of time series
#
# * *k values*: number of neighbors (k) from 1 to 9 
#
# For each combination, we compute the F1 score of the test set predictions.
from sklearn.metrics import accuracy_score, f1_score
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

distance_metrics = ["dtw", "euclidean", "sax"]
k_values = [1, 3, 5, 7, 9]  
results = []

for metric in distance_metrics:
    # The SAX distance metric requires special parameters
    metric_params = {'n_segments': 10, 'alphabet_size_avg': 5}  if metric == "sax" else {}

    for k in k_values:
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric=metric,
                                                  metric_params=metric_params )
        knn_clf.fit(X_train, y_train)
        y_pred = knn_clf.predict(X_test)

        # Store results
        results.append({ 
            'metric': metric,
            'k': k,
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        })

##############################################################################
# Now we create a radar plot to visualize the F1 scores for each distance metric
# and k value combination, where:
# 
# * Each axis represents a k value (1, 3, 5, 7, 9)
# * Each colored line represents a distance metric (DTW, Euclidean, SAX)
# * The distance from center shows the F1 score (higher is better)
#
# This visualization helps identifying the optimal (distance metric, k value) configuration.
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
plt.subplot(111, projection='polar')

# Number of variables
categories = [f'k={k}' for k in k_values]
N = len(categories)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Colors for each metric
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, metric in enumerate(distance_metrics):
    # Get values for this metric across all k values
    values = [res['f1_score'] for k in sorted(k_values) 
              for res in results if res['metric'] == metric and res['k'] == k]
    values += values[:1]  # Complete the circle
    
    plt.plot(angles, values, 'o-', linewidth=2, label=metric, color=colors[i])
    plt.fill(angles, values, alpha=0.25, color=colors[i])

# Add best F1 score label
best_model = max(results, key=lambda x: x['f1_score'])
plt.text(1.3, 0, 
         f'Best score: {best_model["f1_score"]:.3f}\n@({best_model["metric"]}, k={best_model["k"]})',
         ha ='right', va='bottom', weight='bold', 
         transform=plt.gca().transAxes,         
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='lightgrey')  )


plt.xticks(angles[:-1], categories, fontsize=12)
plt.ylim(0, 1)
plt.title('F1 score Radar Chart', fontsize=16)
plt.legend(title='Distance Metric', loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.show()

##############################################################################
# Conclusion
# ----------
#
# We observe that DTW distance generally outperforms both Euclidean and SAX metrics across most k values.
# This confirms that accounting for temporal distortion is beneficial for the `Chinatown`
# dataset where the same action may be performed at different speeds.
#
# The clear performance difference highlights why choosing an appropriate distance metric
# is crucial for time series classification tasks, particularly for datasets with temporal variations.



