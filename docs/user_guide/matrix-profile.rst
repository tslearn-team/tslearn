.. _matrix-profile:

Matrix Profile
==============

The Matrix Profile, :math:`MP`, is a new time series that can be calculated based on an input time series :math:`T` and a subsequence length :math:`m`. :math:`MP_i` corresponds to the minimal distance from the query subsequence :math:`T_{i\rightarrow i+m}` to any subsequence in :math:`T` [1]_.  As the distance from the query subsequence to itself will be equal to zero, :math:`T_{i-\frac{m}{4}\rightarrow i+\frac{m}{4}}` is considered as an exclusion zone. In order to construct the Matrix Profile, a distance profile which is :ref:`similar to the distance calculation used to transform time series into their shapelet-transform space <shapelets>`, is calculated for each subsequence, as illustrated below:

.. figure:: ../../_images/sphx_glr_plot_distance_and_matrix_profile_001.svg
    :width: 80%
    :align: center

    For each segment, the distances to all subsequences of the time series are calculated and the minimal distance that does not correspond to the original location of the segment (where the distance is zero) is returned.


Implementation
---------------

The Matrix Profile implementation provided in ``tslearn`` uses numpy or wraps around STUMPY [2]_. Three different versions are available:

* ``numpy``: a slow implementation 
* ``stump``: a fast CPU version, which requires STUMPY to be installed
* ``gpu_stump``: the fastest version, which requires STUMPY to be installed and a GPU


Possible Applications
---------------------

The Matrix Profile allows for many possible applications, which are well documented on the page created by the original authors [3]_. Some of these applications include: motif and shapelet extraction, discord detection, earthquake detection, and many more.


.. minigallery:: tslearn.matrix_profile.MatrixProfile
    :add-heading: Examples Involving Matrix Profile
    :heading-level: -


.. raw:: html

    <div style="clear: both;" />

References
----------

.. [1] C. M. Yeh, Y. Zhu, L. Ulanova, N.Begum et al.
       Matrix Profile I: All Pairs Similarity Joins for Time Series: A
       Unifying View that Includes Motifs, Discords and Shapelets.
       ICDM 2016.
.. [2] https://github.com/TDAmeritrade/stumpy
.. [3] https://www.cs.ucr.edu/~eamonn/MatrixProfile.html
