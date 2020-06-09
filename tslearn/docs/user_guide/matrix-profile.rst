.. _matrix-profile:

Matrix Profile
==============

The Matrix Profile, :math:`MP`, is a new time series that can be calculated based on an input time series :math:`T` and a subsequence length :math:`m`. :math:`MP_i` corresponds to the minimal distance (excluding the distance segment itself) from the query subsequence :math:`T_{i\rightarrow i+m}` to :math:`T` [1]_.  The Matrix Profile is closely related to the distance calculation used to transform time series into their shapelet-transform space, as illustrated below.

.. figure:: ../_static/img/distance_matrix_profile.svg
    :width: 80%
    :align: center

    For each segment, the distances to all subsequences of the time series are calculated and the minimal distance that not corresponds to the original location of the segment (where the distance is zero) is returned.


Possible Applications
---------------------

The Matrix Profile allows for many possible applications, which are well documented on the page created by the original authors [2]_. Some of these applications include: motif and shapelet extraction, discord detection, earthquake detection, and many more.


.. minigallery:: tslearn.matrix_profile.MatrixProfile
    :add-heading: Examples Involving Matrix Profile
    :heading-level: -


.. raw:: html

    <div style="clear: both;" />

Other implementations
---------------------

We acknowledge that there are more efficient Matrix Profile implementations out there that include GPU support, such as [3]_. We provide a matrix profile implementation such that it can easily be used in complement with other tslearn estimators, for example in a pipeline. Moreover, we provide `functionality <https://tslearn.readthedocs.io/en/stable/integration_other_software.html#stumpy>`_ to convert from one format to the other and vice versa.

References
----------

.. [1] C. M. Yeh, Y. Zhu, L. Ulanova, N.Begum et al.
       Matrix Profile I: All Pairs Similarity Joins for Time Series: A
       Unifying View that Includes Motifs, Discords and Shapelets.
       ICDM 2016.
.. [2] https://www.cs.ucr.edu/~eamonn/MatrixProfile.html
.. [3] https://github.com/TDAmeritrade/stumpy
