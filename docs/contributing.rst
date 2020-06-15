Contributing
------------

First of all, thank you for considering contributing to ``tslearn``.
It's people like you that will help make ``tslearn`` a great toolkit.

Contributions are managed through GitHub Issues and Pull Requests.

We are welcoming contributions in the following forms:

- **Bug reports**: when filing an issue to report a bug, please use the search tool to ensure the bug hasn't been reported yet;
- **New feature suggestions**: if you think ``tslearn`` should include a new algorithm, please open an issue to ask for it (of course, you should always check that the feature has not been asked for yet :). Think about linking to a pdf version of the paper that first proposed the method when suggesting a new algorithm.
- **Bug fixes and new feature implementations**: if you feel you can fix a reported bug/implement a suggested feature yourself, do not hesitate to:

  1. fork the project;
  2. implement your bug fix;
  3. submit a pull request referencing the ID of the issue in which the bug was reported / the feature was suggested;

If you would like to contribute by implementing a new feature reported in the Issues, maybe starting with `Issues that are attached the "good first issue" label <https://github.com/tslearn-team/tslearn/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_ would be a good idea.

When submitting code, please think about code quality, adding proper docstrings including doctests with high code coverage.

More details on Pull requests
=============================

The preferred workflow for contributing to tslearn is to fork the
`main repository <https://github.com/tslearn-team/tslearn>`_ on
GitHub, clone, and develop on a branch. Steps:

1. Fork the `project repository <https://github.com/tslearn-team/tslearn>`_
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone your fork of the tslearn repo from your GitHub account to your local disk::

      $ git clone git@github.com:YourLogin/tslearn.git
      $ cd tslearn

3. Create a ``my-feature`` branch to hold your development changes.
   Always use a ``my-feature`` branch. It's good practice to never work on the ``master`` branch::

     $ git checkout -b my-feature

4. Develop the feature on your feature branch. To record your changes in git,
   add changed files using ``git add`` and then ``git commit`` files::

     $ git add modified_files
     $ git commit

5. Push the changes to your GitHub account with::

    $ git push -u origin my-feature

6. Follow `these instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
`Git documentation <https://git-scm.com/documentation>`_ on the web, or ask a friend or another contributor for help.)

Pull Request Checklist
^^^^^^^^^^^^^^^^^^^^^^

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  Follow the PEP8 Guidelines.

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description.
   This will make sure a link back to the original issue is created.

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  Please prefix the title of your pull request with ``[MRG]`` (Ready for
   Merge), if the contribution is complete and ready for a detailed review.
   An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be prefixed ``[WIP]`` (to indicate a work
   in progress) and changed to ``[MRG]`` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   `task list <https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments>`_
   in the PR description.

-  When adding additional functionality, provide at least one
   example script in the ``tslearn/docs/examples/`` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available in tslearn.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with
   `non-regression tests <https://en.wikipedia.org/wiki/Non-regression_testing>`_.
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, this tests should fail for
   the code base in master and pass for the PR code.

-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and
   the example.

Here is a description of useful tools to check your code locally:

- No `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ or `PEP257 <https://www.python.org/dev/peps/pep-0257/>`_ errors;
  check with the `flake8 <https://flake8.pycqa.org/en/latest/>`_ Python package::

     $ pip install flake8
     $ flake8 path/to/module.py  # check for errors in one file
     $ flake8 path/to/folder  # check for errors in all the files in a folder
     $ git diff -u | flake8 --diff  # check for errors in the modified code only

- To run the tests locally and get code coverage, use the
  `pytest <https://docs.pytest.org/en/latest/>`_ and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ Python packages::

     $ pip install pytest pytest-cov
     $ pytest --cov tslearn

- To build the documentation locally, install the following packages and run
  the ``make html`` command in the ``tslearn/docs`` folder::

     $ pip install sphinx==1.8.5 sphinx-gallery sphinx-bootstrap-theme nbsphinx
     $ pip install numpydoc matplotlib
     $ cd tslearn/docs
     $ make html

  The documentation will be generated in the ``_build/html``. You can double
  click on ``index.html`` to open the index page, which will look like
  the first page that you see on the online documentation. Then you can move to
  the pages that you modified and have a look at your changes.

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output.
