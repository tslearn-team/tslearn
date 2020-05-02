# Contributing

First of all, thank you for considering contributing to `tslearn`. 
It's people like you that will help make `tslearn` a great toolkit.

Contributions are managed through GitHub Issues and Pull Requests.

We are welcoming contributions in the following forms:
* **Bug reports**: when filing an issue to report a bug, please use the search tool to ensure the bug hasn't been reported yet;
* **New feature suggestions**: if you think `tslearn` should include a new algorithm, please open an issue to ask for it (of course, you should always check that the feature has not been asked for yet :). Think about linking to a pdf version of the paper that first proposed the method when suggesting a new algorithm. 
* **Bugfixes and new feature implementations**: if you feel you can fix a reported bug/implement a suggested feature yourself, do not hesitate to:
  1. fork the project;
  2. implement your bugfix;
  3. submit a pull request referencing the ID of the issue in which the bug was reported / the feature was suggested;
  
If you would like to contribute by implementing a new feature reported in the Issues, maybe starting with [Issues that are attached the "good first issue" label](https://github.com/rtavenar/tslearn/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) would be a good idea.

When submitting code, please think about code quality, adding proper docstrings including doctests with high code coverage.

## More details on Pull requests

The preferred workflow for contributing to tslearn is to fork the
[main repository](https://github.com/rtavenar/tslearn) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/rtavenar/tslearn)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the tslearn repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourLogin/tslearn.git
   $ cd tslearn
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

### Pull Request Checklist

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  Follow the PEP8 Guidelines.

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description. 
   This will make sure a link back to the original issue is created.

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  Please prefix the title of your pull request with `[MRG]` (Ready for
   Merge), if the contribution is complete and ready for a detailed review.
   An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be prefixed `[WIP]` (to indicate a work
   in progress) and changed to `[MRG]` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.

-  When adding additional functionality, provide at least one
   example script in the ``tslearn/docs/examples/`` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available in tslearn.

-  Documentation and high-coverage tests are necessary for enhancements to be
   accepted. Bug-fixes or new features should be provided with 
   [non-regression tests](https://en.wikipedia.org/wiki/Non-regression_testing).
   These tests verify the correct behavior of the fix or feature. In this
   manner, further modifications on the code base are granted to be consistent
   with the desired behavior.
   For the Bug-fixes case, at the time of the PR, this tests should fail for
   the code base in master and pass for the PR code.

-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and
   the example.

You can also check for common programming errors with the following
tools:


-  No pyflakes warnings, check with:

  ```bash
  $ pip install pyflakes
  $ pyflakes path/to/module.py
  ```

-  No PEP8 warnings, check with:

  ```bash
  $ pip install pep8
  $ pep8 path/to/module.py
  ```

-  AutoPEP8 can help you fix some of the easy redundant errors:

  ```bash
  $ pip install autopep8
  $ autopep8 path/to/pep8.py
  ```

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the GitHub issue).
