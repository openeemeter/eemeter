### Your checklist for this pull request

Please review the [guidelines for contributing](../CONTRIBUTING.md) to this repository.

- [ ] Make sure you are requesting to **pull a feature/bugfix branch** (right side). Don't request your master!
- [ ] Make sure tests pass and coverage has not fallen `docker-compose run --rm test`.
- [ ] Update the [CHANGELOG.md](../CHANGELOG.md) to describe your changes in a bulleted list under the "Development" section at the top of the changelog. If this section does not exist, create it.
- [ ] Make sure code style follows PEP 008 using `docker-compose run --rm blacken`.
- [ ] Make sure that new functions and classes have inline docstrings and are
  included in [docs/api.rst](../docs/api.rst) for the sphinx build. Please use [numpy-style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#google-vs-numpy).
  Sphinx docs can be built with the following command: `docker-compose run --rm --entrypoint="make -C docs html" shell`. Please note and fix any warnings.
- [ ] Make sure that all git commits are have the "Signed-off-by" message for
  the Developer Certificate of Origin. When you're making a commit, just add
  the `-s/--signoff` flag (e.g., `git commit -s`).

### Description

Please describe your pull request.
