# Checklist for releasing new eemeter package versions

## All releases (major, minor, patch)

### Before and after merging branches into develop or master:

#### Tests

- [ ] Tests are passing for all python versions locally
- [ ] Tests are passing for all python versions on travis
- [ ] Test coverage has not decreased

#### Documentation

- [ ] New functions and classes that have appropriate doc strings
- [ ] Sphinx documentation finds new fuctions and classes
- [ ] Usage, parameters, and return values for new functions and classes are documented
- [ ] Documentation builds with no warnings locally
- [ ] Documentation builds with no warnings on readthedocs

#### Code

- [ ] Should be reviewed by an Open EE team member
- [ ] Should conform to PEP 008 (use flake8)

#### Security

- [ ] Should undergo automated security testing

#### Pull requests

- [ ] Should point at develop branch (not master)

### Release flow (optionally use the git-flow tool to help enforce this pattern)

- [ ] Merge feature branches into 'develop'
- [ ] Create a new release branch from develop called 'release/vMAJOR.MINOR.PATCH-TAG', e.g., 'release/v0.4.13-alpha'
- [ ] Bump version using semantic versioning ([SemVer](http://semver.org/))
  - In `docs/conf.py`, the release and version strings
  - In `eemeter/__init__.py`, the `VERSION` tuple
  - In `docs/eemeter_installation.py`, the output string from the line `import eemeter; eemeter.get_version()`
- [ ] Publish release branch to github
- [ ] Merge release branch into 'master'
- [ ] Tag the release with its name (on 'master')
- [ ] Push tags (`git push --tags`)
- [ ] Back-merge 'master' into 'develop'
- [ ] Remove the local release branch (optional)
- [ ] Build a source distribution (`python setup.py sdist`)
- [ ] Build a universal wheel distribution (`python setup.py bdist_wheel --universal`)
- [ ] Publish distributions to the Python Pacakge Index (PyPI) (e.g., `twine upload dist/*X.X.X*`)
- [ ] Verify that `pip install eemeter --upgrade --no-deps` installs the updated version by running `import eemeter; eemeter.get_version()`

## Extra steps for patch releases

None

## Extra steps for minor releases

None

## Extra steps for major releases

None
