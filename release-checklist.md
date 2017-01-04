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

#### 1. Merge feature branch(es) into `develop`
```
   cd eemeter
   git checkout develop
   git merge my-feature-branch develop
```

#### 2. Create release branch
* Verify the latest tagged release on github
* Determine the next version you'll be releasing using [semantic versioning](http://semver.org/)
* Create release branch from `develop`, named in the form _release/vMAJOR.MINOR.PATCH-TAG_, for example _release/v0.4.13-alpha_

  ```
    git checkout -b release/vMAJOR.MINOR.PATCH-TAG
  ```

* Update the version number in the following files:
  * `docs/conf.py`: the release and version strings
  * `eemeter/__init__.py`: the `VERSION` tuple
  * `docs/eemeter_installation.py`: the output string from the line `import eemeter; eemeter.get_version()`
* Publish release branch
  ```
    git push origin release/vMAJOR.MINOR.PATCH-TAG
  ```

#### 3. Tag the release
* Merge release branch into `master`

  ```
    git checkout master
    git merge release/vMAJOR.MINOR.PATCH-TAG master
  ```

* Tag the release with its name on `master`

  ```
    git checkout master   # ensure you're on the right branch
    git tag vMAJOR.MINOR.PATCH-TAG
    git push origin --tags
    git push origin master
  ```

* Merge master into develop

  ```
    git checkout develop
    git merge master
    git push origin develop
    git branch -d release/vMAJOR.MINOR.PATCH-TAG  # Remove the local release branch
  ```

#### 4. Prepare distributions
* Start vagrant and navigate to `eemeter`
* Build a source distribution

  ```
    python3 setup.py sdist
  ```
* Build a universal wheel distribution

  ```
    python3 setup.py bdist_wheel --universal
  ```
* Publish both distributions to the Python Pacakge Index (PyPI)

  ```
    twine upload dist/<dist-filename>
    # Use PyPI credentials in 1Password
  ```
* Verify eemeter version
  * Install eemeter by running `pip install eemeter --upgrade --no-deps`
  * Start python shell and run the following:

  ```
    import eemeter
    eemeter.get_version()
  ```


## Extra steps for patch releases

None

## Extra steps for minor releases

None

## Extra steps for major releases

None
