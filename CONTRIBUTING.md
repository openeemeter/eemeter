How to release code updates 
===========================

1. create branch named feature/examplefeature and make desired changes
2. edit CHANGELOG.md with changes under a new section called `Development`
3. create, review, and merge PR for feature/examplefeature
4. repeat steps 1-3 if desired for other features as convenient, though preference is for frequent version bumps
5. make a new branch called release/vX.X.X
6. bump version - edit __version__.py and rename `Development` section with the new version in CHANGELOG.md
7. commit changes and then create a tag called vX.X.X on release/vX.X.X branch and push both
8. merge release branch to master
9. submit to pypi
