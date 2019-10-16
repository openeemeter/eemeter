#!/bin/bash

set -e  # fail script on any error, show commands

OLD_VERSION=$1  # e.g., 0.0.0
NEW_VERSION=$2  # e.g., 0.0.1
NEW_VERSION_LENGTH=$(printf "%s" "$NEW_VERSION" | wc -c)
DASHES=$(printf "%${NEW_VERSION_LENGTH}s" | sed 's/ /-/g')

echo "git checkout master"
echo "git pull"
echo "git checkout -b release/v${NEW_VERSION}"
echo ""
echo "sed -i -e 's/${OLD_VERSION}/${NEW_VERSION}/g' eemeter/__version__.py"
echo "sed -i -e '/Development/,/-----------/ c\\
Development\\
-----------\\
\\
* Placeholder\\
\\
${NEW_VERSION}\\
${DASHES}\\
' CHANGELOG.md"
echo "rm -f eemeter/__version__.py-e"
echo "rm -f CHANGELOG.md-e"
echo ""
echo "git commit -am \"Bump version\" -s"
echo "git tag v${NEW_VERSION}"
echo "git push -u origin release/v${NEW_VERSION} --tags"
echo "git checkout master"
echo "git pull"
echo "git merge release/v${NEW_VERSION}"
echo "git push"
echo ""
echo "rm -r dist build eemeter.egg-info"  # reset build dirs
echo "git checkout v${NEW_VERSION}"
echo "python setup.py upload"  # requires PyPI credentials
echo "git checkout master"
