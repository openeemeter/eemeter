#!/bin/bash

set -e  # fail script on any error, show commands

OLD_VERSION=$1  # e.g., 0.0.0
NEW_VERSION=$2  # e.g., 0.0.1
NEW_VERSION_LENGTH=$(printf "%s" "$NEW_VERSION" | wc -c)
DASHES=$(printf "%${NEW_VERSION_LENGTH}s" | sed 's/ /-/g')

echo "git checkout master"
echo "git pull"
echo ""
echo "sed -i -e 's/${OLD_VERSION}/${NEW_VERSION}/g' gridmeter/__version__.py"
echo "sed -i -e '/Development/,/-----------/ c\\
Development\\
-----------\\
\\
* Placeholder\\
\\
${NEW_VERSION}\\
${DASHES}\\
' CHANGELOG.md"
echo "rm -f setup.py-e"
echo "rm -f CHANGELOG.md-e"
echo ""
echo "git commit . -m \"Bump version\""
echo "git pull"
echo "git push"
echo "# use PyPi credentials in 1Password"
echo "python setup.py sdist bdist_wheel"
echo "twine upload dist/*"
