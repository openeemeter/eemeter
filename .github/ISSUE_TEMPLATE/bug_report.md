---
name: Bug report
about: A report of or request to fix an issue in the eemeter library
---

A complete bug report will help eemeter contributors to reproduce the bug and provide
insight into fixing.

Bug reports must:

1. Include a short, self-contained Python snippet reproducing the problem. You can
  format the code nicely by using GitHub Flavored Markdown:

    ```python
    >>> import eemeter
    >>> import pandas as pd
    >>> meter_data = pd.DataFrame({"start": ..., "value": ...})
    >>> eemeter.get_baseline_data(meter_data)
    ...
    ```

2. Include the full version string of eemeter, pandas, and their dependencies. You can
  use the built-in function:

    >>> import eemeter
    >>> import pandas as pd
    >>> eemeter.get_version()
    >>> pd.show_versions()

3. Explain why the current behavior is wrong/not desired and what you expect instead.


### Template

**Report installed package versions**
- eemeter: vX.X.X
- pandas: vX.X.X
- scipy: vX.X.X
- numpy: vX.X.X

**Describe the bug**
A clear and concise description of what the bug is, including code samples and
tracebacks.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Additional context**
Add any other context about the problem here.
