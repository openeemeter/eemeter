# -*- coding: utf-8 -*-
"""
eemeter library usage
~~~~~~~~~~~~~~~~~~~~~
The eemeter libary implements core Open Energy Efficiency metering methods.
Basic usage:
   >>> import eemeter
Full documentation is at <https://openee.io>.
:copyright: (c) 2018 by Open Energy Efficiency.
:license: Apache 2.0, see LICENSE for more details.
"""

import logging

from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __author__, __author_email__, __license__
from .__version__ import __copyright__
from .caltrack import *
from .derivatives import *
from .exceptions import *
from .features import *
from .metrics import *
from .warnings import *
from .transform import *
from .io import *
from .visualization import *
from .samples.load import *
from .segmentation import *


def get_version():
    return __version__


# Set default logging handler to avoid "No handler found" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())
