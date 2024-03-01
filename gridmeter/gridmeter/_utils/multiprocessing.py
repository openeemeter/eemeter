#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import multiprocessing as mp


def _execute_with_mp(fcn, args_list, use_mp=True):
    """Runs a function with multiprocessing if use_mp is True, otherwise runs
    the function without multiprocessing.

    Args:
        fcn (function): The function to run.
        args_list (iterable): The list of arguments to pass to the function.
        use_mp (bool): Whether to use multiprocessing.

    Returns:
        The result of the function.
    """
    
    if use_mp:
        with mp.Pool(processes=mp.cpu_count()) as mp_pool:
            result = mp_pool.map(fcn, args_list)

    else:
        result = []
        for args in args_list:
            result.append(fcn(args))

    return result
