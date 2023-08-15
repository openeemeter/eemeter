import platform
import importlib
import time
import numpy as np

# Define a global variable to store the full_model function
full_model = None

# Define a function to initialize the full_model function
def initialize_full_model():
    global full_model

    # Get the machine's processor architecture
    arch = platform.machine()

    if arch == 'x86_64':
        arch_type = "x86"
    elif arch == 'aarch64':
        arch_type = "arm"
    else:
        arch_type = "other"
    # Import the Python version of full_model
    full_model_python = importlib.import_module("eemeter.caltrack.daily.base_models.full_model").full_model

    # Import the Cython version of full_model if present and performs faster, else default to the Python version
    try:
        full_model_cpp = importlib.import_module("bin." + arch_type + ".full_model_ext").full_model_wrapper

        loop_count = 100
        hdd_bp = 50
        hdd_beta = 0.01
        hdd_k = 0.001
        cdd_bp = 80
        cdd_beta = 0.02
        cdd_k = 0.002
        intercept = 100
        T_fit_bnds = np.array([-10, 100]).astype(np.double)
        T = np.linspace(-10, 100, 365).astype(np.double)

        cpp_times = []
        py_times = []

        cpp_times = []
        py_times = []

        # Compile Numba for fair comparison
        full_model_python(hdd_bp,hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T)

        for i in range(10):
            start = time.process_time()

            for i in range(loop_count):
                full_model_cpp(hdd_bp,hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T)

            cpp_time = time.process_time() - start
            cpp_times.append(cpp_time)

            start = time.process_time()

            for i in range(loop_count):
                full_model_python(hdd_bp,hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T)

            py_time = time.process_time() - start
            py_times.append(py_time)

        cpp_mean = np.mean(cpp_times)
        py_mean = np.mean(py_times)

        if cpp_mean < py_mean:
            full_model = full_model_cpp
        else:
            full_model = full_model_python

    except ImportError:
        full_model = full_model_python

# Call the initialize_full_model function to initialize the full_model function
initialize_full_model()
