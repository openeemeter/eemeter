import platform
import importlib
import time
import numpy as np


def get_arch_type():
    """
    Get the machine's processor architecture and return the appropriate architecture type.
    """
    arch = platform.machine()
    if arch in ["x86_64","aarch64"]:
        return arch
    else:
        return "other"


# Define a function to initialize the full_model function
def initialize_full_model():

    # Get the machine's processor architecture to use in identifying the binary to import
    arch_type = get_arch_type()

    current_directory = "eemeter.caltrack.daily.base_models"

    # Import the Python version of full_model
    full_model_python = importlib.import_module(
        current_directory + ".full_model"
    ).full_model

    # Import the Cython version of full_model if present and performs faster, else default to the Python version
    try:
        full_model_cpp = importlib.import_module(
            current_directory + ".cython.bin." + arch_type + ".full_model_ext"
        ).full_model_wrapper

        loop_count = 0
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
        full_model_python(
            hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T
        )

        for i in range(10):
            start = time.process_time()

            for i in range(loop_count):
                full_model_cpp(
                    hdd_bp,
                    hdd_beta,
                    hdd_k,
                    cdd_bp,
                    cdd_beta,
                    cdd_k,
                    intercept,
                    T_fit_bnds,
                    T,
                )

            cpp_time = time.process_time() - start
            cpp_times.append(cpp_time)

            start = time.process_time()

            for i in range(loop_count):
                full_model_python(
                    hdd_bp,
                    hdd_beta,
                    hdd_k,
                    cdd_bp,
                    cdd_beta,
                    cdd_k,
                    intercept,
                    T_fit_bnds,
                    T,
                )

            py_time = time.process_time() - start
            py_times.append(py_time)

        cpp_mean = np.mean(cpp_times)
        py_mean = np.mean(py_times)

        if cpp_mean < py_mean:
            full_model = full_model_cpp
        else:
            full_model = full_model_python

    except:
        full_model = full_model_python
    
    return full_model
        


# Call the initialize_full_model function to initialize the full_model function
full_model = initialize_full_model()
