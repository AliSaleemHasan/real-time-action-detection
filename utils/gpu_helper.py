import subprocess
import tensorflow as tf
import logging
import sys

# Configure logging for GPU helper
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - [GPU Helper] - %(levelname)s - %(message)s')

def validate_gpu_subprocess():
    """
    Runs a tiny isolated Python process to test the GPU.
    Returns True if the process succeeds, False if it crashes/fails.
    """
    code = """
import tensorflow as tf
import os
import sys

# Suppress TF logs in subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        sys.exit(1) # No GPU found

    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    with tf.device('/GPU:0'):
        # The 'Sacrificial' Check: Triggers cuDNN 5003 error here if lib is broken
        x = tf.random.normal([1, 10, 10, 3])
        valid_kernel = tf.random.normal([3, 3, 3, 1])
        _ = tf.nn.conv2d(x, valid_kernel, strides=[1, 1, 1, 1], padding='SAME')
    
    sys.exit(0) # Success
except Exception:
    sys.exit(1) # Failure
"""
    try:
        # Run the validation in a separate process
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=10 # Short timeout to prevent hanging
        )
        return result.returncode == 0
    except Exception as e:
        logging.warning(f"Subprocess validation check failed: {e}")
        return False

def configure_gpu():
    """
    Abstracts GPU configuration using a subprocess check to avoid 
    poisoning the main process runtime.
    """
    try:
        # 1. Check if GPUs *physically* exist (lightweight check)
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if len(physical_devices) > 0:
            logging.info(f"Detected {len(physical_devices)} Physical GPUs.")
            
            # 2. Context-Aware Validation
            # Only run the strict convolution check for scripts that heavily use CNNs (preprocessing, detect)
            # Training (LSTM) often works fine even if Conv2D is broken.
            script_name = sys.argv[0]
            needs_strict_check = any(x in script_name for x in ['preprocessing.py', 'detect.py'])
            
            is_gpu_healthy = True
            if needs_strict_check:
                logging.info(f"Running strict GPU convolution check for {os.path.basename(script_name)}...")
                is_gpu_healthy = validate_gpu_subprocess()
            else:
                logging.info(f"Skipping strict convolution check for {os.path.basename(script_name)} (assuming LSTM-only usage).")

            if is_gpu_healthy:
                if needs_strict_check:
                    logging.info("GPU Verified (Subprocess) and Ready for Use.")
                
                # Safe to initialize in main process now
                for device in physical_devices:
                    try:
                        tf.config.experimental.set_memory_growth(device, True)
                    except:
                        pass
                return True
            else:
                logging.warning("GPU failed validation check (likely cuDNN 5003 error). Hiding GPU before initialization.")
                try:
                    # BLOCK the GPU *before* it ruins the main runtime
                    tf.config.set_visible_devices([], 'GPU')
                except Exception as e:
                    logging.error(f"Failed to hide GPU: {e}")
                return False

        else:
            logging.info("No Physical GPUs detected. Using CPU.")
            return False

    except Exception as e:
        logging.error(f"Critical error during GPU configuration: {e}. Defaulting to CPU.")
        try:
            tf.config.set_visible_devices([], 'GPU')
        except:
            pass
        return False
