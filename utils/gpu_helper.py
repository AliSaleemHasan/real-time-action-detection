import tensorflow as tf
import logging
import sys

# Configure logging for GPU helper
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - [GPU Helper] - %(levelname)s - %(message)s')

def configure_gpu():
    """
    Abstracts GPU configuration for the entire project.
    
    Usage:
        from utils.gpu_helper import configure_gpu
        configure_gpu()
        
    Mechanism:
    1. Detects physical GPUs.
    2. Sets memory growth (prevents hoarding all VRAM).
    3. Validates GPU usability (catches missing library errors like libcudnn).
    4. Automatically falls back to CPU if GPU is unusable.
    
    Returns:
        bool: True if GPU is active and usable, False if using CPU.
    """
    try:
        # 1. Attempt to list physical GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if len(physical_devices) > 0:
            logging.info(f"Detected {len(physical_devices)} Physical GPUs: {physical_devices}")
            
            # 2. Configure Memory Growth
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    # Invalid device or cannot modify virtual devices once initialized.
                    pass
            
            # 3. Validation Check: Try a dummy computation
            # This triggers library loading (libcudnn.so) and catches failures early.
            try:
                with tf.device('/GPU:0'):
                    _ = tf.random.normal((1, 1))
                logging.info("GPU Verified and Ready for Use.")
                return True
            except Exception as e:
                logging.warning(f"GPU detected but unusable (Library Error: {e}). Falling back to CPU.")
                
                # Hide GPU to force CPU usage
                try:
                    tf.config.set_visible_devices([], 'GPU')
                except Exception as hide_error:
                    logging.error(f"Failed to hide GPU: {hide_error}")
                    
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
