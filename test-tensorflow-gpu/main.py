import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

print('Available GPUs:', physical_devices)

# 剩下的代码和你之前写的一样,,,,
# cmd.exe "F:\miniconda3\Scripts\activate.bat",
# cmd.exe "C:\ProgramData\miniconda3\Scripts\activate.bat"
#C:\ProgramData\miniconda3\condabin


# 檢查是否有可用的 GPU
if not tf.test.is_gpu_available():
    print('沒有可用的 GPU')
else:
    print('有可用的 GPU')

# 檢查 CUDA 是否可用
if not tf.test.is_built_with_cuda():
    print('TensorFlow 沒有使用 CUDA')
else:
    print('TensorFlow 使用了 CUDA')



print('Tensorflow version : ' ,tf.__version__)
print('Tensorflow GPU : ' ,tf.test.is_gpu_available())

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print('GPU List : ',gpus)
print('CPU List : ',cpus)