import os
import h5py
import numpy as np

def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """FUNCTION TO WRITE DATASET TO HDF5

    Args :
        hdf5_name (str): hdf5 dataset filename
        hdf5_path (str): dataset path in hdf5
        write_data (ndarray): data to write
        is_overwrite (bool): flag to decide whether to overwrite dataset
    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warn("dataset in hdf5 file already exists.")
                logging.warn("recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error("dataset in hdf5 file already exists.")
                logging.error("if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()

if __name__ == '__main__':

  scp_file = '/home/yangshan/tts_workspace/tacotron-IS2018/wavenet_vocoder/egs/bc2011/sd-melspc/data'
  test_path='test-tacotron1-vq-r2-usar'
  if not os.path.exists(os.path.join(scp_file, test_path)):
    os.mkdir(os.path.join(scp_file, test_path))
  scp_file = os.path.join(scp_file,test_path+'/feats.scp')
  fid = open(scp_file, 'w')
  for line in os.listdir(os.path.join('hdf5', test_path)):
    py_path = os.path.join(test_path, line.strip())
    data = np.load(py_path)
    hdf5_path = os.path.join(test_path, os.path.splitext(line.strip())[0]+'.h5')
    write_hdf5(hdf5_path, "/melspc", data)
    fid.write('hdf5/'+hdf5_path+'\n')
