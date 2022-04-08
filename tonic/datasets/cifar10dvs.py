import os
import numpy as np

from tonic.io import read_aedat4
from tonic.io import read_aedat
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive
from tonic.io import make_structured_array

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event

dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])


def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def load_raw_events(fp,
                    bytes_skip=0,
                    bytes_trim=0,
                    filter_dvs=False,
                    times_first=False):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.fromstring(data, dtype='>u4')
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('---')
        print(data[1:21:2])
        raise ValueError('odd number of data elements')
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    return timestamp, raw_addr


def parse_raw_address(addr,
                      x_mask=x_mask,
                      x_shift=x_shift,
                      y_mask=y_mask,
                      y_shift=y_shift,
                      polarity_mask=polarity_mask,
                      polarity_shift=polarity_shift):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def load_events(
    fp,
    filter_dvs=False,
    # bytes_skip=0,
    # bytes_trim=0,
    # times_first=False,
        **kwargs):
    timestamp, addr = load_raw_events(
        fp,
        filter_dvs=filter_dvs,
        #   bytes_skip=bytes_skip,
        #   bytes_trim=bytes_trim,
        #   times_first=times_first
    )
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity


def load_origin_data(file_name: str):
    '''
    :param file_name: path of the events file
    :type file_name: str
    :return: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
    :rtype: Dict

    This function defines how to read the origin binary data.
    '''
    with open(file_name, 'rb') as fp:
        t, x, y, p = load_events(fp,
                                 x_mask=0xfE,
                                 x_shift=1,
                                 y_mask=0x7f00,
                                 y_shift=8,
                                 polarity_mask=1,
                                 polarity_shift=None)
        # return {'t': t, 'x': 127 - x, 'y': y, 'p': 1 - p.astype(int)}  # this will get the same data with http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/dat2mat.m
        # see https://github.com/jackd/events-tfds/pull/1 for more details about this problem
        # return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}
        return make_structured_array(t, 127 - y, 127 - x, 1 - p.astype(int), dtype)


class CIFAR10DVS(Dataset):
    """Li, H., Liu, H., Ji, X., Li, G., & Shi, L. (2017). Cifar10-dvs: an event-stream dataset for object
    classification. Frontiers in neuroscience, 11, 309. ::

        @article{li2017cifar10,
        title={Cifar10-dvs: an event-stream dataset for object classification},
        author={Li, Hongmin and Liu, Hanchao and Ji, Xiangyang and Li, Guoqi and Shi, Luping},
        journal={Frontiers in neuroscience},
        volume={11},
        pages={309},
        year={2017},
        publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    url = "http://cifar10dvs.ridger.top/CIFAR10DVS.zip"

    filename = "CIFAR10DVS.zip"
    file_md5 = "ce3a4a0682dc0943703bd8f749a7701c"
    data_filename = [
        "airplane.zip",
        "automobile.zip",
        "bird.zip",
        "cat.zip",
        "deer.zip",
        "dog.zip",
        "frog.zip",
        "horse.zip",
        "ship.zip",
        "truck.zip",
    ]

    folder_name = "CIFAR10DVS"

    sensor_size = (128, 128, 2)

    def __init__(self, save_to, transform=None, target_transform=None):
        super(CIFAR10DVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        # classes for CIFAR10DVS dataset

        classes = {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }

        # if not self._check_exists():
        #     self.download()
        #     for filename in self.data_filename:
        #         extract_archive(os.path.join(
        #             self.location_on_system, filename))

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("aedat"):
                    self.data.append(path + "/" + file)
                    label_number = classes[os.path.basename(path)]
                    self.targets.append(label_number)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = load_origin_data(self.data[index])
        # for correctly reading the data
        # events.dtype.names = ["t", "x", "y", "p"]
        target = self.targets[index]

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            1000, ".aedat"
        )
