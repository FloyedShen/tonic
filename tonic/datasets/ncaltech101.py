import os
import numpy as np

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive


class NCALTECH101(Dataset):
    """N-CALTECH101 dataset <https://www.garrickorchard.com/datasets/n-caltech101>. Events have (xytp) ordering.
    ::

        @article{orchard2015converting,
          title={Converting static image datasets to spiking neuromorphic datasets using saccades},
          author={Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K and Thakor, Nitish},
          journal={Frontiers in neuroscience},
          volume={9},
          pages={437},
          year={2015},
          publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    url = "https://www.dropbox.com/sh/iuv7o3h2gv6g4vd/AADYPdhIBK7g_fPCLKmG6aVpa?dl=1"
    filename = "N-Caltech101-archive.zip"
    file_md5 = "989af2c704103341d616b748b5daa70c"
    data_filename = "Caltech101.zip"
    folder_name = "Caltech101"
    cls_count = [435, 200, 798, 55, 800, 42, 42, 47, 54, 46,
                 33, 128, 98, 43, 85, 91, 50, 43, 123, 47,
                 59, 62, 107, 47, 69, 73, 70, 50, 51, 57,
                 67, 52, 65, 68, 75, 64, 53, 64, 85, 67,
                 67, 45, 34, 34, 51, 99, 100, 42, 54, 88,
                 80, 31, 64, 86, 114, 61, 81, 78, 41, 66,
                 43, 40, 87, 32, 76, 55, 35, 39, 47, 38,
                 45, 53, 34, 57, 82, 59, 49, 40, 63, 39,
                 84, 57, 35, 64, 45, 86, 59, 64, 35, 85,
                 49, 86, 75, 239, 37, 59, 34, 56, 39, 60]
    length = 8242

    sensor_size = None  # all recordings are of different size
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None):
        super(NCALTECH101, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        classes = {
            'Faces_easy': 0,
            'Leopards': 1,
            'Motorbikes': 2,
            'accordion': 3,
            'airplanes': 4,
            'anchor': 5,
            'ant': 6,
            'barrel': 7,
            'bass': 8,
            'beaver': 9,
            'binocular': 10,
            'bonsai': 11,
            'brain': 12,
            'brontosaurus': 13,
            'buddha': 14,
            'butterfly': 15,
            'camera': 16,
            'cannon': 17,
            'car_side': 18,
            'ceiling_fan': 19,
            'cellphone': 20,
            'chair': 21,
            'chandelier': 22,
            'cougar_body': 23,
            'cougar_face': 24,
            'crab': 25,
            'crayfish': 26,
            'crocodile': 27,
            'crocodile_head': 28,
            'cup': 29,
            'dalmatian': 30,
            'dollar_bill': 31,
            'dolphin': 32,
            'dragonfly': 33,
            'electric_guitar': 34,
            'elephant': 35,
            'emu': 36,
            'euphonium': 37,
            'ewer': 38,
            'ferry': 39,
            'flamingo': 40,
            'flamingo_head': 41,
            'garfield': 42,
            'gerenuk': 43,
            'gramophone': 44,
            'grand_piano': 45,
            'hawksbill': 46,
            'headphone': 47,
            'hedgehog': 48,
            'helicopter': 49,
            'ibis': 50,
            'inline_skate': 51,
            'joshua_tree': 52,
            'kangaroo': 53,
            'ketch': 54,
            'lamp': 55,
            'laptop': 56,
            'llama': 57,
            'lobster': 58,
            'lotus': 59,
            'mandolin': 60,
            'mayfly': 61,
            'menorah': 62,
            'metronome': 63,
            'minaret': 64,
            'nautilus': 65,
            'octopus': 66,
            'okapi': 67,
            'pagoda': 68,
            'panda': 69,
            'pigeon': 70,
            'pizza': 71,
            'platypus': 72,
            'pyramid': 73,
            'revolver': 74,
            'rhino': 75,
            'rooster': 76,
            'saxophone': 77,
            'schooner': 78,
            'scissors': 79,
            'scorpion': 80,
            'sea_horse': 81,
            'snoopy': 82,
            'soccer_ball': 83,
            'stapler': 84,
            'starfish': 85,
            'stegosaurus': 86,
            'stop_sign': 87,
            'strawberry': 88,
            'sunflower': 89,
            'tick': 90,
            'trilobite': 91,
            'umbrella': 92,
            'watch': 93,
            'water_lilly': 94,
            'wheelchair': 95,
            'wild_cat': 96,
            'windsor_chair': 97,
            'wrench': 98,
            'yin_yang': 99,
        }

        # if not self._check_exists():
            # self.download()
            # extract_archive(os.path.join(self.location_on_system, self.data_filename))

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            if 'BACKGROUND_Google' in path:
                continue
            for file in files:
                if file.endswith("bin"):
                    self.data.append(path + "/" + file)
                    label_name = os.path.basename(path)

                    if isinstance(label_name, bytes):
                        label_name = label_name.decode()
                    self.targets.append(classes[label_name])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = read_mnist_file(self.data[index], dtype=self.dtype)
        target = self.targets[index]
        events["x"] -= events["x"].min()
        events["y"] -= events["y"].min()
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            8709, ".bin"
        )
