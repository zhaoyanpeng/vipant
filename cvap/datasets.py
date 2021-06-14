import glob
import torch
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from itertools import cycle, islice, chain
from einops import rearrange, repeat

import torch.nn.functional as F


class PairImageSpectrogramTFRecords(object):
    def __init__(
        self,
        local_or_gcs_path,
        batch_size,
        resolution=224,
        prefetch_size=0,
        mel_bins=128,
        max_audio_len=2048,
        input_resolution=224,
    ):
        self.mel_bins = mel_bins
        self.max_audio_len = max_audio_len
        self.path = local_or_gcs_path
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.max_audio_len = max_audio_len
        self.resolution = resolution

    def files(self):
        return self.files

    def __iter__(self):
        files = tf.data.TFRecordDataset.list_files(
            self.path + "/*.tfrecord", shuffle=False
        )
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(self.deserialize_tf_record)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes={
                "name": (),
                "audio": (None, self.mel_bins),
                "image": (None, None, None),
            },
        )
        dataset = dataset.map(self.unsqueeze_trailing)
        dataset = dataset.prefetch(self.prefetch_size)
        dataset = dataset.as_numpy_iterator()

        return dataset

    def deserialize_tf_record(self, record):
        tfrecord_format = {
            "name": tf.io.FixedLenFeature(
                (), dtype=tf.string
            ),
            "audio": tf.io.FixedLenSequenceFeature(
                (self.mel_bins,), dtype=tf.float32, allow_missing=True
            ),
            "image": tf.io.FixedLenSequenceFeature(
                (self.resolution, self.resolution), dtype=tf.float32, allow_missing=True
            ),
        }

        features_tensor = tf.io.parse_single_example(record, tfrecord_format)
        return features_tensor

    def unsqueeze_trailing(self, record):
        record = {
            "name": record["name"],
            "audio": record["audio"],
            "image": record["image"],
        }
        return record

    @staticmethod
    def write(spectrograms, images, image_names, fname="default.tfrecord"):
        tfrecord_writer = tf.io.TFRecordWriter(fname)
        for (spectrogram, image, name) in tqdm(zip(spectrograms, images, image_names)):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "name": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[name.encode("utf8")])
                        ),
                        "audio": tf.train.Feature(
                            float_list=tf.train.FloatList(value=spectrogram.flatten())
                        ),
                        "image": tf.train.Feature(
                            float_list=tf.train.FloatList(value=image.flatten())
                        ),
                    }
                )
            )
            tfrecord_writer.write(example.SerializeToString())

        tfrecord_writer.close()


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))
