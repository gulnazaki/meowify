# DISCLAIMER: Code from the ddsp_timbre_transfer colab notebook demo

# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.filterwarnings("ignore")

import copy
import os
import time

import crepe
import ddsp
import ddsp.training
from ddsp.colab import colab_utils
from ddsp.colab.colab_utils import (
	auto_tune, detect_notes, fit_quantile_transform, 
	get_tuning_factor, download, play, record, 
	specplot, upload, DEFAULT_SAMPLE_RATE)
import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from scipy.io import wavfile
from pydub import AudioSegment
import tempfile

def write_to_file(audio_file, model_dir, output, sample_rate = DEFAULT_SAMPLE_RATE):
	audio_float = audio_file_to_np(audio_file)
	cat_audio_float = tranfer(audio_float, model_dir, sample_rate=sample_rate)
	if len(cat_audio_float.shape) == 2:
		cat_audio_float = cat_audio_float[0]

	normalizer = float(np.iinfo(np.int16).max)
	cat_audio_int = np.array(
		np.asarray(cat_audio_float) * normalizer, dtype=np.int16)
	wavfile.write(output, sample_rate, cat_audio_int)

def audio_file_to_np(audio_file, sample_rate=DEFAULT_SAMPLE_RATE, normalize_db=0.1):
	audio = AudioSegment.from_file(audio_file)
	audio.remove_dc_offset()
	if normalize_db is not None:
		audio.normalize(headroom=normalize_db)
	# Save to tempfile and load with librosa.
	with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav_file:
		fname = temp_wav_file.name
		audio.export(fname, format='wav')
		audio_np, unused_sr = librosa.load(fname, sr=sample_rate)
	return audio_np.astype(np.float32)

def tranfer(audio, model_dir, sample_rate = DEFAULT_SAMPLE_RATE):
	audio = audio[np.newaxis, :]

	ddsp.spectral_ops.reset_crepe()

	audio_features = ddsp.training.metrics.compute_audio_features(audio)
	audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
	audio_features_mod = None

	gin_file = os.path.join(model_dir, 'operative_config-0.gin')

	# Load the dataset statistics.
	DATASET_STATS = None
	dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
	try:
	 	if tf.io.gfile.exists(dataset_stats_file):
	 		with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
	 			DATASET_STATS = pickle.load(f)
	except Exception as err:
		print('Loading dataset statistics from pickle failed: {}.'.format(err))

	# Parse gin config,
	with gin.unlock_config():
		gin.parse_config_file(gin_file, skip_unknown=True)

	# Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
	ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
	ckpt_name = ckpt_files[0].split('.')[0]
	ckpt = os.path.join(model_dir, ckpt_name)

	# Ensure dimensions and sampling rates are equal
	time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
	n_samples_train = gin.query_parameter('Additive.n_samples')
	hop_size = int(n_samples_train / time_steps_train)

	time_steps = int(audio.shape[1] / hop_size)
	n_samples = time_steps * hop_size


	gin_params = [
		'Additive.n_samples = {}'.format(n_samples),
		'FilteredNoise.n_samples = {}'.format(n_samples),
		'DefaultPreprocessor.time_steps = {}'.format(time_steps),
		'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
	]

	with gin.unlock_config():
		gin.parse_config(gin_params)


	# Trim all input vectors to correct lengths 
	for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
		audio_features[key] = audio_features[key][:time_steps]
	audio_features['audio'] = audio_features['audio'][:, :n_samples]

	# Set up the model just to predict audio given new conditioning
	model = ddsp.training.models.Autoencoder()
	model.restore(ckpt)

	# Build model by running a batch through it.
	start_time = time.time()
	_ = model(audio_features, training=False)

	threshold = 1
	ADJUST = True
	quiet = 20
	autotune = 0
	pitch_shift =  -1
	loudness_shift = 3

	audio_features_mod = {k: v.copy() for k, v in audio_features.items()}


	## Helper functions.
	def shift_ld(audio_features, ld_shift=0.0):
		"""Shift loudness by a number of ocatves."""
		audio_features['loudness_db'] += ld_shift
		return audio_features


	def shift_f0(audio_features, pitch_shift=0.0):
		"""Shift f0 by a number of ocatves."""
		audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
		audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], 
										0.0, 
										librosa.midi_to_hz(110.0))
		return audio_features


	mask_on = None

	if ADJUST and DATASET_STATS is not None:
	  # Detect sections that are "on".
		mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
											audio_features['f0_confidence'],
											threshold)

		if np.any(mask_on):
			# Shift the pitch register.
			target_mean_pitch = DATASET_STATS['mean_pitch']
			pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
			mean_pitch = np.mean(pitch[mask_on])
			p_diff = target_mean_pitch - mean_pitch
			p_diff_octave = p_diff / 12.0
			round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
			p_diff_octave = round_fn(p_diff_octave)
			audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)


			# Quantile shift the note_on parts.
			_, loudness_norm = colab_utils.fit_quantile_transform(
				audio_features['loudness_db'],
				mask_on,
				inv_quantile=DATASET_STATS['quantile_transform'])

			# Turn down the note_off parts.
			mask_off = np.logical_not(mask_on)
			loudness_norm[mask_off] -=  quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
			loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)

			audio_features_mod['loudness_db'] = loudness_norm 

			# Auto-tune.
			if autotune:
				f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
				tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
				f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
				audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)

		else:
			print('\nSkipping auto-adjust (no notes detected or ADJUST box empty).')

	else:
		print('\nSkipping auto-adujst (box not checked or no dataset statistics found).')

	# Manual Shifts.
	audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
	audio_features_mod = shift_f0(audio_features_mod, pitch_shift)

	af = audio_features if audio_features_mod is None else audio_features_mod

	return model(af, training=False)
