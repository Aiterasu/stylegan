import tensorflow as tf
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from functools import partial
import io
import base64
import copy
import pydub
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('audio', help="The filename of the audio file.")
parser.add_argument('stylegan', default="2019-04-30-stylegan-danbooru2018-portraits-02095-066083.pkl", help="The filename of the StyleGAN pkl file.")
parser.add_argument('output', help="Set output filename of ffmpeg")
parser.add_argument('--psi', default=0.7, help="The truncation psi.")
parser.add_argument('--truncate_pre', default=True, help="Truncate before dlatent modification.")
parser.add_argument('--truncate_post', default=True, help="Truncate after dlatent modification.")
parser.add_argument('--output_dir', help="Directory to output images to.")
parser.add_argument('--randomize_noise', default=False, help="Whether to randomize the noise every step.")
parser.add_argument('--seed', help="Random seed.")

args, other_args = parser.parse_known_args()

img_dir = os.path.splitext(os.path.basename(args.output))[0]
psi = float(args.psi)

if args.seed:
    seed = int(args.seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

tflib.init_tf()

# Load pre-trained network.
with open(args.stylegan, 'rb') as f:
    _G, _D, Gs = pickle.load(f)

##
# Build things on top for encoding
# Based on https://github.com/Puzer/stylegan
##
def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

dlatent_avg = tf.get_default_session().run(Gs.own_vars["dlatent_avg"])
def create_variable_for_generator(name, batch_size):
    truncation_psi_encode = psi
    layer_idx = np.arange(16)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = tf.where(layer_idx < 8, truncation_psi_encode * ones, ones)
    dlatent_variable = tf.get_variable(
        'learnable_dlatents', 
        shape=(1, 16, 512), 
        dtype='float32', 
        initializer=tf.initializers.zeros()
    )
    dlatent_variable_trunc = tflib.lerp(dlatent_avg, dlatent_variable, coefs)
    return dlatent_variable_trunc

# Generation-from-disentangled-latents part
initial_dlatents = np.zeros((1, 16, 512))
Gs.components.synthesis.run(
    initial_dlatents,
    randomize_noise = True,
    minibatch_size = 1,
    custom_inputs = [
        partial(create_variable_for_generator, batch_size=1),
        partial(create_stub, batch_size = 1)],
    structure = 'fixed'
)

# We have to do truncation ourselves, since we're not using the combined network
def truncate(dlatents, truncation_psi, maxlayer = 8):
    dlatent_avg = tf.get_default_session().run(Gs.own_vars["dlatent_avg"])
    layer_idx = np.arange(16)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = tf.where(layer_idx < maxlayer, truncation_psi * ones, ones)
    return tf.get_default_session().run(tflib.lerp(dlatent_avg, dlatents, coefs))

mod_latents = np.load("mod_latents.npy")
dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0] 

# Load up tags and tag directions
with open("tag_dirs.pkl", 'rb') as f:
    tag_directions = pickle.load(f)

mod_latents = np.random.randn(1, Gs.input_shape[1])
dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0] 

def generate_image(tags):
    if args.truncate_pre:
        dlatents_mod = truncate(copy.deepcopy(dlatents_gen), psi)
    else:
        dlatents_mod = copy.deepcopy(dlatents_gen)

    for tag, value in tags.items():
        dlatents_mod += tag_directions[tag] * value

    if args.truncate_post:
        dlatents_trunc = truncate(dlatents_mod, psi)
    else:
        dlatents_trunc = dlatents_mod
        
    # Run the network
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    result_image = Gs.components.synthesis.run(
        dlatents_trunc.reshape((-1, 16, 512)),
        randomize_noise = args.randomize_noise,
        minibatch_size = 1,
        output_transform=fmt
    )[0]

    return PIL.Image.fromarray(result_image, 'RGB')

song = pydub.AudioSegment.from_mp3(args.audio)
amp_segments = []

index = 0

while index * 20 < len(song):
    segment = song[index * 20:(index + 1) * 20]
    amp_segments.append(segment.rms)
    index += 1

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

ma = moving_average(amp_segments)
# Scale from to (0, max_amplitude) to (0,2)
ma = (2 * ma) / max(ma)

noise_layers = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

for seg_idx, segment in enumerate(ma):

    for i in range(6, 7):
        x = noise_layers[i]
        noise = np.random.rand(x.shape[0], 1, x.shape[2], x.shape[3])
        x.load(noise)


    img = generate_image({"open_mouth": segment, "closed_mouth": -segment})
    
    os.makedirs(img_dir, exist_ok=True)

    # Leading frame (moving average chops off leading and end)
    if seg_idx == 0:
        filename = os.path.join(img_dir, 'frame-'+str(seg_idx)+'.jpg')
        img.save(filename, "JPEG")

    filename = os.path.join(img_dir, 'frame-'+str(seg_idx+1)+'.jpg')
    img.save(filename, "JPEG")

# End frame
filename = os.path.join(img_dir, 'frame-'+str(seg_idx+2)+'.jpg')
img.save(filename, "JPEG")

rc = subprocess.call('ffmpeg -r 50 -f image2 -i %s/frame-%%d.jpg -i %s -vcodec libx264 -acodec aac %s' % (img_dir, args.audio, args.output), shell=True)