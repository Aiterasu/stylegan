from bottle import route, run, request, static_file, response, hook
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
import os
import argparse
import pydub
import subprocess
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('stylegan', default="2019-04-30-stylegan-danbooru2018-portraits-02095-066083.pkl", help="The filename of the StyleGAN pkl file.")
parser.add_argument('--seed', help="Starting seed.")
parser.add_argument('--psi', default=0.7, help="Starting psi")
parser.add_argument('--truncate_pre', default=True, help="Truncate before dlatent modification.")
parser.add_argument('--truncate_post', default=True, help="Truncate after dlatent modification.")
parser.add_argument('--randomize_noise', default=False, help="Whether to randomize the noise every step.")
parser.add_argument('--start_closed_stength', default=0.2, help="Strength of starting latent for closed_mouth.")
parser.add_argument('--normalize_audio_to', default=2, help="Max, normalized strength of audio amplitude.")

args, other_args = parser.parse_known_args()

if args.seed:
    seed = int(args.seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

psi = float(args.psi)
truncate_post = args.truncate_post
truncate_pre = args.truncate_pre
randomize_noise = args.randomize_noise
normalize_audio_to = args.normalize_audio_to
start_closed_stength = args.start_closed_stength

tflib.init_tf()

# Load pre-trained network.
with open(args.stylegan, 'rb') as f:
    _G, _D, Gs = pickle.load(f)

# noise_layers = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]


##
# Build things on top for encoding
# Based on https://github.com/Puzer/stylegan
##
def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

dlatent_avg = tf.get_default_session().run(Gs.own_vars["dlatent_avg"])
def create_variable_for_generator(name, batch_size):
    truncation_psi_encode = 0.7
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

# Load up tags and tag directions
with open("tag_dirs.pkl", 'rb') as f:
    tag_directions = pickle.load(f)

z = np.random.randn(1, Gs.input_shape[1])
z = Gs.components.mapping.run(z, None)[0] 

def generate_image(e, tags):
    if truncate_pre:
        e = truncate(e, psi)

    for tag, value in tags.items():
        e += tag_directions[tag] * value

    if truncate_post:
        e = truncate(e, psi)
        
    # Run the network
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    result_image = Gs.components.synthesis.run(
        e.reshape((-1, 16, 512)),
        randomize_noise = randomize_noise,
        minibatch_size = 1,
        output_transform=fmt
    )[0]

    return PIL.Image.fromarray(result_image, 'RGB')

@route('/gen', method='GET')
def gen():
    tags = dict()
    for tag, value in request.query.items():
        tags[tag] = float(value)

    # TODO: Is this needed
    e = copy.deepcopy(z)
    img = generate_image(e, tags)

    buffer = io.BytesIO()
    img.save(buffer, 'JPEG')
    buffer.seek(0)

    response.set_header('Content-Type', "image/jpeg")
    return buffer

@route('/set', method='POST')
def do_settings():
    # TODO: There's a better way to store state between calls. Probably.
    global z
    global psi
    global truncate_pre
    global truncate_post
    global normalize_audio_to
    global start_closed_stength

    # TODO: Better way. or?
    if request.forms.get('psi'):
        psi = float(request.forms.get('psi'))

    if request.forms.get('normalize_audio_to'):
        normalize_audio_to = float(request.forms.get('normalize_audio_to'))

    if request.forms.get('start_closed_stength'):
        start_closed_stength = float(request.forms.get('start_closed_stength'))

    if request.forms.get('truncate_pre'):
        truncate_pre = bool(request.forms.get('truncate_pre'))

    if request.forms.get('truncate_post'):
        truncate_post = bool(request.forms.get('truncate_post'))

    seed = request.forms.get('seed')
    if seed:
        seed = int(seed)
        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        z = np.random.randn(1, Gs.input_shape[1])
        z = Gs.components.mapping.run(z, None)[0] 

output_filename = "output.mp4"
mp3_name = "in.mp3"

@route('/animate_mouth', method='POST')
def animate_mouth():
    img_dir = "output"

    # Yuck, do it in memory if possible.
    with open(mp3_name, 'wb') as f:
        f.write(request.body.read())

    song = pydub.AudioSegment.from_mp3(mp3_name)
    amp_segments = []

    # TODO: Not exactly 42 ms per frame at 24 fps, fixy
    ms_per_frame = 42
    for i in range(0, len(song), ms_per_frame):
        segment = song[i:i + ms_per_frame]
        amp_segments.append(segment.rms)

    amp_segments = np.array(amp_segments)
    amp_segments = (normalize_audio_to * amp_segments) / max(amp_segments)

    os.makedirs(img_dir, exist_ok=True)

    e = copy.deepcopy(z)
    if truncate_pre:
        e = truncate(e, psi)

    e += tag_directions["closed_mouth"] * start_closed_stength

    if truncate_post:
        e = truncate(e, psi)

    for seg_idx, segment in enumerate(amp_segments):
        img = generate_image(e, {"open_mouth": segment})
        
        filename = os.path.join(img_dir, 'frame-'+str(seg_idx)+'.jpg')
        img.save(filename, "JPEG")

    rc = subprocess.call('ffmpeg -r 24 -f image2 -i %s/frame-%%d.jpg -i %s -vcodec libx264 -acodec aac %s' % (img_dir, mp3_name, output_filename), shell=True)

    shutil.rmtree(img_dir)

    return static_file(output_filename, root=os.getcwd(), download=output_filename)
    
# Couldn't figure out how to only do this after animate_mouth
@hook('after_request')
def remove_mp4():
    if os.path.exists(output_filename):
        os.remove(output_filename)
    if os.path.exists(mp3_name):
        os.remove(mp3_name)

run(host='localhost', port=8081)
