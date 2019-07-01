Script to animate the mouth position of StyleGAN output from an audio file.

## Install
`ffmpeg` and NVIDIA CUDA drivers must be installed.

*Prepare Python Environment*
```bash
conda create -n animate pip python=3.7
conda activate animate
pip install tensorflow-gpu pillow requests sklearn pydub
```

Download a StyleGAN model file, like:

- Female: https://mega.nz/#!CRtiDI7S!xo4zm3n7pkq1Lsfmuio1O8QPpUwHrtFTHjNJ8_XxSJs
- Male: https://mega.nz/#!fMNDkYwS!X-7_nBtsC6P_09CINIJAoVqR3V8Ffbv5On74rVoUbik

## Usage

```bash
python animate_mouth.py <MP3 path> <Model path> <Output path>
```

Eg:

```bash
python animate_mouth.py ~/data/presentday.mp3 ~/data 2019-04-30-stylegan-danbooru2018-portraits-02095-066083.pkl presentday.mp4
```

Arguments:
```bash
usage: animate_mouth.py [-h] [--psi PSI] [--truncate_pre TRUNCATE_PRE]
                        [--truncate_post TRUNCATE_POST]
                        [--output_dir OUTPUT_DIR] [--delete_imgs DELETE_IMGS]
                        [--randomize_noise RANDOMIZE_NOISE] [--seed SEED]
                        audio stylegan output

positional arguments:
  audio                 The filename of the audio file.
  stylegan              The filename of the StyleGAN pkl file.
  output                Set output filename of ffmpeg

optional arguments:
  -h, --help            show this help message and exit
  --psi PSI             The truncation psi. Default 0.7.
  --truncate_pre TRUNCATE_PRE
                        Truncate before dlatent modification. Default True.
  --truncate_post TRUNCATE_POST
                        Truncate after dlatent modification. Default True.
  --output_dir OUTPUT_DIR
                        Directory to output images to.
  --randomize_noise RANDOMIZE_NOISE
                        Whether to randomize the noise every step. Default False.
  --seed SEED           Random seed. 
```

## Notes

The audio amplitude to mouth position mapping is pretty basic and could use some improvement.

See https://github.com/NVlabs/stylegan for original README. Tags from https://github.com/halcy/stylegan.
