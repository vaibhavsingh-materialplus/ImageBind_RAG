# ImageBind: One Embedding Space To Bind Them All

**[FAIR, Meta AI](https://ai.facebook.com/research/)** 

[[`Paper`](https://facebookresearch.github.io/ImageBind/paper)] [[`Blog`](https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/)] [[`Demo`](https://imagebind.metademolab.com/)] [[`Supplementary Video`](https://dl.fbaipublicfiles.com/imagebind/imagebind_video.mp4)] [[`BibTex`](#citing-imagebind)]

ImageBind learns a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. It enables novel emergent applications ‘out-of-the-box’ including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection and generation.

![ImageBind](https://user-images.githubusercontent.com/8495451/236859695-ffa13364-3e39-4d99-a8da-fbfab17f9a6b.gif)

## ImageBind model

Emergent zero-shot classification performance.

<table style="margin: auto">
  <tr>
    <th>Model</th>
    <th><span style="color:blue">IN1k</span></th>
    <th><span style="color:purple">K400</span></th>
    <th><span style="color:green">NYU-D</span></th>
    <th><span style="color:LightBlue">ESC</span></th>
    <th><span style="color:orange">LLVIP</span></th>
    <th><span style="color:purple">Ego4D</span></th>
    <th>download</th>
  </tr>
  <tr>
    <td>imagebind_huge</td>
    <td align="right">77.7</td>
    <td align="right">50.0</td>
    <td align="right">54.0</td>
    <td align="right">66.9</td>
    <td align="right">63.4</td>
    <td align="right">25.0</td>
    <td><a href="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth">checkpoint</a></td>
  </tr>
  
</table>

## Usage

Install pytorch 1.13+ and other 3rd party dependencies.

```shell
conda create --name imagebind python=3.10 -y
conda activate imagebind

pip install .
```

For windows users, you might need to install `soundfile` for reading/writing audio files. (Thanks @congyue1977)

```
pip install soundfile
```

## Citing ImageBind
```
@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang
and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2023}
}
```
