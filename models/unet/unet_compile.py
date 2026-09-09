"""Compile a fixed-resolution, single-HALT U-Net deployment image."""
import argparse
import json
from pathlib import Path
import os
import urllib.request
import torch
from unet_common import build_layers, sha256
from unet_precompiled import FORMAT, compile_hardware

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE/'unet_bin/unet-andromeda.bin')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--resolution', default='256x256')
    args = parser.parse_args()
    try:
        width,height = map(int,args.resolution.lower().split('x'))
        if min(width,height) < 16 or width%16 or height%16:
            raise ValueError()
    except ValueError:
        parser.error('resolution must be WIDTHxHEIGHT, both multiples of 16')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    config = json.loads((HERE/'unet_config.json').read_text())
    checkpoint = args.checkpoint or HERE/'unet_bin/unet_carvana_scale0.5_epoch2.pth'
    if args.output.exists() and not args.force:
        parser.error('output exists; pass --force to rebuild')
    if not checkpoint.exists():
        if not args.download:
            parser.error('checkpoint missing; provide --checkpoint or --download')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temp = checkpoint.with_suffix('.download')
        urllib.request.urlretrieve(config['source']['weights_url'], temp)
        os.replace(temp, checkpoint)
    digest = sha256(checkpoint)
    if args.checkpoint is None and digest != config['source']['weights_sha256']:
        raise ValueError(f'official checkpoint SHA256 mismatch: {digest}')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    layers = build_layers(state)
    hardware = compile_hardware(layers,height,width)
    payload = dict(format=FORMAT, precision='IF8-INT/BF16', checkpoint_sha256=digest,
                   layers=layers, mask_values=state.get('mask_values',[0,1]),
                   hardware=hardware, full_graph=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, args.output)
    print(f'Artifact: {args.output}\nCheckpoint SHA256: {digest}')
    print(f'{len(layers)} lowered convolutions; full_graph=true, one HALT; '
          f'image={hardware["model_image"].numel()} bytes, '
          f'program={hardware["program_size"]//32} instructions')

if __name__ == '__main__':
    main()
