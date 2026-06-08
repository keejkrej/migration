# migration

CLI for running Cellpose CPSAM segmentation and Trackastra tracking on a single
`position/channel/z` plane from an ND2 file.

## Usage

```powershell
uv run migration segment sample.nd2 --position 0 --channel 0 --z 0 --output ./results
uv run migration track sample.nd2 --position 0 --channel 0 --z 0 --output ./results --delta-t 1
```

Run `segment` first to produce cached Cellpose masks, then `track` to link trajectories with Trackastra.

Pass `--channel all` or a comma-separated list like `--channel 0,1` to feed multiple ND2 channels to Cellpose CPSAM directly, for example phase contrast plus DAPI. Use the same `--channel` value for `track`. Trackastra receives a weighted fusion of the selected channels after per-channel normalization across time; optional `--track-weights 1,1` sets the weights (equal by default).

To drop short trajectories, pass `--min-track-length 50` or another threshold.
To allow links across missing frames, increase `--delta-t` above `1`.

Outputs:

- cached segmentation masks in `./results/segmentation/Pos{position}/`
- cached mask TIFFs named like `img_channel000_position000_time000000000_z000_mask.tif`
- trajectory overlay PNG on the first frame
- trajectories CSV with `track_id,parent_track_id,frame,y,x`
