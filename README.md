# migration

CLI for running Cellpose CPSAM segmentation and Trackastra tracking on a single
`position/channel/z` plane from an ND2 file.

## Usage

```powershell
uv run migration-track sample.nd2 --position 0 --channel 0 --z 0 --output ./results
```

To drop short trajectories, pass `--min-track-length 50` or another threshold.

Outputs:

- cached segmentation masks in `./results/segmentation/Pos{position}/`
- cached mask TIFFs named like `img_channel000_position000_time000000000_z000_mask.tif`
- trajectory overlay PNG on the first frame
- trajectories CSV with `track_id,parent_track_id,frame,y,x`
