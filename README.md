# migration

CLI for running Cellpose CPSAM segmentation and Trackastra tracking on a single
`position/channel/z` plane from an ND2 file.

## Usage

```powershell
uv run migration-track sample.nd2 --position 0 --channel 0 --z 0
```

Outputs:

- trajectory overlay PNG on the first frame
- trajectories CSV with `track_id,parent_track_id,frame,y,x`
