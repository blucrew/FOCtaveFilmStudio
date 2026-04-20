# FOCtave Film Studio

**Private companion to [FOCtave](https://github.com/blucrew/FOCtave).**

FOCtave (public) converts stereo e-stim audio into 4-phase restim funscripts.
FOCtave Film Studio (this repo, private) wraps that conversion in a full GUI
plus a video rendering pipeline so you get a synced MPC-HC-playable visual
alongside the funscripts — animated electrode glows, flowing ribbon between
electrodes, bloom breathing on the base image, multi-scene rotation with
crossfades, per-scene placement library, effect color picker with image
eyedropper, and project save/load.

Keep private: contains personal tuning, in-progress effects, and renders
of personal stills.

---

## Layout

```
foctave.py   # bundled copy of public FOCtave converter (imported by studio.py)
place.py     # standalone click-to-place electrode GUI (image -> .electrodes.json)
render.py    # audio + funscripts + image + electrodes -> MP4
studio.py    # full-pipeline GUI: pick audio, add scenes, pick preset, render
examples/    # demo assets + sample frames
```

`studio.py` does `import foctave` and `import render` — both sit next to it.

---

## Quick start

```bash
pip install -r requirements.txt
python studio.py
```

Pick an audio file, add one or more images, place four electrodes on each,
pick a preset, pick an effect style (ribbon / lights / sparks) and colour,
hit **Render**. Output lands in `<output>/<project_name>/`:

```
my_project/
    my_project.e1.funscript   # for restim auto-detect
    my_project.e2.funscript
    my_project.e3.funscript
    my_project.e4.funscript
    my_project.volume.funscript
    my_project.mp4            # play in MPC-HC for restim timeline
    my_project.foctave.json   # project manifest (audio, scenes, tuning)
```

Electrode placements themselves live in a central library at
`~/.foctave/library.json` (keyed by image path) plus `.electrodes.json`
sidecars next to each image — they're not in the render output folder.

---

## Standalone tools

```bash
python place.py path/to/still.jpg    # click 4 electrodes, save sidecar
python render.py path/to/still.jpg   # render single-scene video from sidecar
```

---

## Syncing the converter

`foctave.py` here is a copy of the public repo's converter. To refresh:

```bash
cp ../FOCtave/foctave.py ./foctave.py
```

(The `output_stem=` and `progress=` kwargs on `convert()` are the hooks
Studio uses. They're already in the public repo as documented extension
points, so upstream refreshes stay drop-in compatible.)

---

## License

MIT (inherited from FOCtave).
