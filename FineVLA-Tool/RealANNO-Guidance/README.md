# RealANNO-Guidance

Human annotation guidelines for fine-grained robot manipulation descriptions. This is a static website built with [MkDocs](https://www.mkdocs.org/) that provides annotators with detailed instructions, examples, and reference videos.

## Contents

- **Annotation guidelines** — Detailed rules for writing fine-grained action descriptions across ten dimensions (action sequence, target object, contact & approach, trajectory & orientation, etc.)
- **Example videos** — ~30 robot manipulation clips from BC-Z, Bridge, DROID, RoboCOIN, RoboSet, and other datasets, demonstrating various manipulation tasks
- **Gold trajectory metadata** — Reference annotations (`assets/Gold_Trajectory_metadata.json`) for quality benchmarking

## Usage

Open `index.html` in a browser:

```bash
# Local file
open index.html

# Or serve with Python
python -m http.server 8000 --directory .
# Then visit http://localhost:8000
```

## File Structure

```
RealANNO-Guidance/
├── index.html                          Main guidelines page
├── videos/                             Example manipulation videos (~30 clips)
└── assets/
    ├── Gold_Trajectory_metadata.json   Gold-standard annotation references
    ├── images/                         Logos, screenshots
    ├── stylesheets/                    CSS (MkDocs Material theme)
    └── javascripts/                    JS (MkDocs bundle)
```
