#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from pathlib import Path
import render

projects = [
    'coordinate-attention',
    'dharma',
    # 'gated-logit-attention',
    'grounded-attention',
    'principled-attention',
    'ploro',
    'one-pass-forgetting',
    'qana',
    'rotational-transformer',
    'rret',
]

docs_dir = Path('./docs')

for proj in projects:
    render.markdown_to_html(f'{proj}/{proj}.md', html_path=docs_dir / f'{proj}.html')

render.markdown_to_html('README.md', html_path=docs_dir / 'index.html')