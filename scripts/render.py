import sys
import markdown
import re

from pathlib import Path

extensions = ['extra']

def clean(m: re.Match[str]) -> str:
    old = m[1]
    new = old.replace('_', '\\_')
    # print('old:', old)
    # print('new:', new)
    return '$' + new + '$'

patterns = {
    re.compile(pat, re.S): sub for pat, sub in {
        r'\\([]()[\\])': r'\\\\\1',
        r'\$([^$]+_[^$]*)\$': clean,
    }.items()
}

default_stylesheet = Path(__file__).parent / 'paper.css'


def markdown_to_html(fname, html_path: Path = None, stylesheet=default_stylesheet):

    path = Path(fname)

    use_mdx_math = False
    mathjax_version = 4

    md_ext = '.md'
    html_ext = '.html'

    if path.suffix == md_ext:
        md_path = path
        path = path.with_suffix('')
    else:
        md_path = path.with_suffix(md_ext)

    if html_path is None:
        html_path = path.with_suffix(html_ext)


    # Read the Markdown file
    print('Reading markdown',  md_path)
    text = md_path.read_text()

    if use_mdx_math:
        md = markdown.Markdown(extensions=['mdx_math'],
                             extension_configs={
                                'mdx_math': {
                                    'enable_dollar_delimiter': False,
                                    'use_gitlab_delimiters': False,
                                    'use_asciimath': False
                                }
                            })
    else:
        md = markdown.Markdown(extensions=extensions)
        for pat, sub in patterns.items():
            text = pat.sub(sub, text)


    # Convert Markdown to HTML with the mdx_math extension
    html_content = md.convert(text)

    if mathjax_version == 2:
      mathjax_script = """
    <script type="text/x-mathjax-config">
      MathJax.Hub.Config({
        config: ["MMLorHTML.js"],
        jax: ["input/TeX", "output/HTML-CSS", "output/NativeMML"],
        extensions: ["AMSmath.js", "AMSsymbols.js", "MathMenu.js", "MathZoom.js"]
      });
     </script>
     <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js"></script>
"""
    elif mathjax_version == 3:
      mathjax_script = """
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
    elif mathjax_version == 4:
      mathjax_script = """
    <script>
        window.MathJax = {
            tex: {
              inlineMath: [['\\\\(', '\\\\)'], ['$', '$']],
              displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
              processEscapes: true,
              processEnvironments: true,
              tags: 'ams',  // Enable equation numbering
              tagSide: 'right',
              tagIndent: '.8em',
            },
            options: {
              ignoreHtmlClass: 'tex2jax_ignore',
              processHtmlClass: 'tex2jax_process'
            }
        };
    </script>
    <script type="text/javascript" id="MathJax-script" defer src="https://cdn.jsdelivr.net/npm/mathjax@4/tex-svg.js">
    </script>
    """
    else:
      raise ValueError(f'Unknown mathjax version: {mathjax_version}')

    # title = fname
    title = path.stem

    inline_css = True

    if inline_css:
        css_path = Path(stylesheet)
        css = '<style type="text/css">' + css_path.read_text() + '</style>'
    else:
        css = f'<link rel="stylesheet" href="{stylesheet}">'

    # Create the full HTML document
    html_output = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {mathjax_script}
  {css}
</head>
<body>
    <div id="content" class="tex2jax_process">
        {html_content}
    </div>
</body>
</html>
"""

    # Write the output to an HTML file
    print('Writing html', html_path)
    html_path.write_text(html_output)

