# redundant-trees


# Requirements
- Python 3.8 or above
- GraphViz:
```bash
$ sudo apt  install graphviz
```
- convert (from ImageMagic):
```bash
$ sudo apt install imagemagick
```

If errors ocurred when debuggin try remote the following lines from 
`/etc/ImageMagick-6/policy.xml` (https://stackoverflow.com/a/59193253):
```
<!-- disable ghostscript format types -->
<policy domain="coder" rights="none" pattern="PS" />
<policy domain="coder" rights="none" pattern="PS2" />
<policy domain="coder" rights="none" pattern="PS3" />
<policy domain="coder" rights="none" pattern="EPS" />
<policy domain="coder" rights="none" pattern="PDF" />
<policy domain="coder" rights="none" pattern="XPS" />

```




# Instalation
```bash
$ pip install pipenv
$ pipenv run jupyter-lab
```
