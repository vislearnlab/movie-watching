Vendored copy of the [I2MC](https://github.com/dcnieho/I2MC_Python) Python
package (Hessels et al., 2016), pinned at version 2.2.8, unpacked from the
`I2MC` PyPI wheel rather than installed via pip -- kept in-repo so fixation
detection (`preprocessing/fixations/i2mc_fixations.py`) doesn't depend on an
external install step. MIT-licensed; see `LICENSE`.

To update: download a newer wheel (`pip download --no-deps I2MC==<version>`),
replace `I2MC.py`, `version.py`, `plot/plot.py`, and `__init__.py` with its
contents, and update this file's version note.
