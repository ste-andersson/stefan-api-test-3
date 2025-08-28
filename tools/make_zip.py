import os, zipfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
out = ROOT.with_suffix('')  # path without extension
zip_path = ROOT.parent / (ROOT.name + '.zip')

def zipdir(path, ziph):
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            arc = os.path.relpath(fp, start=path)
            ziph.write(fp, arcname=arc)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    zipdir(str(ROOT), z)
print(f'Wrote {zip_path}')
