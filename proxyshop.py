from PIL import Image
from subprocess import Popen, PIPE

def run(image, name, artist):
    path_in = f"/tmp/{name} ({artist}).jpg"
    path_out = f"/tmp/{name} ({artist}) - rendered.jpg"
    image.save(path_in, format='png')
    inner_run(path_in, path_out)
    return Image.open(path_out, formats=['jpeg', 'png'])

def inner_run(path_in, path_out):
    script = '/Users/mdite/Personal/MTG-Autoproxy-master/render-target.scpt'
    p = Popen(['osascript', script, path_in, path_out], stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    p.communicate()
    return p.returncode == 0
