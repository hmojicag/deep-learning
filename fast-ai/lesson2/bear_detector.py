from duckduckgo_search import DDGS #DuckDuckGo has changed the api so we need to update
from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url

def search_images(keywords, max_images=200): return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')
import time, json

bear_types = 'grizzly','black','teddy'
path = Path('bears')
if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        download_images(dest, urls=search_images(f'{o} bear', max_images=150))

fns = get_image_files(path)
failed = verify_images(fns)
failed.map(Path.unlink)