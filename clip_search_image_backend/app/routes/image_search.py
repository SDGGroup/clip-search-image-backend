from flask import Blueprint
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests

IMAGES = [
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/RBWAMFC0100RS0M014?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXM14C0CM30NXIB999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XBWAPAT9000QRI9P13?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXM68C0DP30BYEM026?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/RVW51923610D1PB999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/RBWAMFC0100RS0B999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXM0LR00011RE0U820?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXW60C0DE30AKTB999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/RVW634311905IUB999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/RVW00627760BSSC019?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/HXM3210Y850KLAB999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXM51K0GH50OPX0002?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXM52K00640RE0U805?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXM62C00P20RE0B999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXW00G00010RE0S611?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXW60C0DY40AKTB999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXW00G0DD30RE0B999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/HXM6300EU50R370001?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/RBWAMFD0200JQO1920?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXM62C0DH60RE0B999?w=150",
    "https://bookhtml-todsgroup.msappproxy.net/REST/api/images/XXM0TV0FQ80BYEU824?w=150",
]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
images = [Image.open(requests.get(url, stream=True).raw) for url in IMAGES]

image_search = Blueprint("image_search", __name__, url_prefix="/api")


@image_search.route("/items", methods=["GET"])
def get_items():
    return {"items": IMAGES}


@image_search.route("/search/<string:query>", methods=["GET"])
def search_image(query: str):
    inputs = processor(
        text=query,
        images=images,
        return_tensors="pt",
        padding=True,
    )  # pyright: ignore
    outputs = model(**inputs)  # pyright: ignore
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=0)
    return {"probs": [round(p, 2) for p in torch.flatten(probs).tolist()]}
