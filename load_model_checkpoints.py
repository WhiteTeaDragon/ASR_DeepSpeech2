import gdown
from hw_asr.utils import ROOT_PATH

english_model_url = "https://drive.google.com/uc?id" \
                    "=1QQ5KVPOYM_hT1uAP0oiOeXUS6ww7bkNd"
russian_model_url = "https://drive.google.com/uc?id" \
                    "=1hdb63daCZ4dBkONc4mCvdne9XfSz78md"
english_file_path = ROOT_PATH / "model_checkpoints" / "english" / \
                    "deep_speech_english_575.pth"
gdown.download(english_model_url, str(english_file_path), quiet=False)
russian_file_path = ROOT_PATH / "model_checkpoints" / "russian" / \
                    "deep_speech_russian_805.pth"
gdown.download(russian_model_url, str(russian_file_path), quiet=False)
