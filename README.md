# ASR project

## Installation guide

```shell
pip install -r ./requirements.txt
```

```kenlm``` module might not be installed properly if there is no ```g++``` 
installed beforehand. If there are problems with ```kenlm```, follow these 
instructions:

```shell
sudo apt-get update
sudo apt-get install g++ -y
pip install https://github.com/kpu/kenlm/archive/master.zip
```

## How to run ```test.py``` on pretrained models

The script [load_model_checkpoints.py](load_model_checkpoints.py) loads model
checkpoints for English and Russian language from Google Drive. The loading of
language models for both languages is happening in 
[ctc_char_text_encoder.py](hw_asr/text_encoder/ctc_char_text_encoder.py) in
lines 67-68 and 77-78. An additional option was added to ```test.py```: beam
search can be turned on by specifying ```-bs``` command line option, otherwise
it is not executed (for testing Russian model beam search can take up to one
hour). Also, evaluating Russian model might not work because of the version
conflict between ```torch_audiomentations``` requirements and requirements of 
this project, and because evaluation there is happening on mp3 files. For
testing of Russian model to work one should comment the first line in the
[init](hw_asr/augmentations/wave_augmentations/__init__.py) file of wave
augmentations.

```shell
python hw-asr/load_model_checkpoints.py
python hw-asr/test.py -r hw-asr/model_checkpoints/english/deep_speech_english_575.pth -c hw-asr/hw_asr/configs/test_config.json -bs
python hw-asr/test.py -r hw-asr/model_checkpoints/english/deep_speech_english_575.pth -c hw-asr/hw_asr/configs/test_other_config.json -bs
python hw-asr/test.py -r hw-asr/model_checkpoints/russian/deep_speech_russian_805.pth -c hw-asr/hw_asr/configs/russian_test_config.json -bs
```

The results of the evaluation will be printed to ```stdout```, and they should
be roughly equal to:

 Datasets   |      WER      |  CER | WER (bs) | CER (bs) |
|--------------|:-------------:|:------:|:-------------:|:------:|
| ```libri-test-clean``` |  25,3 | 8,9 | 17,7 | 7,3
| ```libri-test-other``` |    55   |   25,3 |44,3 | 23,2
| ```common-voice-russian-test``` | 91 |    44,2 | 82,1 | 49,4|

## Training logs

Training of the English model consisted of two steps: firstly, the model was
trained on LJ-dataset (the logs can be found
[here](https://wandb.ai/whiteteadragon/asr_project_lj/overview)), and,
secondly, the model was trained on Librispeech (part ```train-clean-360```, the
logs can be found
[here](https://wandb.ai/whiteteadragon/asr_project_libri/overview)).
Training logs for the Russian model can be found
[here](https://wandb.ai/whiteteadragon/asr_project_russian/overview).
