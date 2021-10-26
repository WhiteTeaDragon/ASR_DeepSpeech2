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
it is not executed (for testing russian model beam search can take up to one
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

## Recommended implementation order

You might be a little intimidated by the number of folders and classes. Try to
follow this steps to gradually undestand the workflow.

1) Test `hw_asr/tests/test_dataset.py`  and `hw_asr/tests/test_config.py` and
   make sure everythin works for you
2) Implement missing functions to fix tests
   in  `hw_asr\tests\test_text_encoder.py`
3) Implement missing functions to fix tests
   in  `hw_asr\tests\test_dataloader.py`
4) Implement functions in `hw_asr\metric\utils.py`
5) Implement missing function to run `train.py` with a baseline model
6) Write your own model and try to overfit it on a single batch
7) ~~Pain and suffering~~ Implement your own models and train them. You've
   mastered this template when you can tune your experimental setup just by
   tuning `configs.json` file and running `train.py`
8) Don't forget to write a report about your work
9) Get hired by Google the next day

## Before submitting

0) Make sure your projects run on a new machine after complemeting installation
   guide
1) Search project for `# TODO: your code here` and implement missing
   functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```
3) Make sure `test.py` works fine and works as expected. You should create
   files `default_test_config.json` and your installation guide should download
   your model checpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Credits

this repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template)
repository.

## TODO

These barebones can use more tests. We highly encourage students to create pull
requests to add more tests / new functionality. Current demands:

* Tests for beam search
* W&B logger backend
* README section to describe folders
* Notebook to show how to work with `ConfigParser`
  and `config_parser.init_obj(...)`
