from operator import xor
import shutil
import youtokentome as yttm

from torch.utils.data import DataLoader, ChainDataset

import hw_asr.augmentations
import hw_asr.batch_sampler as batch_sampler_module
import hw_asr.datasets
from hw_asr.collate_fn.collate import collate_fn
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.utils import ROOT_PATH
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


def concatenate_files(file_1, file_2):
    with open(file_1, 'ab') as destination:
        destination.write(bytes("\n", "utf-8"))
        with open(file_2, 'rb') as source:
            shutil.copyfileobj(source, destination)


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    train_text_encoder = None
    bpe_vocab_size = int(configs["bpe_vocab_size"])
    config_params = list(configs["data"].items())
    for i in range(len(config_params)):
        assert config_params[i][0] in ("train", "val", "test"), "Data types " \
                                                             "must be one " \
                                                             "of train, " \
                                                             "val, test"
    config_params = sorted(config_params, key=lambda x: x[0] != "train")
    for split, params in config_params:
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            wave_augs, spec_augs = hw_asr.augmentations.from_configs(configs)
        else:
            wave_augs, spec_augs = None, None

        # create and join datasets
        datasets = []
        if split == "train":
            datasets_path = ROOT_PATH / "data" / "datasets"
            datasets_path.mkdir(exist_ok=True, parents=True)
            all_txt_file_path = str(datasets_path / "train_bpe_texts.txt")
            all_txt_file = open(all_txt_file_path, "w")
            all_txt_file.close()
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                ds, hw_asr.datasets, config_parser=configs,
                wave_augs=wave_augs, spec_augs=spec_augs))
            if split == "train":
                curr_txt_file_path = datasets[-1].all_text_txt_file_path
                concatenate_files(all_txt_file_path, curr_txt_file_path)

        if split == "train":
            model_path = str(ROOT_PATH / "data" / "datasets" / "bpe.model")
            yttm.BPE.train(data=all_txt_file_path, vocab_size=bpe_vocab_size,
                           model=model_path)
            bpe = yttm.BPE(model=model_path)
            alphabet = bpe.vocab()
            for i in range(len(alphabet)):
                new_symbol = []
                for j in range(len(alphabet[i])):
                    if alphabet[i][j] != "▁":
                        new_symbol.append(alphabet[i][j])
                    else:
                        new_symbol.append(" ")
                alphabet[i] = ''.join(new_symbol)
            train_text_encoder = CTCCharTextEncoder(alphabet)
        if train_text_encoder is not None:
            for i in range(len(datasets)):
                datasets[i].set_text_encoder(train_text_encoder)

        assert len(datasets)
        if len(datasets) > 1:
            dataset = ChainDataset(datasets)
        else:
            dataset = datasets[0]
        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_sampler = configs.init_obj(params["batch_sampler"],
                                             batch_sampler_module,
                                             data_source=dataset)
            bs, shuffle = 1, False
        else:
            raise Exception()

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler)
        dataloaders[split] = dataloader
    if train_text_encoder is None:
        train_text_encoder = CTCCharTextEncoder.get_simple_alphabet()
    return dataloaders, train_text_encoder
