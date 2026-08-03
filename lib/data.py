from pathlib import Path
import random

from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset, load_from_disk


def _hub_load_kwargs(cache_dir=None, offline=False):
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if offline:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    return kwargs


def _select_split(dataset, preferred_split):
    if not isinstance(dataset, DatasetDict):
        return dataset

    candidates = [preferred_split, "train", "validation", "test"]
    for split_name in candidates:
        if split_name in dataset:
            return dataset[split_name]
    raise ValueError(
        f"Cannot select split '{preferred_split}' from {list(dataset.keys())}"
    )


def _load_local_dataset(path, preferred_split):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local dataset path does not exist: {path}")

    if path.is_dir():
        dataset = load_from_disk(str(path))
        return _select_split(dataset, preferred_split)

    lower_name = path.name.lower()
    if lower_name.endswith(".arrow"):
        return Dataset.from_file(str(path))
    if lower_name.endswith(".parquet"):
        return load_dataset(
            "parquet",
            data_files={preferred_split: str(path)},
            split=preferred_split,
        )
    if lower_name.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
        return load_dataset(
            "json",
            data_files={preferred_split: str(path)},
            split=preferred_split,
        )

    raise ValueError(
        "Unsupported local dataset format. Expected a load_from_disk directory "
        f"or an Arrow/Parquet/JSON file, got: {path}"
    )


def get_wikitext2(
    nsamples,
    seed,
    seqlen,
    tokenizer,
    train_path=None,
    test_path=None,
    cache_dir=None,
    offline=False,
):
    if train_path:
        traindata = _load_local_dataset(train_path, "train")
    else:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            **_hub_load_kwargs(cache_dir, offline),
        )

    if test_path:
        testdata = _load_local_dataset(test_path, "test")
    else:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            **_hub_load_kwargs(cache_dir, offline),
        )

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
 
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(
    nsamples,
    seed,
    seqlen,
    tokenizer,
    train_path=None,
    validation_path=None,
    cache_dir=None,
    offline=False,
):
    if train_path:
        traindata = _load_local_dataset(train_path, "train")
    else:
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            **_hub_load_kwargs(cache_dir, offline),
        )

    if validation_path:
        valdata = _load_local_dataset(validation_path, "validation")
    else:
        valdata = load_dataset(
            "allenai/c4",
            data_files={
                "validation": "en/c4-validation.00000-of-00008.json.gz"
            },
            split="validation",
            **_hub_load_kwargs(cache_dir, offline),
        )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb(nsamples, seed, seqlen, tokenizer, cache_dir=None, offline=False):
    load_kwargs = _hub_load_kwargs(cache_dir, offline)
    traindata = load_dataset(
        "ptb_text_only", "penn_treebank", split="train", **load_kwargs
    )
    testdata = load_dataset(
        "ptb_text_only", "penn_treebank", split="test", **load_kwargs
    )

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc
    

def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    tokenizer=None,
    cache_dir=None,
    offline=False,
    c4_train_path=None,
    c4_validation_path=None,
    wikitext_train_path=None,
    wikitext_test_path=None,
):
    if "wikitext2" in name:
        return get_wikitext2(
            nsamples,
            seed,
            seqlen,
            tokenizer,
            train_path=wikitext_train_path,
            test_path=wikitext_test_path,
            cache_dir=cache_dir,
            offline=offline,
        )
    if "c4" in name:
        return get_c4(
            nsamples,
            seed,
            seqlen,
            tokenizer,
            train_path=c4_train_path,
            validation_path=c4_validation_path,
            cache_dir=cache_dir,
            offline=offline,
        )
    if "ptb" in name:
        return get_ptb(
            nsamples,
            seed,
            seqlen,
            tokenizer,
            cache_dir=cache_dir,
            offline=offline,
        )
    raise ValueError(f"Unsupported dataset: {name}")
