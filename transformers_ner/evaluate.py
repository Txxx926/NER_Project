import logging
from itertools import chain
import hydra
import torch
from omegaconf import omegaconf
from tqdm import tqdm
from utils.scorer import SeqevalScorer

from models.pl_modules import NERModule

log = logging.getLogger(__name__)


def predict(conf: omegaconf.DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model loading
    log.info("Using {} as device".format(device))
    log.info("Loading model")
    model = NERModule.load_from_checkpoint(
        checkpoint_path=conf.evaluate.checkpoint_path
    )
    model.to(device)
    model.eval()

    print(model.labels._index_to_labels)
    print(model.labels._labels_to_index)

    id2labels=model.labels._index_to_labels['labels']
    labels2id=model.labels._labels_to_index['labels']
    # data module
    data_module = hydra.utils.instantiate(conf.data.datamodule)
    data_module.prepare_data()

    # metric
    metric = SeqevalScorer()

    # predict
    predictions, labels = [], []
    lengths=[]
    for idx,batch in tqdm(enumerate(data_module.test_dataloader()), desc="Predictions"):
        batch = model.transfer_batch_to_device(batch, device,idx)
        predictions_kwargs = {**batch, "compute_predictions": True}
        outputs = model(**predictions_kwargs)
        predictions += (outputs["predictions"])
        labels += batch.labels.tolist()
        lengths+= batch["sentence_lengths"]

    # overall score print
    log.info("Overall scores")
    scores = model.compute_f1_score(predictions, labels,lengths)
    for k, v in scores.items():
        log.info(f"{k}: {v}")


@hydra.main(config_path="../conf", config_name="default")
def main(conf: omegaconf.DictConfig):
    predict(conf)


if __name__ == "__main__":
    main()

#python3 transformers_ner/evaluate.py "model.model.language_model=bert-base-cased" evaluate.checkpoint_path="/usr/local/NER/transformers-ner/experiments/bert-base-cased/2023-04-03/07-50-05/ner/pef442of/checkpoints/checkpoint-val_f1_0.6791-epoch_02.ckpt"