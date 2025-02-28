"""
This script evaluates named entity recognition (NER) annotations from multiple systems
and compares their performance against a gold standard dataset. The results are visualized
using a radar chart.

Dependencies:
    - numpy
    - matplotlib
    - os
    - glob

Functions:
    - load_ann(ann_path): Loads entity annotations from a given .ann file.
    - meta_label(named_entities, rule): Maps named entities to a new label format using a given rule.
    - meta_labels_gold_casen(entity_type): Maps entity labels for CasEN model evaluation.
    - meta_labels_gold_spacy_flair(entity_type): Maps entity labels for SpaCy and Flair model evaluation.
    - filter_entities(entities, labels_to_discard): Filters out unwanted entity labels.
    - evaluate_annotations(gold, predicted): Computes precision, recall, and f-score for model evaluation.
    - create_radar_chart(metrics): Generates a radar chart comparing model performances.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob


def load_ann(ann_path):
    """Loads entity annotations from an .ann file and returns a set of named entities."""
    named_entities = set()
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("T"):  # Process entity annotations only
                #T101\tLOC\t100\t200\tMeudon
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue  # Skip malformed lines

                entity_info = parts[1]
                entity_info_parts = entity_info.split(" ")
                label = entity_info_parts[0]  # Entity type (e.g., PERSON, LOC)
                start = int(entity_info_parts[1])  # Start offset
                end = int(entity_info_parts[2])  # End offset
                named_entities.add((label, (start, end)))
    return named_entities


def meta_label(named_entities, rule):
    """Applies a given rule to rename entity labels."""
    return {(rule(named_entity[0]), named_entity[1])
            for named_entity in named_entities}


def meta_labels_gold_casen(entity_type):
    """Maps entity labels to CasEN-compatible format."""
    code = entity_type.split(".")[0].upper()
    if code == "PERS":
        return "PER"
    return code


def meta_labels_gold_spacy_flair(entity_type):
    """Maps entity labels to a SpaCy/Flair-compatible format."""
    code = entity_type.split(".")[0][:3].upper()
    return code if code in ["ORG", "LOC", "PER"] else "MISC"


def filter_entities(entities, labels_to_discard):
    """Filters out entities that contain unwanted labels."""
    return {
        entity
        for entity in entities
        if all(label not in entity[0] for label in labels_to_discard)
    }


def evaluate_annotations(gold, predicted):
    """Computes precision, recall, and f-score for entity annotation evaluation."""
    POS = len(gold)
    PPOS = len(predicted)
    TPOS = len(gold & predicted)
    return {
        "precision": TPOS / PPOS,
        "recall": TPOS / POS,
        "f-score": 2 * TPOS / (PPOS + POS)
    }


def create_radar_chart(metrics):
    """Generates a radar chart comparing NER model performances across datasets."""
    models = list(metrics.keys())
    datasets = [
        "global", "spoken", "encyclopedia", "information", "multi", "poetry",
        "prose"
    ]
    colors = {
        "spacy": "cyan",
        "flair": "blue",
        "casen": "red",
        "casen_minimal": "green",
        "qwen": "black",
        "qwen_more": "grey"
    }
    datasets = [d for d in datasets if d in metrics[models[0]].keys()]
    num_datasets = len(datasets)
    angles = np.linspace(2 * np.pi, 0, num_datasets, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for idx, model in enumerate(models):
        values = [metrics[model][dataset]['f-score'] for dataset in datasets]
        values += values[:1]
        ax.plot(angles,
                values,
                linestyle="dashed" if idx > 1 else "solid",
                linewidth=3,
                label=model,
                color=colors[model])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets)
    ax.set_title('Performances sur les 6 genres et global', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()


# Define folders for models
flair_spacy_labels_to_discard = ["amount", "time.date"]
gold_folder = "FENEC/"
flair_folder = "systems-outputs/original-tagsets/Flair/"
qwen_folder = "systems-outputs/original-tagsets/qwen/"
qwen_more_shots_folder = "systems-outputs/original-tagsets/qwen_more_shots/"
spacy_folder = "systems-outputs/original-tagsets/SpaCy-fr_core_news_lg/"
casEN_large_folder = "systems-outputs/quaero-x-casEN-tagset/CasEN"
casEN_minimal_folder = "systems-outputs/quaero-x-WikiNER-tagset/CasEN"
metrics = {
    "spacy": {},
    "flair": {},
    "casen": {},
    "casen_minimal": {},
    "qwen": {},
    "qwen_more": {}
}
annotations = {}

# Process annotation files
for file in glob(os.path.join(gold_folder, "*.ann")):
    base_name = os.path.basename(file)
    spacy_file = os.path.join(spacy_folder, base_name)
    flair_file = os.path.join(flair_folder, base_name)
    casEN_large_file = os.path.join(casEN_large_folder, base_name)
    casEN_minimal_file = os.path.join(casEN_minimal_folder, base_name)

    qwen_file = os.path.join(qwen_folder, base_name)
    qwen_more_file = os.path.join(qwen_more_shots_folder, base_name)
    if os.path.exists(casEN_large_file) and os.path.exists(
            spacy_file) and os.path.exists(flair_file) and os.path.exists(
                casEN_minimal_file) and os.path.exists(
                    qwen_file) and os.path.exists(qwen_more_file):
        gold_doc = load_ann(file)
        spacy_doc = load_ann(spacy_file)
        flair_doc = load_ann(flair_file)
        qwen_doc = load_ann(qwen_file)
        qwen_more_doc = load_ann(qwen_more_file)
        casen_doc = load_ann(casEN_large_file)
        casen_minimal_doc = load_ann(casEN_minimal_file)
        gold_spacy_flair = filter_entities(gold_doc,
                                           flair_spacy_labels_to_discard)
        gold_casen = meta_label(gold_doc, meta_labels_gold_casen)
        gold_qwen = meta_label(gold_doc, meta_labels_gold_spacy_flair)
        gold_spacy_flair = meta_label(gold_spacy_flair,
                                      meta_labels_gold_spacy_flair)
        domains = [base_name.split("0")[0], "global"]
        for domain in domains:
            if domain in annotations:
                annotations[domain]["spacy"].update(spacy_doc)
                annotations[domain]["flair"].update(flair_doc)
                annotations[domain]["casen"].update(casen_doc)
                annotations[domain]["qwen"].update(qwen_doc)
                annotations[domain]["qwen_more"].update(qwen_more_doc)
                annotations[domain]["casen_minimal"].update(casen_minimal_doc)
                annotations[domain]["gold_spacy_flair"].update(
                    gold_spacy_flair)
                annotations[domain]["gold_casen"].update(gold_casen)
                annotations[domain]["gold_qwen"].update(gold_qwen)
            else:
                annotations[domain] = {
                    "spacy": spacy_doc,
                    "flair": flair_doc,
                    "casen": casen_doc,
                    "qwen": qwen_doc,
                    "qwen_more": qwen_more_doc,
                    "casen_minimal": casen_minimal_doc,
                    "gold_spacy_flair": gold_spacy_flair,
                    "gold_casen": gold_casen,
                    "gold_qwen": gold_qwen
                }
for domain in annotations.keys():
    for model in metrics.keys():
        if model not in ["casen", "qwen"]:
            metrics[model][domain] = evaluate_annotations(
                annotations[domain]["gold_spacy_flair"],
                annotations[domain][model])
    metrics["qwen"][domain] = evaluate_annotations(
        annotations[domain]["gold_spacy_flair"], annotations[domain]["qwen"])
    metrics["casen"][domain] = evaluate_annotations(
        annotations[domain]["gold_casen"], annotations[domain]["casen"])
#lq = (list(annotations["multi"]["qwen"]))
#lg = (list(annotations["multi"]["gold_spacy_flair"]))
#print(sorted([l for l in lq if l not in lg], key=lambda x:x[1][1]))
#print(metrics["qwen"]["multi"])
print(metrics["qwen"])
print(metrics["qwen_more"])
create_radar_chart(metrics)
