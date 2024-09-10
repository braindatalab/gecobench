import json
from typing import Optional, Literal, Dict
import pydantic
import numpy as np
import random

from utils import (
    is_punctuation,
    is_verb,
    is_adjective,
    is_adverb,
    is_preposition,
    pronoun_map,
    FEMALE_REPLACEMENTS,
    MALE_REPLACEMENTS,
    TAGS,
    gender_neutral_pronouns,
    verb_map,
    gender_neutral_mapping,
)


class Tag(pydantic.BaseModel):
    tag: str
    id: int
    color: Optional[str] = None


class SentenceLabelling(pydantic.BaseModel):
    tag: Optional[int] = None
    replacement: Optional[str] = None


class Gender(pydantic.BaseModel):
    type: Literal["male", "female", "neutral"]
    labelling: Dict[int, SentenceLabelling]


class Version(pydantic.BaseModel):
    type: Literal["all", "subj"]
    gender: Dict[str, Gender]


class DatasetEntry(pydantic.BaseModel):
    sentence: list[str]
    ground_truth: list[int] = []
    gender: Literal["male", "female", "neutral"]
    target: int
    sentence_idx: int


class Sentence(pydantic.BaseModel):
    idx: int
    sentence: list[str]
    tags: Optional[Dict[int, SentenceLabelling]] = {}
    versions: Dict[str, Version]

    def __iter__(self):
        return iter(self.sentence)

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        if idx >= len(self.sentence):
            return None
        elif idx < 0:
            return None

        return self.sentence[idx]

    def to_jsonl_format(self, surname_list=[]):
        surname_replacment = random.choice(surname_list)

        targets = {
            "female": 0,
            "male": 1,
            "neutral": 2,
        }

        ds_versions = {}
        for ds_version in self.versions.values():
            gender_versions = []
            for gender_version in ds_version.gender.values():
                base_sentence = self.sentence.copy()
                for pos, labelling in gender_version.labelling.items():
                    if labelling.replacement is not None:
                        base_sentence[pos] = labelling.replacement
                    elif labelling.tag is not None:
                        if labelling.tag == TAGS.surname:
                            base_sentence[pos] = surname_replacment
                        elif labelling.tag == TAGS.surname_pl:
                            base_sentence[pos] = surname_replacment + "s"

                gender_versions.append(
                    DatasetEntry(
                        sentence=base_sentence,
                        gender=gender_version.type,
                        target=targets[gender_version.type],
                        sentence_idx=self.idx,
                    )
                )

            # Add ground_truth
            gt = []
            for pos in range(len(self)):
                words = [v.sentence[pos] for v in gender_versions]
                if len(set(words)) == 1:
                    gt.append(0)
                else:
                    gt.append(1)

            for v in gender_versions:
                v.ground_truth = gt

            ds_versions[ds_version.type] = gender_versions

        return ds_versions

    def iter_versions(self):
        for version in self.versions.values():
            for gender in version.gender.values():
                for pos, labelling in gender.labelling.items():
                    yield pos, labelling

    def remove_word_from_sentence(self, pos: int):
        # Remove from raw sentence
        self.sentence.pop(pos)

        # Remove from labelling
        for version in self.versions.values():
            for gender in version.gender.values():
                if pos in gender.labelling:
                    del gender.labelling[pos]

                for cur_pos, labelling in list(gender.labelling.items()):
                    if cur_pos > pos:
                        gender.labelling[cur_pos - 1] = labelling
                        del gender.labelling[cur_pos]

        # Remove from raw tags
        if pos in self.tags:
            del self.tags[pos]

        for cur_pos, labelling in list(self.tags.items()):
            if cur_pos > pos:
                self.tags[cur_pos - 1] = labelling
                del self.tags[cur_pos]

    def add_sentence_tag(self):
        self.tags = {}
        for pos, labelling in self.iter_versions():
            if TAGS.is_first_name_tag(labelling.tag):
                self.tags[pos] = SentenceLabelling(tag=TAGS.name.value)
            elif TAGS.is_surname_tag(labelling.tag):
                self.tags[pos] = SentenceLabelling(tag=labelling.tag)

    def remove_subsequent_lastnames(self):
        """
        Remove lastnames which follow firstnames
        """
        first_name_and_last_name_pos = {}

        for pos, labelling in self.iter_versions():
            if TAGS.is_first_name_tag(labelling.tag):
                first_name_and_last_name_pos[pos] = "first_name"
            elif TAGS.is_surname_tag(labelling.tag):
                first_name_and_last_name_pos[pos] = "last_name"

        # Remove the lastname, if it is preceded by a first name
        pos_to_remove = []
        for pos, tag in first_name_and_last_name_pos.items():
            if (
                tag == "last_name"
                and pos - 1 in first_name_and_last_name_pos
                and first_name_and_last_name_pos[pos - 1] == "first_name"
            ):
                pos_to_remove.append(pos)

        for pos in pos_to_remove:
            self.remove_word_from_sentence(pos)

    def remove_first_name(self):
        self.remove_subsequent_lastnames()

        # Replace name_female with "She" and name_male with "He"
        success = True
        to_remove_idx = set()
        for pos, labelling in self.iter_versions():
            if TAGS.is_first_name_tag(labelling.tag):
                if labelling.replacement is None:
                    prev_word = self[pos - 1]
                    next_word = self[pos + 1] or ""
                    next_next_word = self[pos + 2] or ""

                    type = "UNKNOWN"
                    if next_word.strip().replace("'", "") == "s":
                        type = "POSSESSIVE"  # His, Her, Their
                        to_remove_idx.add(pos + 1)
                    elif is_punctuation(prev_word) and is_punctuation(next_word):
                        type = "REMOVE"
                    elif (
                        is_punctuation(prev_word)
                        or is_verb(next_word)
                        or is_verb(next_next_word)
                        or pos == 0
                    ) and not is_verb(prev_word):
                        type = "SUBJ_PRONOUN"  # He, She, They
                    elif is_verb(prev_word) or is_preposition(prev_word):
                        type = "OBJECT_PRONOUN"  # Him, Her, Them

                    if type == "UNKNOWN" or type == "REMOVE":
                        success = False
                        continue

                    if labelling.tag == TAGS.name_female:
                        labelling.replacement = FEMALE_REPLACEMENTS[type]
                    elif labelling.tag == TAGS.name_male:
                        labelling.replacement = MALE_REPLACEMENTS[type]

                    if labelling.replacement is not None:
                        if pos == 0:
                            labelling.replacement = labelling.replacement.capitalize()
                        labelling.tag = None

        # Remove the possessive 's
        for remove_idx in sorted(list(to_remove_idx), reverse=True):
            self.remove_word_from_sentence(remove_idx)

        return success

    def add_neutral_gender(self):
        def to_neutral(word):
            word = word.strip()
            if word in pronoun_map:
                return pronoun_map[word]

            elif word in gender_neutral_mapping:
                return gender_neutral_mapping[word]

            return None

        for version in self.versions.values():
            neutral = {}

            male_l = version.gender["male"].labelling
            female_l = version.gender["female"].labelling

            for pos in range(len(self)):
                m_p = male_l.get(pos)
                f_p = female_l.get(pos)

                if m_p and f_p:
                    if m_p.replacement and f_p.replacement:
                        if m_p.replacement == f_p.replacement:
                            # If male and female have the sign labelling, it is not part of the GT
                            # and we keep the same labelling without neutralizing it

                            neutral[pos] = m_p.model_copy()
                        else:
                            # If they are different, they are part of the GT and we need to neutralize them

                            neutral_form = to_neutral(m_p.replacement)
                            if neutral_form:
                                neutral[pos] = SentenceLabelling(
                                    tag=None, replacement=neutral_form
                                )

                    elif m_p.tag and f_p.tag and m_p.tag == f_p.tag:
                        neutral[pos] = m_p.model_copy()
                else:
                    for t in [m_p, f_p]:
                        if t:
                            if t.replacement:
                                neutral_form = to_neutral(t.replacement)
                                if neutral_form:
                                    neutral[pos] = SentenceLabelling(
                                        tag=None, replacement=neutral_form
                                    )
                            elif t.tag:
                                raise ValueError(
                                    f"Tag in just one of the male and female labelling in sentence {idx}"
                                )

                # If we replaced the word with a neutral form, we need to check if we need
                # to adjust the verb after the pronoun
                if (
                    pos in neutral
                    and neutral[pos].replacement is not None
                    and neutral[pos].replacement.lower() in gender_neutral_pronouns
                ):
                    incr = 1
                    next_word = self[pos + incr]

                    if is_adjective(next_word) or is_adverb(next_word):
                        incr += 1
                        next_word = self[pos + incr]

                    if is_verb(next_word) and next_word.endswith("s"):
                        new_verb = verb_map.get(next_word, next_word[:-1])
                        neutral[pos + incr] = SentenceLabelling(
                            tag=None, replacement=new_verb
                        )

            version.gender["neutral"].labelling = neutral


class LabellingTemplate(pydantic.BaseModel):
    tags: Dict[int, Tag]
    sentences: Dict[int, Sentence]


class Labelling:
    def __init__(self, path: str | None = None, parsed: dict | None = None):
        if path is not None:
            with open(path, "r") as f:
                self.labelling_template = LabellingTemplate(**json.load(f))
        elif parsed is not None:
            self.labelling_template = LabellingTemplate(**parsed)
        else:
            raise ValueError("Either path or parsed must be provided")

    def __len__(self):
        return len(self.labelling_template.sentences)

    def __iter__(self):
        return iter(self.labelling_template.sentences.values())

    def format_to_jsonl(
        self, sentences: list[Sentence], surname_list: list[str]
    ) -> dict[str, list[DatasetEntry]]:
        ret: dict[str, list[DatasetEntry]] = {}
        for sentence in sentences:
            v = sentence.to_jsonl_format(surname_list=surname_list)

            for key in v:
                if key not in ret:
                    ret[key] = []

                ret[key].extend(v[key])

        return ret

    def to_dataset(self, surname_list: list[str]):
        SEED = 1234
        random.seed(SEED)
        frac = 0.8

        # creating training and validation indexes
        indexes = list(np.arange(len(self)))
        train_idx = random.sample(indexes, k=int(len(indexes) * frac))
        test_idx = [i for i in indexes if i not in train_idx]

        sentences_train = [self.labelling_template.sentences[i] for i in train_idx]
        sentences_test = [self.labelling_template.sentences[i] for i in test_idx]

        sentences_train_jsonl = self.format_to_jsonl(sentences_train, surname_list)
        sentences_test_jsonl = self.format_to_jsonl(sentences_test, surname_list)

        return sentences_train_jsonl, sentences_test_jsonl

    def to_json_file(self, path: str):
        ret = self.labelling_template.model_dump()

        # Remove null tags and replacements fields from json
        for sentence in ret["sentences"].values():
            for version in sentence["versions"].values():
                for gender in version["gender"].values():
                    for labelling in gender["labelling"].values():
                        if labelling["tag"] is None:
                            del labelling["tag"]
                        if labelling["replacement"] is None:
                            del labelling["replacement"]

            for labelling in sentence["tags"].values():
                del labelling["replacement"]

        with open(path, "w") as f:
            json.dump(ret, f, indent=4)
