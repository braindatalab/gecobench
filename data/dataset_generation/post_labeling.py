from copy import deepcopy

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download necessary NLTK data files (only needed once)
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')


def is_verb(word: str) -> bool:
    word_tokenized = word_tokenize(text=word)
    pos = pos_tag(word_tokenized)[0][1]
    # Verb tags in NLTK include 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
    return pos.startswith('VB')


def replace_pronouns_by_gender_neutral_pronouns(sentence: list) -> list:
    pronoun_map = {
        "he": "they",
        "she": "they",
        "him": "them",
        "her": "them",
        "his": "their",
        "hers": "theirs",
        "He": "They",
        "She": "They",
        "Him": "Them",
        "Her": "Them",
        "His": "Their",
        "Hers": "Theirs"
    }

    for k, word in enumerate(sentence):
        if word in pronoun_map:
            sentence[k] = pronoun_map[word]

    return sentence


def is_adjective(word: str) -> bool:
    word_tokenized = word_tokenize(text=word)
    pos = pos_tag(word_tokenized)[0][1]
    # Adjective tags in NLTK include 'JJ', 'JJR', 'JJS'
    return pos.startswith('JJ')

def is_adverb(word: str) -> bool:
    word_tokenized = word_tokenize(text=word)
    pos = pos_tag(word_tokenized)[0][1]
    # Adverb tags in NLTK include 'RB', 'RBR', 'RBS'
    return pos.startswith('RB')

def make_gender_neutral_sentence(sentence: list) -> list:
    gender_neutral_pronouns = ['they', 'them', 'their', 'theirs']
    verb_map = {
        'is': 'are',
        'has': 'have'
    }

    copy_of_sentence = deepcopy(sentence)
    neutral_sentence = replace_pronouns_by_gender_neutral_pronouns(
        sentence=copy_of_sentence)

    # Adjust verbs after pronouns
    for k in range(len(neutral_sentence) - 1):
        if neutral_sentence[k].lower() in gender_neutral_pronouns:
            pos = 1
            next_word = neutral_sentence[k + pos]
            if is_adjective(word=next_word) or is_adverb(word=next_word):
                pos += 1
                next_word = neutral_sentence[k + pos]

            if is_verb(word=next_word) and next_word.endswith('s'):
                new_verb = verb_map.get(next_word, next_word[:-1])
                neutral_sentence[k + pos] = new_verb

    return neutral_sentence


def replace_personal_names_by_pronouns(sentence: list) -> list:
    personal_name = 'female_name'
    pronoun = 'she'
    for k, word in enumerate(sentence):
        if word == personal_name:
            sentence[k] = pronoun
    




def main():
    sentences_with_pronouns = [
        ['he', 'is', 'a', 'good',
         'person', 'and', 'he', 'is', 'nice'],
        ['she', 'is', 'a', 'good',
         'person', 'and', 'she', 'is', 'nice'],
        ['After,'    'learning', 'that', 'Jane', 'may', 'soon', 'be', 'engaged', ',',
         'he', 'quickly', 'decides', 'on', 'Elizabeth', 'the', 'next', 'daughter', '.']
    ]

    for sentence in sentences_with_pronouns:
        print(f'Original sentence: {sentence}')
        print(f'Neutral sentence: {make_gender_neutral_sentence(sentence=sentence)}')


if __name__ == '__main__':
    main()
