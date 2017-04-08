# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

"""Lexicons from literature.

STAB_GUREVYCH_2015 refers to "Parsing Argumentation Structures in Persuasive
Essays" by Christian Stab and Iryna Gurevych, Computational Linguistics, 2015.
See Table C1 therein.

"""

STAB_GUREVYCH_2015 = {
    'forward': ["As a result", "As the consequence", "Because", "Clearly",
                "Consequently", "Considering the subject", "Furthermore",
                "Hence", "leading to the consequence", "so", "So",
                "taking account on this fact", "That is the reason why",
                "The reason is that", "Therefore", "therefore",
                "This means that", "This shows that", "This will result",
                "Thus", "thus", "Thus, it is clearly seen that",
                "Thus, it is seen", "Thus, the example shows"],

    'backward': ["Additionally", "As a matter of fact", "because", "Besides",
                 "due to", "Finally", "First of all", "Firstly", "for example",
                 "For example", "For instance", "for instance", "Furthermore",
                 "has proved it", "In addition", "In addition to this",
                 "In the first place", "is due to the fact that",
                 "It should also be noted", "Moreover", "On one hand",
                 "On the one hand", "On the other hand",
                 "One of the main reasons", "Secondly", "Similarly", "since",
                 "Since", "So", "The reason", "To begin with",
                 "To offer an instance", "What is more"],

    'thesis': ["All in all", "All things considered",
               "As far as I am concerned", "Based on some reasons",
               "by analyzing both the views",
               "considering both the previous fact", "Finally",
               "For the reasons mentioned above",
               "From explanation above",  # sic
               "From this point of view", "I agree that", "I agree with",
               "I agree with the statement that", "I believe",
               "I believe that", "I do not agree with this statement",
               "I firmly believe that", "I highly advocate that",
               "I highly recommend", "I strongly believe that", "I think that",
               "I think the view is", "I totally agree",
               "I totally agree to this opinion", "I would have to argue that",
               "I would reaffirm my position that", "In conclusion",
               "in conclusion", "in my opinion", "In my opinion",
               "In my personal point of view", "in my point of view",
               "In my point of view", "In summary",
               "In the light of the facts outlined above",  # sic
               "it can be said that", "it is clear that",
               "it seems to me that", "my deep conviction", "My sentiments",
               "Overall", "Personally",
               "the above explanations and example shows that",
               "This, however", "To conclude", "To my way of thinking",
               "To sum up", "Ultimately"],

    'rebuttal': ["Admittedly", "although", "Although",
                 "besides these advantages", "but", "But", "Even though",
                 "even though", "However", "Otherwise"]}

MODALS = set(['can', 'could', 'may', 'might', 'must', 'shall', 'should',
              'will', 'would'])  # from wikipedia

# part of speech attributes from Spacy (https://spacy.io/).
POS_ATTRIB = {
    ".": {"pos": "punct", "puncttype": "peri"},
    ",": {"pos": "punct", "puncttype": "comm"},
    "-LRB-": {"pos": "punct", "puncttype": "brck", "punctside": "ini"},
    "-RRB-": {"pos": "punct", "puncttype": "brck", "punctside": "fin"},
    "``": {"pos": "punct", "puncttype": "quot", "punctside": "ini"},
    "\"\"": {"pos": "punct", "puncttype": "quot", "punctside": "fin"},
    "''": {"pos": "punct", "puncttype": "quot", "punctside": "fin"},

    ":": {"pos": "punct"},
    "$": {"pos": "sym", "other": {"symtype": "currency"}},
    "#": {"pos": "sym", "other": {"symtype": "numbersign"}},
    "AFX": {"pos": "adj",  "hyph": "hyph"},
    "CC": {"pos": "conj", "conjtype": "coor"},
    "CD": {"pos": "num", "numtype": "card"},
    "DT": {"pos": "det"},
    "EX": {"pos": "adv", "advtype": "ex"},
    "FW": {"pos": "x", "foreign": "foreign"},
    "HYPH": {"pos": "punct", "puncttype": "dash"},
    "IN": {"pos": "adp"},
    "JJ": {"pos": "adj", "degree": "pos"},
    "JJR": {"pos": "adj", "degree": "comp"},
    "JJS": {"pos": "adj", "degree": "sup"},
    "LS": {"pos": "punct", "numtype": "ord"},
    "MD": {"pos": "verb", "verbtype": "mod"},
    "NIL": {"pos": ""},
    "NN": {"pos": "noun", "number": "sing"},
    "NNP": {"pos": "propn", "nountype": "prop", "number": "sing"},
    "NNPS": {"pos": "propn", "nountype": "prop", "number": "plur"},
    "NNS": {"pos": "noun", "number": "plur"},
    "PDT": {"pos": "adj", "adjtype": "pdt", "prontype": "prn"},
    "POS": {"pos": "part", "poss": "poss"},
    "PRP": {"pos": "pron", "prontype": "prs"},
    "PRP$": {"pos": "adj", "prontype": "prs", "poss": "poss"},
    "RB": {"pos": "adv", "degree": "pos"},
    "RBR": {"pos": "adv", "degree": "comp"},
    "RBS": {"pos": "adv", "degree": "sup"},
    "RP": {"pos": "part"},
    "SYM": {"pos": "sym"},
    "TO": {"pos": "part", "parttype": "inf", "verbform": "inf"},
    "UH": {"pos": "intJ"},
    "VB": {"pos": "verb", "verbform": "inf"},
    "VBD": {"pos": "verb", "verbform": "fin", "tense": "past"},
    "VBG": {"pos": "verb", "verbform": "part", "tense": "pres", "aspect": "prog"},
    "VBN": {"pos": "verb", "verbform": "part", "tense": "past", "aspect": "perf"},
    "VBP": {"pos": "verb", "verbform": "fin", "tense": "pres"},
    "VBZ": {"pos": "verb", "verbform": "fin", "tense": "pres", "number": "sing", "person": 3},
    "WDT": {"pos": "adj", "prontype": "int|rel"},
    "WP": {"pos": "noun", "prontype": "int|rel"},
    "WP$": {"pos": "adj", "poss": "poss", "prontype": "int|rel"},
    "WRB": {"pos": "adv", "prontype": "int|rel"},
    "SP": {"pos": "space"},
    "ADD": {"pos": "x"},
    "NFP": {"pos": "punct"},
    "GW": {"pos": "x"},
    "AFX": {"pos": "x"},
    "HYPH": {"pos": "punct"},
    "XX": {"pos": "x"},
    "BES": {"pos": "verb"},
    "HVS": {"pos": "verb"}
}
if __name__ == '__main__':

    # tests based on lengths given in paper
    assert len(STAB_GUREVYCH_2015['forward']) == 24
    assert len(STAB_GUREVYCH_2015['backward']) == 33
    assert len(STAB_GUREVYCH_2015['thesis']) == 48
    assert len(STAB_GUREVYCH_2015['rebuttal']) == 10
