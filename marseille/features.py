# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

import numpy as np
from nltk import Tree
import json

from marseille.indicators import STAB_GUREVYCH_2015, POS_ATTRIB, MODALS
from marseille.preprocess import _transitive


def root_verb_ix(sentence):
    words = sentence['tokens']
    deps = sorted(sentence['basic-dependencies'],
                  key=lambda x: x['dependent'])
    min_height = len(words) + 2
    highest_ix = None

    for k, (tok, dep) in enumerate(zip(words, deps)):
        height = 0
        gov = dep['governor']

        while gov != 0:
            height += 1
            gov = deps[gov - 1]['governor']

        if tok['pos'].startswith('V') and height < min_height:
            min_height = height
            highest_ix = k

    return highest_ix


def productions_between_leaves(tree, start, end):

    productions = set()

    start_pos = tree.leaf_treeposition(start)

    if end + 1 >= len(tree.leaves()):
        end_pos = None
        foul_end_pos = None
    else:
        end_pos = tree.leaf_treeposition(end)
        foul_end_pos = tree.leaf_treeposition(end + 1)

    hit_start = False

    for position in tree.treepositions():

        # if the start leaf is THE FIRST LEAF in our span, turn on hit_start
        if (position == start_pos[:len(position)] and
                all(k == 0 for k in start_pos[len(position):])):
            hit_start = True

        # if the end leaf is in our span at all, ignore the tree
        if (foul_end_pos is not None and
                foul_end_pos[:len(position)] == position):
            continue

        if hit_start:
            subtree = tree[position]
            if hasattr(subtree, 'productions'):
                productions.update(subtree.productions())

        if end_pos is not None and position == end_pos:
            break

    return productions


def iter_discourse_spans(rel, arg_no):
    key = "arg{}_span".format(arg_no)
    label = "Arg{}".format(arg_no)
    arg1_span = rel[key]
    if arg1_span:
        feature = "_".join((rel['sem_cls_1_1'], rel['reltype'], label))
        for span in arg1_span.split(";"):
            start, end = map(int, span.split(".."))
            yield start, end, feature


def get_indicator_features(raw):
    indicator_features = set()

    for type, indicators in STAB_GUREVYCH_2015.items():
        for indicator in indicators:
            if indicator in raw:
                indicator_features.add(type)
                break

    return sorted(list(indicator_features))


def nps_vps(tree):
    nps = [np.leaves()
           for np in tree.subtrees(lambda t: t.label() == 'NP')]
    vps = [vp.leaves()
           for vp in tree.subtrees(lambda t: t.label() == 'VP')]
    return nps, vps


def doc_features(doc):
    return {
        '_doc_id': doc.doc_id,
        'n_props': len(doc.prop_offsets)
    }


def prop_features(doc, prop_id, include_preceding=False, use_intro=False):

    # BEGIN COMMON FOR ALL DOC
    sents = doc.nlp['sentences']

    flat_toks, sent_ix, tok_offset = zip(
        *((tok, sent['index'], tok['characterOffsetBegin'])
          for sent in sents
          for tok in sent['tokens']))

    tok_offset = np.array(tok_offset)
    sent_ix = np.array(sent_ix)

    if len(sents) == 1:
        sent_offset = np.array([0])
    else:
        sent_offset = np.where(sent_ix != np.roll(sent_ix, 1))[0]

    # get discourse arg spans
    discourse_spans = []
    for rel in doc.discourse:
        discourse_spans.extend(iter_discourse_spans(rel, arg_no=1))
        discourse_spans.extend(iter_discourse_spans(rel, arg_no=2))

    discourse_spans = sorted(discourse_spans)

    # paragraph boundaries
    if len(doc.para_offsets) > 1:
        prop_first_in_para = doc.prop_para != np.roll(doc.prop_para, 1)
        prop_last_in_para = doc.prop_para != np.roll(doc.prop_para, -1)
    else:
        prop_first_in_para = np.zeros(len(doc.prop_para), dtype=bool)
        prop_last_in_para = np.zeros(len(doc.prop_para), dtype=bool)
        prop_first_in_para[0] = True
        prop_last_in_para[-1] = True

    # END COMMON FOR ALL DOC

    span_start, span_end = doc.prop_offsets[prop_id]

    first_tok, last_tok = np.searchsorted(tok_offset, [span_start, span_end])

    first_tok_strict = first_tok
    n_tokens_strict = last_tok - first_tok  # n_tokens not counting preceding

    preceding = []

    if include_preceding:
        if (first_tok != 0 and  # if not the beginning of sentence
                sent_ix[first_tok] == sent_ix[first_tok - 1]):

            # move first_tok and span_start to beginning of sentence
            sent_start = np.searchsorted(sent_ix, sent_ix[first_tok])

            preceding = [w['originalText'] for w
                in flat_toks[sent_start:first_tok]]

            first_tok = sent_start
            span_start = tok_offset[first_tok]

    toks = flat_toks[first_tok:last_tok]

    words, tags, lemmas = zip(*((tok['word'], tok['pos'], tok['lemma'])
                                for tok in toks))

    sents_in_span = np.unique(sent_ix[first_tok:last_tok])

    # syntactic, production and dependency features
    prods = set()
    deps = set()
    n_toks_in_sent = 0
    tree_heights = []
    n_subclauses = 0

    root_vb_modal = False
    root_vb_tense = 'unk'

    for k, sent in enumerate(sents_in_span):
        n_toks_in_sent += len(sents[sent]['tokens'])

        sent_start = sent_offset[sent]
        sent_end = sent_start + n_toks_in_sent - 1

        first_in_sent = first_tok - sent_start if k == 0 else 0
        last_in_sent = last_tok - sent_start - 1  # it's fine if > sent_end

        tree = Tree.fromstring(sents[sent]['parse'])
        tree_heights.append(tree.height())
        n_subclauses += 1 + sum(1 for _ in tree.subtrees(lambda t:
                                                         t.label() == 'SBAR'))
        prods.update(productions_between_leaves(tree,
                                                first_in_sent,
                                                last_in_sent))

        dep_ids = ((arc['dependent'], arc['governor'])
                   for arc in sents[sent]['collapsed-ccprocessed-dependencies']
                   if first_in_sent <= arc['dependent'] - 1 <= last_in_sent)
        lemmas_ = ['ROOT'] + [w['lemma'] for w in sents[sent]['tokens']]
        dep_feats = (('{} -> {}'.format(lemmas_[src], lemmas_[trg])
                      for src, trg in dep_ids))

        deps.update(dep_feats)

        # if sentence is not partial:
        if sent_start == first_tok and sent_end == last_tok:
            sent_nlp = sents[sent]
            vb_ix = root_verb_ix(sent_nlp)

            if vb_ix is None:
                continue

            sent_toks = [w['originalText'].lower() for w in sent_nlp['tokens']]

            # is modal?
            # get siblings of root vb
            root_governor = [w for w in sent_nlp['basic-dependencies']
                             if w['dependent'] == vb_ix + 1][0]['governor']
            siblings = [w['dependentGloss'].lower()
                         for w in sent_nlp['basic-dependencies']
                         if w['governor'] == root_governor]

            for sibling in siblings:
                if sibling in MODALS:
                    root_vb_modal = True

            if (vb_ix > 1 and sent_toks[vb_ix - 1] == 'will'):
                root_vb_tense = 'future'
            elif (vb_ix > 2 and
                    sent_toks[vb_ix - 2] == 'going' and
                    sent_toks[vb_ix - 1] == 'to'):
                root_vb_tense = 'future'
            else:
                attribs = POS_ATTRIB.get(sent_nlp['tokens'][vb_ix]['pos'], {})
                root_vb_tense = attribs.get('tense', 'pres')  # default to pres

    tree_height = np.mean(tree_heights)

    prods = list(sorted(str(prod) for prod in prods))
    deps = sorted(list(deps))

    # Discourse features
    discourse_features = set()

    for edu_start, edu_end, feature in discourse_spans:
        if edu_end <= span_start:  # skip while edu fully precedes span
            continue
        if edu_start >= span_end:  # break if edu fully succedes span
            break
        discourse_features.add(feature)  # everything else must overlap

    discourse_features = sorted(list(discourse_features))

    # Indicator features
    indicator_features = get_indicator_features(doc.text[span_start:span_end])

    first_person_markers = {'I', 'me', 'my', 'mine', 'myself'}
    first_person_any = any(word in first_person_markers for word in words)

    this_para = doc.prop_para[prop_id]
    para_start = doc.para_offsets[this_para]

    assert para_start <= span_start
    if this_para + 1 < len(doc.para_offsets):
        para_end = doc.para_offsets[this_para + 1]
        assert span_end <= para_end
    else:
        para_end = None

    indicator_preceding_in_para = get_indicator_features(
        doc.text[para_start:span_start])
    indicator_following_in_para = get_indicator_features(
        doc.text[span_end:para_end])

    # tokens in para
    para_end = len(doc.text) if para_end is None else para_end
    first_tok_para, last_tok_para = np.searchsorted(tok_offset,
                                                    [para_start, para_end])

    n_toks_in_para = last_tok_para - first_tok_para

    n_toks_preceding_in_sent = first_tok_strict - sent_offset[sents_in_span[0]]

    n_toks_following_in_sent = (sent_offset[sents_in_span[-1]] +
                                len(sents[sents_in_span[-1]]['tokens']) -
                                last_tok)

    relative_in_para = (first_tok - first_tok_para) / n_toks_in_para

    preceding_props_in_para = sum(1 for other_prop_id in range(0, prop_id)
                                  if doc.prop_para[other_prop_id] == this_para)
    following_props_in_para = sum(1 for other_prop_id
                                  in range(prop_id + 1, len(doc.prop_offsets))
                                  if doc.prop_para[other_prop_id] == this_para)

    # domain_specific features
    # it is unclear how the original authrors define the intro and conclusion.
    # Based on manual inspection we decide: para 0 = title (unused)
    # (para 1 never appears) para 2 = introduction, para -1 = conclusion
    in_intro = None
    in_conclusion = None
    shared_np_intro = None
    shared_vp_intro = None
    shared_np_conclusion = None
    shared_vp_conclusion = None

    if use_intro:
        in_intro = this_para == 2
        in_conclusion = this_para == len(doc.para_offsets) - 1

        all_sents_start = tok_offset[sent_offset]
        all_sents_end = np.append(all_sents_start[1:] - 1, len(doc.text))
        intro_start, intro_end = doc.para_offsets[2], doc.para_offsets[3]
        conclusion_start = doc.para_offsets[-1]
        intro_sents = ((all_sents_start >= intro_start) &
                       (all_sents_end <= intro_end))

        conclusion_sents = all_sents_start >= conclusion_start

        my_nps, my_vps = nps_vps(tree)
        shared_np_intro = 0
        shared_vp_intro = 0
        shared_np_conclusion = 0
        shared_vp_conclusion = 0

        for sent, sent_in_intro, sent_in_conclusion in zip(sents,
                                                           intro_sents,
                                                           conclusion_sents):
            if sent_in_intro:
                other_tree = Tree.fromstring(sent['parse'])
                nps, vps = nps_vps(other_tree)
                shared_np_intro += sum(1 for np in my_nps if np in nps)
                shared_vp_intro += sum(1 for vp in my_vps if vp in vps)

            if sent_in_conclusion:
                other_tree = Tree.fromstring(sent['parse'])
                nps, vps = nps_vps(other_tree)
                shared_np_conclusion += sum(1 for np in my_nps if np in nps)
                shared_vp_conclusion += sum(1 for vp in my_vps if vp in vps)

    feats = {
        'prop_id_': prop_id,
        'label_': doc.prop_labels[prop_id],
        'n_tokens': int(n_tokens_strict),
        'n_toks_in_sent': n_toks_in_sent,
        'n_toks_in_para': int(n_toks_in_para),
        'n_toks_preceding_in_sent': int(n_toks_preceding_in_sent),
        'n_toks_following_in_sent': int(n_toks_following_in_sent),
        'toks_to_sent_ratio': float(n_tokens_strict / n_toks_in_sent),
        'words': words,
        'pos': tags,
        'lemmas': lemmas,
        'nouns_': [w for w, t in zip(lemmas, tags) if t.startswith('N')],
        'sentences_': [int(i) for i in sents_in_span],
        'productions': prods,
        'dependency_tuples': deps,
        'discourse': discourse_features,
        'indicators': indicator_features,
        'first_person_any': first_person_any,
        'indicator_preceding_in_para': indicator_preceding_in_para,
        'indicator_following_in_para': indicator_following_in_para,
        'is_first_in_para': bool(prop_first_in_para[prop_id]),
        'is_last_in_para': bool(prop_last_in_para[prop_id]),
        'relative_in_para': float(relative_in_para),
        'preceding_props_in_para': preceding_props_in_para,
        'following_props_in_para': following_props_in_para,
        'root_vb_modal': root_vb_modal,
        'root_vb_tense': root_vb_tense,
        'parse_tree_height': float(tree_height),
        'n_subordinate_clauses': n_subclauses,
        'preceding_': preceding
    }

    if use_intro:
        feats.update({
            'is_in_intro': bool(in_intro),
            'is_in_conclusion': bool(in_conclusion),
            'n_shared_np_intro': shared_np_intro,
            'n_shared_vp_intro': shared_vp_intro,
            'n_shared_np_conclusion': shared_np_conclusion,
            'n_shared_vp_conclusion': shared_vp_conclusion,
            'has_shared_np_intro': shared_np_intro > 0,
            'has_shared_vp_intro': shared_vp_intro > 0,
            'has_shared_np_conclusion': shared_np_conclusion > 0,
            'has_shared_vp_conclusion': shared_vp_conclusion > 0,
        })

    return feats


def link_features(doc, src_id, trg_id, all_prop_feats=None):

    if all_prop_feats:  # use cached, precomputed proposition features
        src_feats = all_prop_feats[src_id]
        trg_feats = all_prop_feats[trg_id]
    else:
        # unused atm
        src_feats = prop_features(doc, src_id)
        trg_feats = prop_features(doc, trg_id)

    shared_nouns = (set(src_feats['nouns_'])
                    .intersection(set(trg_feats['nouns_'])))

    shared_sents = (set(src_feats['sentences_'])
                    .intersection(set(trg_feats['sentences_'])))

    link_feats = {
        'same_sentence': len(shared_sents) > 0,
        'src_precedes_trg': src_id < trg_id,
        'trg_precedes_src': trg_id < src_id,
        'props_between': abs(src_id - trg_id) - 1,
        'any_shared_nouns': len(shared_nouns) > 0,
        'n_shared_nouns': len(shared_nouns)
    }

    link_feats.update({"src__" + key: val for key, val in src_feats.items()})
    link_feats.update({"trg__" + key: val for key, val in trg_feats.items()})
    return link_feats


def second_order_features(doc, a, b, c, all_prop_feats):

    a, b, c = int(a), int(b), int(c)

    if all_prop_feats:  # use cached, precomputed proposition features
        a_feats = all_prop_feats[a]
        b_feats = all_prop_feats[b]
        c_feats = all_prop_feats[c]
    else:
        # unused atm
        a_feats = prop_features(doc, a)
        b_feats = prop_features(doc, b)
        c_feats = prop_features(doc, c)

    sents_a, sents_b, sents_c = (set(a_feats['sentences_']),
                                 set(b_feats['sentences_']),
                                 set(c_feats['sentences_']))

    shared_sents = sents_a.intersection(sents_b).intersection(sents_c)
    shared_sents_ab = sents_a.intersection(sents_b)
    shared_sents_ac = sents_a.intersection(sents_c)
    shared_sents_bc = sents_b.intersection(sents_c)

    total_sents = sents_a.union(sents_b).union(sents_c)
    sent_range = max(total_sents) - min(total_sents)

    nouns_a, nouns_b, nouns_c = (set(a_feats['nouns_']),
                                 set(b_feats['nouns_']),
                                 set(c_feats['nouns_']))
    shared_nouns = nouns_a.intersection(nouns_b).intersection(nouns_c)
    shared_nouns_ab = nouns_a.intersection(nouns_b)
    shared_nouns_ac = nouns_a.intersection(nouns_c)
    shared_nouns_bc = nouns_b.intersection(nouns_c)

    total_nouns_a = 1 + len(nouns_a)
    total_nouns_b = 1 + len(nouns_b)
    total_nouns_c = 1 + len(nouns_c)
    total_nouns_ab = 1 + len(nouns_a.union(nouns_b))
    total_nouns_ac = 1 + len(nouns_a.union(nouns_c))
    total_nouns_bc = 1 + len(nouns_b.union(nouns_c))
    total_nouns = 1 + len(nouns_a.union(nouns_b).union(nouns_c))

    feats = {
        'same_sentence': len(shared_sents) > 0,
        'same_sentence_ab': len(shared_sents_ab) > 0,
        'same_sentence_ac': len(shared_sents_ac) > 0,
        'same_sentence_bc': len(shared_sents_bc) > 0,

        'order_abc': a < b < c,
        'order_acb': a < c < b,
        'order_bac': b < a < c,
        'order_bca': b < c < a,
        'order_cab': c < a < b,
        'order_cba': c < b < a,

        'range_leq_1': sent_range <= 1,
        'range_leq_2': sent_range <= 2,
        'range_leq_3': sent_range <= 3,
        'range_leq_4': sent_range <= 4,
        'range_leq_5': sent_range <= 5,

        'any_shared_nouns': len(shared_nouns) > 0,
        'any_shared_nouns_ab': len(shared_nouns_ab) > 0,
        'any_shared_nouns_ac': len(shared_nouns_ac) > 0,
        'any_shared_nouns_bc': len(shared_nouns_bc) > 0,

        'jaccard': len(shared_nouns) / total_nouns,
        'jaccard_ab': len(shared_nouns_ab) / total_nouns_ab,
        'jaccard_ac': len(shared_nouns_ac) / total_nouns_ac,
        'jaccard_bc': len(shared_nouns_bc) / total_nouns_bc,

        'shared_nouns_ratio_a': len(shared_nouns) / total_nouns_a,
        'shared_nouns_ratio_b': len(shared_nouns) / total_nouns_b,
        'shared_nouns_ratio_c': len(shared_nouns) / total_nouns_c,

        'shared_nouns_ratio_ab': len(shared_nouns) / total_nouns_ab,
        'shared_nouns_ratio_ac': len(shared_nouns) / total_nouns_ac,
        'shared_nouns_ratio_bc': len(shared_nouns) / total_nouns_bc,

        'shared_nouns_ab_ratio_a': len(shared_nouns_ab) / total_nouns_a,
        'shared_nouns_ab_ratio_b': len(shared_nouns_ab) / total_nouns_b,
        'shared_nouns_ac_ratio_a': len(shared_nouns_ac) / total_nouns_a,
        'shared_nouns_ac_ratio_c': len(shared_nouns_ac) / total_nouns_c,
        'shared_nouns_bc_ratio_b': len(shared_nouns_bc) / total_nouns_b,
        'shared_nouns_bc_ratio_c': len(shared_nouns_bc) / total_nouns_c,
    }

    return feats



# TODO this is a mess. clean me up!
def add_pmi_features(f, pmi_in, pmi_out):
    src_n_tokens = f['src__n_tokens']
    trg_n_tokens = f['trg__n_tokens']
    src_lemmas = [w.lower() for w in f['src__lemmas']]
    trg_lemmas = [w.lower() for w in f['trg__lemmas']]
    n_pos_src = sum(
        1 for w in src_lemmas if pmi_in.get(w, 0) > 0 or pmi_out.get(w, 0) > 0)
    n_neg_src = sum(
        1 for w in src_lemmas if pmi_in.get(w, 0) < 0 or pmi_out.get(w, 0) < 0)
    n_pos_trg = sum(
        1 for w in trg_lemmas if pmi_in.get(w, 0) > 0 or pmi_out.get(w, 0) > 0)
    n_neg_trg = sum(
        1 for w in trg_lemmas if pmi_in.get(w, 0) < 0 or pmi_out.get(w, 0) < 0)

    f.update({
        'src__pmi_pos_ratio': n_pos_src / src_n_tokens,
        'src__pmi_neg_ratio': n_neg_src / src_n_tokens,
        'trg__pmi_pos_ratio': n_pos_trg / trg_n_tokens,
        'trg__pmi_neg_ratio': n_neg_trg / trg_n_tokens,
        'src__pmi_pos_any': n_pos_src > 0,
        'src__pmi_neg_any': n_neg_src > 0,
        'trg__pmi_pos_any': n_pos_trg > 0,
        'trg__pmi_neg_any': n_neg_trg > 0,
    })


if __name__ == '__main__':
    from itertools import permutations

    from marseille.argdoc import CdcpArgumentationDoc, UkpEssayArgumentationDoc
    from marseille.datasets import cdcp_train_ids, ukp_ids, cdcp_test_ids
    from docopt import docopt

    usage = """
        Usage:
            features (cdcp|ukp|cdcp-test) [--template=S]

    """

    args = docopt(usage)

    template = args['--template']

    if args['cdcp']:

        if template is None:
            template = 'data/process/erule/train/{:05d}'
        ids = cdcp_train_ids
        Doc = CdcpArgumentationDoc

    elif args['cdcp-test']:

        if template is None:
            template = 'data/process/erule/test/{:05d}'
        ids = cdcp_test_ids
        Doc = CdcpArgumentationDoc

    elif args['ukp']:

        if template is None:
            template = 'data/process/ukp-essays/essay{:03d}'
        ids = ukp_ids
        Doc = UkpEssayArgumentationDoc

    for id in ids:

        doc = Doc(template.format(id))

        if args['cdcp'] or args['cdcp-test']:
            include_preceding = False
            use_intro = False
            links = _transitive(doc.links)
        else:
            links = doc.links
            include_preceding = True
            use_intro = True

        prop_ids = range(len(doc.prop_offsets))

        # get doc features
        doc_feats = doc_features(doc)
        all_prop_feats = [prop_features(doc, prop_id, include_preceding,
                                        use_intro)
                          for prop_id in prop_ids]

        with open(template.format(id) + ".propfeatures.json", "w") as f:
           json.dump(all_prop_feats, f, indent=4, sort_keys=True)

        res = []

        for src_id, trg_id in permutations(prop_ids, 2):

            if args['ukp'] and doc.prop_para[src_id] != doc.prop_para[trg_id]:
                continue  # no links across paragraphs in Ukp Essays

            feats = link_features(doc, src_id, trg_id, all_prop_feats)
            feats.update(doc_feats)

            feats['label_'] = (src_id, trg_id) in links  # transitive for cdcp!
            res.append(feats)

        with open(template.format(id) + ".features.json", "w") as f:
            json.dump(res, f, indent=4, sort_keys=True)

        res = []
        for a, b, c in doc.second_order:  # should be available now
            feats = second_order_features(doc, a, b, c, all_prop_feats)
            res.append(feats)

        with open(template.format(id) + ".sec_ord_features.json", "w") as f:
            json.dump(res, f, indent=4, sort_keys=True)
