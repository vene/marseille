"""dataset statistics"""

from marseille.datasets import get_dataset_loader

def counts(docs):
    n_docs = 0
    n_sents = 0
    n_words = 0
    n_props = 0
    n_possible_links = 0
    n_links = 0
    n_empty = 0

    for doc in docs:
        n_docs += 1
        n_sents += len(doc.nlp['sentences'])
        n_words += sum(len(sent['tokens']) for sent in doc.nlp['sentences'])
        n_props += len(doc.prop_offsets)
        this_n_links = doc.label.links.sum()
        n_links += this_n_links
        n_possible_links += len(doc.label.links)

        if this_n_links == 0:
            n_empty += 1

    return n_docs, n_sents, n_words, n_props, n_possible_links, n_links, \
           n_empty

if __name__ == '__main__':
    for ds in ('cdcp', 'ukp'):

        print("dataset=", ds)

        load_tr, ids_tr = get_dataset_loader(ds, split="train")
        load_te, ids_te = get_dataset_loader(ds, split="test")

        (n_docs_tr, n_sents_tr, n_words_tr, n_props_tr, n_possible_tr,
         n_links_tr, n_empty_tr) = counts_tr = counts(load_tr(ids_tr))
        (n_docs_te, n_sents_te, n_words_te, n_props_te, n_possible_te,
         n_links_te, n_empty_te) = counts_te = counts(load_te(ids_te))

        print("n_docs={}, n_sents={}, n_words={}, n_props={}, "
              "n_possible_links={}, "
              "n_links={}, n_empty={}".format(n_docs_tr + n_docs_te,
                                              n_sents_tr + n_sents_te,
                                              n_words_tr + n_words_te,
                                              n_props_tr + n_props_te,
                                              n_possible_tr + n_possible_te,
                                              n_links_tr + n_links_te,
                                              n_empty_tr + n_empty_te))

        print("train split")
        print("n_docs={}, n_sents={}, n_words={}, n_props={}, "
              "n_possible_links={}, "
              "n_links={}, n_empty={}".format(*counts_tr))

        print("test split")
        print("n_docs={}, n_sents={}, n_words={}, n_props={}, "
              "n_possible_links={}, "
              "n_links={}, n_empty={}".format(*counts_te))
