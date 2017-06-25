import os
import shutil
import tempfile
import sys
from subprocess import run, PIPE

from .argdoc import _BaseArgumentationDoc
from .custom_logging import logging


CORENLP_PATH = os.path.join(os.environ.get('MARSEILLE_CORENLP_PATH', "."), "*")
WINGNUS_PATH = os.environ.get('MARSEILLE_WINGNUS_PATH', ".")


def _corenlp_process(filename, corenlp_mem="2g"):
    """Run Stanford CoreNLP on the passed file"""

    result = run([
        'java', '-cp', CORENLP_PATH, "-Xmx{}".format(corenlp_mem),
        "edu.stanford.nlp.pipeline.StanfordCoreNLP", "-annotators",
        "tokenize,ssplit,pos,lemma,ner,parse,depparse,sentiment",
        "-file", filename, "-outputFormat", "json"], stdout=PIPE, stderr=PIPE)

    if result.returncode != 0:
        raise ValueError("CoreNLP failed: {}".format(result.stderr.decode()))


def _wing_nus_pdtb_process(filename):
    """Run Wing-NUS Penn Discourse Treebank parser on the passed string"""

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fn_in = os.path.join(tmp_dir, "tmp.txt")
            shutil.copyfile(filename, fn_in)

            cwd = os.getcwd()
            os.chdir(WINGNUS_PATH)
            result = run(['java', '-jar', "parser.jar", fn_in],
                         stdout=PIPE, stderr=PIPE)
            if result.returncode == 0:
                os.chdir(cwd)
                fn_out = os.path.join(tmp_dir, "output", "tmp.txt.pipe")
                shutil.copyfile(fn_out, filename + ".pipe")
            else:
                raise ValueError("WingNUS parser failed: {}".format(
                    result.stderr.decode()))
    finally:
        os.chdir(cwd)


class UserDoc(_BaseArgumentationDoc):

    def __init__(self, filename):
        """Custom argumentation document.

        NLP annotations are performed on the fly if necessary.

        Arguments
        ---------

        filename: string,
            The *base* name of the file to process. A ".txt" extension is
            assumed and added automatically.
        """

        # create corenlp annotations
        if not os.path.exists(filename + ".txt.json"):
            logging.info("Running CoreNLP...")
            _corenlp_process(filename + ".txt")
        # create pdtb annotation
        if not os.path.exists(filename + ".txt.pipe"):
            logging.info("Running WingNUS PDTB parser...")
            _wing_nus_pdtb_process(filename + ".txt")

        self.doc_id = filename
        super(UserDoc, self).__init__(filename)

        # By default, proposition offsets are sentence offsets.
        # TODO: allow user to override
        self.prop_offsets = []
        for sentence in self.nlp['sentences']:
            begin = sentence['tokens'][0]['characterOffsetBegin']
            end = sentence['tokens'][-1]['characterOffsetEnd']
            self.prop_offsets.append((int(begin), int(end)))

        self.para_offsets = [0]
        self.prop_para = [0 for _ in self.prop_offsets]

        # Document is not labeled
        self.links = []
        self.prop_labels = [None for _ in self.prop_offsets]


if __name__ == '__main__':

    doc = UserDoc(sys.argv[1])
