# This script is adopted from https://github.com/Diego999/py-rouge/blob/master/rouge/rouge.py
# to compute ROUGE scores for non-English languages.

# Currently, the script supports Hindi.
# Additional language support can be added by adding language specific
# 1) sentence splitter (SENTENCE_SPLIT_DICT or the sentence_split_func argument)
# 2) word tokenizer (WORD_TOKENIZE_DICT or the word_tokenize_func argument)
# 3) pattern of characters to remove (REMOVE_CHAR_PATTERN_DICT or the remove_char_pattern
#    argument)
# 4) stemmer (STEMMER_DICT or the stemming_func argument), this is optional since
#    stemming is not applicable to all languages
# 5) word splitter (WORD_SPLIT_DICT or the word_split_func_argument)

# Major changes made to the original rouge.py include:
# 1) Don't remove non-English or non-numeric characters
# 2) Removed the ensure_compatibility argument as we don't need to reproduce the results of
#    the original perl script that only supports English.


import re
import string
import itertools
import collections

from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from ...language_utils.hi.hindi_stemmer import hi_stem
from rouge import Rouge


class RougeExt(Rouge):
    DEFAULT_METRICS = {"rouge-n"}
    DEFAULT_N = 1
    STATS = ["f", "p", "r"]
    AVAILABLE_METRICS = {"rouge-n", "rouge-l", "rouge-w"}
    AVAILABLE_LENGTH_LIMIT_TYPES = {"words", "bytes"}

    SENTENCE_SPLIT_DICT = {"hi": sentence_tokenize.sentence_split}
    WORD_TOKENIZE_DICT = {"hi": indic_tokenize.trivial_tokenize}
    REMOVE_CHAR_PATTERN_DICT = {
        "hi": re.compile(r"([" + string.punctuation + r"\u0964\u0965" + r"])")
    }
    STEMMER_DICT = {"hi": hi_stem}
    WORD_SPLIT_DICT = {}

    # REMOVE_CHAR_PATTERN = re.compile('[^A-Za-z0-9]')

    # Hack to not tokenize "cannot" to "can not" and consider them different as in the
    # official ROUGE script
    # KEEP_CANNOT_IN_ONE_WORD = re.compile('cannot')
    # KEEP_CANNOT_IN_ONE_WORD_REVERSED = re.compile('_cannot_')

    # WORDNET_KEY_VALUE = {}
    # WORDNET_DB_FILEPATH = 'wordnet_key_value.txt'
    # WORDNET_DB_FILEPATH_SPECIAL_CASE = 'wordnet_key_value_special_cases.txt'
    # WORDNET_DB_DELIMITER = '|'
    # STEMMER = None

    def __init__(
        self,
        metrics=None,
        max_n=None,
        limit_length=True,
        length_limit=665,
        length_limit_type="bytes",
        apply_avg=True,
        apply_best=False,
        stemming=True,
        alpha=0.5,
        weight_factor=1.0,
        language="hi",
        sentence_split_func=None,
        word_tokenize_func=None,
        remove_char_pattern=None,
        stemming_func=None,
        word_split_func=None,
    ):
        """
        Handle the ROUGE score computation as in the official perl script.

        Note 1: Small differences might happen if the resampling of the perl script is not
                high enough (as the average depends on this).
        Note 2: Stemming of the official Porter Stemmer of the ROUGE perl script is slightly
                different and the Porter one implemented in NLTK. However, special cases of
                DUC 2004 have been traited.
                The solution would be to rewrite the whole perl stemming in python from
                the original script

        Args:
            metrics: What ROUGE score to compute. Available: ROUGE-N, ROUGE-L, ROUGE-W.
                Default: ROUGE-N
            max_n: N-grams for ROUGE-N if specify. Default:1
            limit_length: If the summaries must be truncated. Defaut:True
            length_limit: Number of the truncation where the unit is express int length_limit_Type.
                Default:665 (bytes)
            length_limit_type: Unit of length_limit. Available: words, bytes. Default: 'bytes'
            apply_avg: If we should average the score of multiple samples. Default: True. If
                apply_Avg & apply_best = False, then each ROUGE scores are independant
            apply_best: Take the best instead of the average. Default: False, then each ROUGE
                scores are independant
            stemming: Apply stemming to summaries. Default: True
            alpha: Alpha use to compute f1 score: P*R/((1-a)*P + a*R). Default:0.5
            weight_factor: Weight factor to be used for ROUGE-W. Official rouge score defines
                it at 1.2. Default: 1.0
            sentence_split_func (function, optional): Language specific function for splitting
                sentences. Defaults to None.
            word_tokenize_func (function, optional): Language specific function for tokenizing text.
                Defaults to None.
            remove_char_pattern (_sre.SRE_Pattern, optional): Langauge specific regular expression
                pattern for removing special characters, e.g. punctuations. Defaults to None.
            stemming_func (function, optional): Language specific stemmer. Defaults to None.
            word_split_func (function, optional): Language specific word splitter. Only needed if
            the language words are not separated by space, e.g. Chinese. Defaults to None.

        Raises:
            ValueError: raises exception if metric is not among AVAILABLE_METRICS
            ValueError: raises exception if length_limit_type is not among
                AVAILABLE_LENGTH_LIMIT_TYPES
            ValueError: raises exception if weight_factor < 0
        """
        self.metrics = metrics[:] if metrics is not None else RougeExt.DEFAULT_METRICS
        for m in self.metrics:
            if m not in RougeExt.AVAILABLE_METRICS:
                raise ValueError("Unknown metric '{}'".format(m))

        self.max_n = max_n if "rouge-n" in self.metrics else None
        # Add all rouge-n metrics
        if self.max_n is not None:
            index_rouge_n = self.metrics.index("rouge-n")
            del self.metrics[index_rouge_n]
            self.metrics += ["rouge-{}".format(n) for n in range(1, self.max_n + 1)]
        self.metrics = set(self.metrics)

        self.limit_length = limit_length
        if self.limit_length:
            if length_limit_type not in RougeExt.AVAILABLE_LENGTH_LIMIT_TYPES:
                raise ValueError("Unknown length_limit_type '{}'".format(length_limit_type))

        self.length_limit = length_limit
        if self.length_limit == 0:
            self.limit_length = False
        self.length_limit_type = length_limit_type
        self.stemming = stemming

        self.apply_avg = apply_avg
        self.apply_best = apply_best
        self.alpha = alpha
        self.weight_factor = weight_factor
        if self.weight_factor <= 0:
            raise ValueError("ROUGE-W weight factor must greater than 0.")

        self.language = language
        if sentence_split_func is None:
            self.sentence_split = RougeExt.SENTENCE_SPLIT_DICT[self.language]
        else:
            self.sentence_split = sentence_split_func
        if word_tokenize_func is None:
            self.word_tokenize = RougeExt.WORD_TOKENIZE_DICT[self.language]
        else:
            self.word_tokenize = word_tokenize_func
        if remove_char_pattern is None:
            self.remove_char_pattern = RougeExt.REMOVE_CHAR_PATTERN_DICT[self.language]
        else:
            self.remove_char_pattern = remove_char_pattern
        if self.language not in RougeExt.STEMMER_DICT.keys() and stemming_func is None:
            self.stemmer = None
            warnings.warn("Language-specific stemmer is not available. Skipping stemming.")
        elif stemming_func is None:
            self.stemmer = RougeExt.STEMMER_DICT[self.language]
        else:
            self.stemmer = stemming_func

        if self.language not in RougeExt.WORD_SPLIT_DICT.keys() and word_split_func is None:
            self.word_split = None
        elif word_split_func is None:
            self.word_split = RougeExt.WORD_SPLIT_DICT[self.language]
        else:
            self.word_split = word_split_func

        # # Load static objects
        # if len(Rouge.WORDNET_KEY_VALUE) == 0:
        #     Rouge.load_wordnet_db(ensure_compatibility)
        # if Rouge.STEMMER is None:
        #     Rouge.load_stemmer(ensure_compatibility)

    # @staticmethod
    # def load_stemmer(ensure_compatibility):
    #     """
    #     Load the stemmer that is going to be used if stemming is enabled
    #     Args
    #         ensure_compatibility: Use same stemmer and special "hacks" to product
    #             same results as in the official perl script (besides the number of
    #             sampling if not high enough)
    #     """
    #     Rouge.STEMMER = nltk.stem.porter.PorterStemmer('ORIGINAL_ALGORITHM') if
    #         ensure_compatibility else nltk.stem.porter.PorterStemmer()

    # @staticmethod
    # def load_wordnet_db(ensure_compatibility):
    #     """
    #     Load WordNet database to apply specific rules instead of stemming + load file for
    #     special cases to ensure kind of compatibility (at list with DUC 2004) with the
    #     original stemmer used in the Perl script
    #     Args
    #         ensure_compatibility: Use same stemmer and special "hacks" to product same
    #             results as in the official perl script (besides the number of sampling
    #             if not high enough)

    #     Raises:
    #         FileNotFoundError: If one of both databases is not found
    #     """
    #     files_to_load = [Rouge.WORDNET_DB_FILEPATH]
    #     if ensure_compatibility:
    #         files_to_load.append(Rouge.WORDNET_DB_FILEPATH_SPECIAL_CASE)

    #     for wordnet_db in files_to_load:
    #         filepath = pkg_resources.resource_filename(__name__, wordnet_db)
    #         if not os.path.exists(filepath):
    #             raise FileNotFoundError("The file '{}' does not exist".format(filepath))

    #         with open(filepath, 'r', encoding='utf-8') as fp:
    #             for line in fp:
    #                 k, v = line.strip().split(Rouge.WORDNET_DB_DELIMITER)
    #                 assert k not in Rouge.WORDNET_KEY_VALUE
    #                 Rouge.WORDNET_KEY_VALUE[k] = v

    def tokenize_text(self, text):
        """
        Tokenize text in the specific language

        Args:
          text: The string text to tokenize
          language: Language of the text

        Returns:
          List of tokens of text
        """
        return self.word_tokenize(text, self.language)

    def split_into_sentences(self, text):
        """
        Split text into sentences, using specified language.

        Args:
          text: The string text to tokenize
          language: Language of the text

        Returns:
          List of tokens of text
        """

        return self.sentence_split(text, self.language)

    # @staticmethod
    # def stem_tokens(tokens):
    #     """
    #     Apply WordNetDB rules or Stem each token of tokens

    #     Args:
    #       tokens: List of tokens to apply WordNetDB rules or to stem

    #     Returns:
    #       List of final stems
    #     """
    #     # Stemming & Wordnet apply only if token has at least 3 chars
    #     for i, token in enumerate(tokens):
    #         if len(token) > 0:
    #             if len(token) > 3:
    #                 if token in Rouge.WORDNET_KEY_VALUE:
    #                     token = Rouge.WORDNET_KEY_VALUE[token]
    #                 else:
    #                     token = Rouge.STEMMER.stem(token)
    #                 tokens[i] = token

    #     return tokens

    def stem_tokens(self, tokens):
        """
        Stem each token of tokens

        Args:
          tokens: List of tokens to stem

        Returns:
          List of final stems
        """
        for i, token in enumerate(tokens):
            tokens[i] = self.stemmer(token)

        return tokens

    @staticmethod
    def _get_ngrams(n, text):
        """
        Calcualtes n-grams.

        Args:
          n: which n-grams to calculate
          text: An array of tokens

        Returns:
          A set of n-grams with their number of occurences
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        ngram_set = collections.defaultdict(int)
        max_index_ngram_start = len(text) - n
        for i in range(max_index_ngram_start + 1):
            ngram_set[tuple(text[i : i + n])] += 1
        return ngram_set

    @staticmethod
    def _split_into_words(sentences):
        """
        Splits multiple sentences into words and flattens the result

        Args:
          sentences: list of string

        Returns:
          A list of words (split by white space)
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        if self.word_split is None:
            return list(itertools.chain(*[_.split() for _ in sentences]))
        else:
            return list(itertools.chain(*[self.word_split(_) for _ in sentences]))

    @staticmethod
    def _get_word_ngrams_and_length(n, sentences):
        """
        Calculates word n-grams for multiple sentences.

        Args:
          n: wich n-grams to calculate
          sentences: list of string

        Returns:
          A set of n-grams, their frequency and #n-grams in sentences
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        assert len(sentences) > 0
        assert n > 0

        tokens = RougeExt._split_into_words(sentences)
        return RougeExt._get_ngrams(n, tokens), tokens, len(tokens) - (n - 1)

    @staticmethod
    def _get_unigrams(sentences):
        """
        Calcualtes uni-grams.

        Args:
          sentences: list of string

        Returns:
          A set of n-grams and their freqneucy
        """
        assert len(sentences) > 0

        tokens = RougeExt._split_into_words(sentences)
        unigram_set = collections.defaultdict(int)
        for token in tokens:
            unigram_set[token] += 1
        return unigram_set, len(tokens)

    @staticmethod
    def _compute_p_r_f_score(
        evaluated_count, reference_count, overlapping_count, alpha=0.5, weight_factor=1.0
    ):
        """
        Compute precision, recall and f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          evaluated_count: #n-grams in the hypothesis
          reference_count: #n-grams in the reference
          overlapping_count: #n-grams in common between hypothesis and reference
          alpha: Value to use for the F1 score (default: 0.5)
          weight_factor: Weight factor if we have use ROUGE-W (default: 1.0, no impact)

        Returns:
          A dict with 'p', 'r' and 'f' as keys fore precision, recall, f1 score
        """
        precision = 0.0 if evaluated_count == 0 else overlapping_count / evaluated_count
        if weight_factor != 1.0:
            precision = precision ** (1.0 / weight_factor)
        recall = 0.0 if reference_count == 0 else overlapping_count / reference_count
        if weight_factor != 1.0:
            recall = recall ** (1.0 / weight_factor)
        f1_score = RougeExt._compute_f_score(precision, recall, alpha)
        return {"f": f1_score, "p": precision, "r": recall}

    @staticmethod
    def _compute_f_score(precision, recall, alpha=0.5):
        """
        Compute f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          precision: precision
          recall: recall
          overlapping_count: #n-grams in common between hypothesis and reference

        Returns:
            f1 score
        """
        return (
            0.0
            if (recall == 0.0 or precision == 0.0)
            else precision * recall / ((1 - alpha) * precision + alpha * recall)
        )

    @staticmethod
    def _compute_ngrams(evaluated_sentences, reference_sentences, n):
        """
        Computes n-grams overlap of two text collections of sentences.
        Source: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf

        Args:
          evaluated_sentences: The sentences that have been picked by the
                               summarizer
          reference_sentences: The sentences from the referene set
          n: Size of ngram

        Returns:
          Number of n-grams for evaluated_sentences, reference_sentences and intersection of both.
          intersection of both count multiple of occurences in n-grams match several times

        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams, _, evaluated_count = RougeExt._get_word_ngrams_and_length(
            n, evaluated_sentences
        )
        reference_ngrams, _, reference_count = RougeExt._get_word_ngrams_and_length(
            n, reference_sentences
        )

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = set(evaluated_ngrams.keys()).intersection(set(reference_ngrams.keys()))
        overlapping_count = 0
        for ngram in overlapping_ngrams:
            overlapping_count += min(evaluated_ngrams[ngram], reference_ngrams[ngram])

        return evaluated_count, reference_count, overlapping_count

    @staticmethod
    def _compute_ngrams_lcs(evaluated_sentences, reference_sentences, weight_factor=1.0):
        """
        Computes ROUGE-L (summary level) of two text collections of sentences.
        http://research.microsoft.com/en-us/um/people/cyl/download/papers/
        rouge-working-note-v1.3.1.pdf
        Args:
          evaluated_sentences: The sentences that have been picked by the summarizer
          reference_sentence: One of the sentences in the reference summaries
          weight_factor: Weight factor to be used for WLCS (1.0 by default if LCS)
        Returns:
          Number of LCS n-grams for evaluated_sentences, reference_sentences and intersection
              of both.
          intersection of both count multiple of occurences in n-grams match several times
        Raises:
          ValueError: raises exception if a param has len <= 0
        """

        def _lcs(x, y):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(int)
            dirs = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        vals[i, j] = vals[i - 1, j - 1] + 1
                        dirs[i, j] = "|"
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = "^"
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = "<"

            return vals, dirs

        def _wlcs(x, y, weight_factor):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(float)
            dirs = collections.defaultdict(int)
            lengths = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        length_tmp = lengths[i - 1, j - 1]
                        vals[i, j] = (
                            vals[i - 1, j - 1]
                            + (length_tmp + 1) ** weight_factor
                            - length_tmp ** weight_factor
                        )
                        dirs[i, j] = "|"
                        lengths[i, j] = length_tmp + 1
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = "^"
                        lengths[i, j] = 0
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = "<"
                        lengths[i, j] = 0

            return vals, dirs

        def _mark_lcs(mask, dirs, m, n):
            while m != 0 and n != 0:
                if dirs[m, n] == "|":
                    m -= 1
                    n -= 1
                    mask[m] = 1
                elif dirs[m, n] == "^":
                    m -= 1
                elif dirs[m, n] == "<":
                    n -= 1
                else:
                    raise UnboundLocalError("Illegal move")

            return mask

        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_unigrams_dict, evaluated_count = RougeExt._get_unigrams(evaluated_sentences)
        reference_unigrams_dict, reference_count = RougeExt._get_unigrams(reference_sentences)

        # Has to use weight factor for WLCS
        use_WLCS = weight_factor != 1.0
        if use_WLCS:
            evaluated_count = evaluated_count ** weight_factor
            reference_count = 0

        overlapping_count = 0.0
        for reference_sentence in reference_sentences:
            reference_sentence_tokens = reference_sentence.split()
            if use_WLCS:
                reference_count += len(reference_sentence_tokens) ** weight_factor
            hit_mask = [0 for _ in range(len(reference_sentence_tokens))]

            for evaluated_sentence in evaluated_sentences:
                evaluated_sentence_tokens = evaluated_sentence.split()

                if use_WLCS:
                    _, lcs_dirs = _wlcs(
                        reference_sentence_tokens, evaluated_sentence_tokens, weight_factor
                    )
                else:
                    _, lcs_dirs = _lcs(reference_sentence_tokens, evaluated_sentence_tokens)
                _mark_lcs(
                    hit_mask,
                    lcs_dirs,
                    len(reference_sentence_tokens),
                    len(evaluated_sentence_tokens),
                )

            overlapping_count_length = 0
            for ref_token_id, val in enumerate(hit_mask):
                if val == 1:
                    token = reference_sentence_tokens[ref_token_id]
                    if evaluated_unigrams_dict[token] > 0 and reference_unigrams_dict[token] > 0:
                        evaluated_unigrams_dict[token] -= 1
                        reference_unigrams_dict[ref_token_id] -= 1

                        if use_WLCS:
                            overlapping_count_length += 1
                            if (
                                ref_token_id + 1 < len(hit_mask) and hit_mask[ref_token_id + 1] == 0
                            ) or ref_token_id + 1 == len(hit_mask):
                                overlapping_count += overlapping_count_length ** weight_factor
                                overlapping_count_length = 0
                        else:
                            overlapping_count += 1

        if use_WLCS:
            reference_count = reference_count ** weight_factor

        return evaluated_count, reference_count, overlapping_count

    def get_scores(self, hypothesis, references):
        """
        Compute precision, recall and f1 score between hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between hypothesis and references

        Raises:
          ValueError: raises exception if a type of hypothesis is different than the one of
              reference
          ValueError: raises exception if a len of hypothesis is different than the one of reference
        """
        if isinstance(hypothesis, str):
            hypothesis, references = [hypothesis], [references]

        if type(hypothesis) != type(references):
            raise ValueError("'hyps' and 'refs' are not of the same type")

        if len(hypothesis) != len(references):
            raise ValueError("'hyps' and 'refs' do not have the same length")

        scores = {}
        has_rouge_n_metric = (
            len([metric for metric in self.metrics if metric.split("-")[-1].isdigit()]) > 0
        )
        if has_rouge_n_metric:
            scores = {**scores, **self._get_scores_rouge_n(hypothesis, references)}

        has_rouge_l_metric = (
            len([metric for metric in self.metrics if metric.split("-")[-1].lower() == "l"]) > 0
        )
        if has_rouge_l_metric:
            scores = {**scores, **self._get_scores_rouge_l_or_w(hypothesis, references, False)}

        has_rouge_w_metric = (
            len([metric for metric in self.metrics if metric.split("-")[-1].lower() == "w"]) > 0
        )
        if has_rouge_w_metric:
            scores = {**scores, **self._get_scores_rouge_l_or_w(hypothesis, references, True)}

        return scores

    def _get_scores_rouge_n(self, all_hypothesis, all_references):
        """
        Computes precision, recall and f1 score between all hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
        metrics = [metric for metric in self.metrics if metric.split("-")[-1].isdigit()]

        if self.apply_avg or self.apply_best:
            scores = {metric: {stat: 0.0 for stat in RougeExt.STATS} for metric in metrics}
        else:
            scores = {
                metric: [{stat: [] for stat in RougeExt.STATS} for _ in range(len(all_hypothesis))]
                for metric in metrics
            }

        for sample_id, (hypothesis, references) in enumerate(zip(all_hypothesis, all_references)):
            assert isinstance(hypothesis, str)
            has_multiple_references = False
            if isinstance(references, list):
                has_multiple_references = len(references) > 1
                if not has_multiple_references:
                    references = references[0]

            # Prepare hypothesis and reference(s)
            hypothesis = self._preprocess_summary_as_a_whole(hypothesis)
            references = (
                [self._preprocess_summary_as_a_whole(reference) for reference in references]
                if has_multiple_references
                else [self._preprocess_summary_as_a_whole(references)]
            )

            # Compute scores
            for metric in metrics:
                suffix = metric.split("-")[-1]
                n = int(suffix)

                # Aggregate
                if self.apply_avg:
                    # average model
                    total_hypothesis_ngrams_count = 0
                    total_reference_ngrams_count = 0
                    total_ngrams_overlapping_count = 0

                    for reference in references:
                        n_grams_counts = RougeExt._compute_ngrams(hypothesis, reference, n)
                        hypothesis_count, reference_count, overlapping_ngrams = n_grams_counts
                        total_hypothesis_ngrams_count += hypothesis_count
                        total_reference_ngrams_count += reference_count
                        total_ngrams_overlapping_count += overlapping_ngrams

                    score = RougeExt._compute_p_r_f_score(
                        total_hypothesis_ngrams_count,
                        total_reference_ngrams_count,
                        total_ngrams_overlapping_count,
                        self.alpha,
                    )

                    for stat in RougeExt.STATS:
                        scores[metric][stat] += score[stat]
                else:
                    # Best model
                    if self.apply_best:
                        best_current_score = None
                        for reference in references:
                            n_grams_counts = RougeExt._compute_ngrams(hypothesis, reference, n)
                            hypothesis_count, reference_count, overlapping_ngrams = n_grams_counts
                            score = RougeExt._compute_p_r_f_score(
                                hypothesis_count, reference_count, overlapping_ngrams, self.alpha
                            )
                            if best_current_score is None or score["r"] > best_current_score["r"]:
                                best_current_score = score

                        for stat in RougeExt.STATS:
                            scores[metric][stat] += best_current_score[stat]
                    # Keep all
                    else:
                        for reference in references:
                            n_grams_counts = RougeExt._compute_ngrams(hypothesis, reference, n)
                            hypothesis_count, reference_count, overlapping_ngrams = n_grams_counts
                            score = RougeExt._compute_p_r_f_score(
                                hypothesis_count, reference_count, overlapping_ngrams, self.alpha
                            )
                            for stat in RougeExt.STATS:
                                scores[metric][sample_id][stat].append(score[stat])

        # Compute final score with the average or the the max
        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for metric in metrics:
                for stat in RougeExt.STATS:
                    scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _get_scores_rouge_l_or_w(self, all_hypothesis, all_references, use_w=False):
        """
        Computes precision, recall and f1 score between all hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
        metric = "rouge-w" if use_w else "rouge-l"
        if self.apply_avg or self.apply_best:
            scores = {metric: {stat: 0.0 for stat in RougeExt.STATS}}
        else:
            scores = {
                metric: [{stat: [] for stat in RougeExt.STATS} for _ in range(len(all_hypothesis))]
            }

        for sample_id, (hypothesis_sentences, references_sentences) in enumerate(
            zip(all_hypothesis, all_references)
        ):
            assert isinstance(hypothesis_sentences, str)
            has_multiple_references = False
            if isinstance(references_sentences, list):
                has_multiple_references = len(references_sentences) > 1
                if not has_multiple_references:
                    references_sentences = references_sentences[0]

            # Prepare hypothesis and reference(s)
            hypothesis_sentences = self._preprocess_summary_per_sentence(hypothesis_sentences)
            references_sentences = (
                [
                    self._preprocess_summary_per_sentence(reference)
                    for reference in references_sentences
                ]
                if has_multiple_references
                else [self._preprocess_summary_per_sentence(references_sentences)]
            )

            # Compute scores
            # Aggregate
            if self.apply_avg:
                # average model
                total_hypothesis_ngrams_count = 0
                total_reference_ngrams_count = 0
                total_ngrams_overlapping_count = 0

                for reference_sentences in references_sentences:
                    n_grams_counts = RougeExt._compute_ngrams_lcs(
                        hypothesis_sentences,
                        reference_sentences,
                        self.weight_factor if use_w else 1.0,
                    )
                    hypothesis_count, reference_count, overlapping_ngrams = n_grams_counts
                    total_hypothesis_ngrams_count += hypothesis_count
                    total_reference_ngrams_count += reference_count
                    total_ngrams_overlapping_count += overlapping_ngrams

                score = RougeExt._compute_p_r_f_score(
                    total_hypothesis_ngrams_count,
                    total_reference_ngrams_count,
                    total_ngrams_overlapping_count,
                    self.alpha,
                    self.weight_factor,
                )

                for stat in RougeExt.STATS:
                    scores[metric][stat] += score[stat]
            else:
                # Best model
                if self.apply_best:
                    best_current_score = None
                    best_current_score_wlcs = None
                    for reference_sentences in references_sentences:
                        n_grams_counts = RougeExt._compute_ngrams_lcs(
                            hypothesis_sentences,
                            reference_sentences,
                            self.weight_factor if use_w else 1.0,
                        )
                        hypothesis_count, reference_count, overlapping_ngrams = n_grams_counts
                        score = RougeExt._compute_p_r_f_score(
                            hypothesis_count,
                            reference_count,
                            overlapping_ngrams,
                            self.alpha,
                            self.weight_factor,
                        )

                        if use_w:
                            reference_count_for_score = reference_count ** (
                                1.0 / self.weight_factor
                            )
                            overlapping_ngrams_for_score = overlapping_ngrams
                            score_wlcs = (
                                overlapping_ngrams_for_score / reference_count_for_score
                            ) ** (1.0 / self.weight_factor)

                            if (
                                best_current_score_wlcs is None
                                or score_wlcs > best_current_score_wlcs
                            ):
                                best_current_score = score
                                best_current_score_wlcs = score_wlcs
                        else:
                            if best_current_score is None or score["r"] > best_current_score["r"]:
                                best_current_score = score

                    for stat in RougeExt.STATS:
                        scores[metric][stat] += best_current_score[stat]
                # Keep all
                else:
                    for reference_sentences in references_sentences:
                        n_grams_counts = RougeExt._compute_ngrams_lcs(
                            hypothesis_sentences,
                            reference_sentences,
                            self.weight_factor if use_w else 1.0,
                        )
                        hypothesis_count, reference_count, overlapping_ngrams = n_grams_counts
                        score = RougeExt._compute_p_r_f_score(
                            hypothesis_count,
                            reference_count,
                            overlapping_ngrams,
                            self.alpha,
                            self.weight_factor,
                        )

                        for stat in RougeExt.STATS:
                            scores[metric][sample_id][stat].append(score[stat])

        # Compute final score with the average or the the max
        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for stat in RougeExt.STATS:
                scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _preprocess_summary_as_a_whole(self, summary):
        """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering)
        of a summary as a whole

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
        sentences = self.split_into_sentences(summary)

        # Truncate
        if self.limit_length:
            # By words
            if self.length_limit_type == "words":
                summary = " ".join(sentences)
                all_tokens = summary.split()  # Counting as in the perls script
                summary = " ".join(all_tokens[: self.length_limit])

            # By bytes
            elif self.length_limit_type == "bytes":
                summary = ""
                current_len = 0
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)

                    if current_len + sentence_len < self.length_limit:
                        if current_len != 0:
                            summary += " "
                        summary += sentence
                        current_len += sentence_len
                    else:
                        if current_len > 0:
                            summary += " "
                        summary += sentence[: self.length_limit - current_len]
                        break
        else:
            summary = " ".join(sentences)

        # summary = Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary.lower()).strip()
        summary = self.remove_char_pattern.sub(" ", summary.lower()).strip()

        # # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot" and
        #   "can not" as "can not",
        # # we have to hack nltk tokenizer to not transform "cannot/can not" to "can not"
        # if self.ensure_compatibility:
        #     tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', summary))
        # else:
        #     tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary))

        # if self.stemming:
        #     self.stem_tokens(tokens) # stemming in-place

        # if self.ensure_compatibility:
        #     preprocessed_summary = [Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub(
        #         'cannot', ' '.join(tokens))]
        # else:
        #     preprocessed_summary = [' '.join(tokens)]

        # return preprocessed_summary

        tokens = self.tokenize_text(summary)
        if self.stemming:
            self.stem_tokens(tokens)  # stemming in-place
        summary = [" ".join(tokens)]

        return summary

    def _preprocess_summary_per_sentence(self, summary):
        """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering)
        of a summary by sentences

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
        sentences = self.split_into_sentences(summary)

        # Truncate
        if self.limit_length:
            final_sentences = []
            current_len = 0
            # By words
            if self.length_limit_type == "words":
                for sentence in sentences:
                    tokens = sentence.strip().split()
                    tokens_len = len(tokens)
                    if current_len + tokens_len < self.length_limit:
                        sentence = " ".join(tokens)
                        final_sentences.append(sentence)
                        current_len += tokens_len
                    else:
                        sentence = " ".join(tokens[: self.length_limit - current_len])
                        final_sentences.append(sentence)
                        break
            # By bytes
            elif self.length_limit_type == "bytes":
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)
                    if current_len + sentence_len < self.length_limit:
                        final_sentences.append(sentence)
                        current_len += sentence_len
                    else:
                        sentence = sentence[: self.length_limit - current_len]
                        final_sentences.append(sentence)
                        break
            sentences = final_sentences

        final_sentences = []
        for sentence in sentences:
            # sentence = Rouge.REMOVE_CHAR_PATTERN.sub(' ', sentence.lower()).strip()
            sentence = self.remove_char_pattern.sub(" ", sentence.lower()).strip()

            #     # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot"
            #       and "can not" as "can not",
            #     # we have to hack nltk tokenizer to not transform "cannot/can not" to "can not"
            #     if self.ensure_compatibility:
            #         tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub(
            #             '_cannot_', sentence))
            #     else:
            #         tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', sentence))

            #     if self.stemming:
            #         self.stem_tokens(tokens) # stemming in-place

            #     if self.ensure_compatibility:
            #         sentence = Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub(
            #             'cannot', ' '.join(tokens)
            #         )
            #     else:
            #         sentence = ' '.join(tokens)

            tokens = self.tokenize_text(sentence)
            if self.stemming:
                self.stem_tokens(tokens)  # stemming in-place
            sentence = " ".join(tokens)
            final_sentences.append(sentence)

        return final_sentences
