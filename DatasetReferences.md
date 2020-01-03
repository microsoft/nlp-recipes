MICROSOFT PROVIDES THE DATASETS ON AN "AS IS" BASIS. MICROSOFT MAKES NO WARRANTIES, EXPRESS OR IMPLIED, GUARANTEES OR CONDITIONS WITH RESPECT TO YOUR USE OF THE DATASETS. TO THE EXTENT PERMITTED UNDER YOUR LOCAL LAW, MICROSOFT DISCLAIMS ALL LIABILITY FOR ANY DAMAGES OR LOSSES, INLCUDING DIRECT, CONSEQUENTIAL, SPECIAL, INDIRECT, INCIDENTAL OR PUNITIVE, RESULTING FROM YOUR USE OF THE DATASETS.

The datasets are provided under the original terms that Microsoft received such datasets. See below for more information about each dataset.

### <a name="cnndm"></a> CNN/Daily Mail (CNN/DM) Dataset
The training and evaluation for CNN/DM  dataset is available https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz and released under MIT License. This is a processed version of data that's originally released by Hermann et al. (2015) in ["Teaching machines to read and comprehend"](https://arxiv.org/abs/1506.03340) and then made available by Kyunghyun Cho at https://cs.nyu.edu/~kcho/DMQA/.

### Preprocessed CNN/Daily Mail (CNN/DM) Dataset by BERTSUM
The preprocessed dataset of [CNN/DM dataset](#cnndm), originally published by BERTSUM paper ["Fine-tune BERT for Extractive Summarization"](https://arxiv.org/pdf/1903.10318.pdf), can be found at https://github.com/nlpyang/BertSum and released under Apache License 2.0.


### Microsoft Research Paraphrase Corpus
>Original source: https://www.microsoft.com/en-us/download/details.aspx?id=52398


### The Multi-Genre NLI Corpus (MultiNLI)
>The majority of the corpus is released under the [OANC](https://www.anc.org/OANC/license.txt)’s license, The data in the FICTION section falls under several permissive licenses. See the [data description paper](https://www.nyu.edu/projects/bowman/multinli/paper.pdf) for details.
Redistributing the datasets "MultiNLI 1.0.zip", "MultiNLI Matched.zip", and "MultiNLI Mismatched.zip" with attribution:
Adina Williams, Nikita Nangia, Samuel R. Bowman. 2018. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers).
Original source: https://www.nyu.edu/projects/bowman/multinli/

### The Stanford Natural Language Inference (SNLI) Corpus
>This dataset is provided under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Redistributing the dataset "snli_1.0.zip" with attribution:
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).
Original source: https://nlp.stanford.edu/projects/snli/
The dataset is preprocessed to remove unused columns and badly formatted rows.

### Wikigold dataset
>This dataset is provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ast).
Redistributing the dataset "wikigold.conll.txt" with attribution:
Balasuriya, Dominic, et al. "Named entity recognition in wikipedia."
Proceedings of the 2009 Workshop on The People's Web Meets NLP: Collaboratively Constructed Semantic Resources. Association for Computational Linguistics, 2009.
Original source: https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold/CONLL-format/data
The dataset is preprocessed to fit data format requirement of BERT.

### The Cross-Lingual NLI Corpus (XNLI)
>The majority of the corpus sentences are released under the [OANC](https://www.anc.org/OANC/license.txt)’s license. The data in the Fiction genre from Captain Blood are under [The_Project_Gutenberg_License](http://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License). See details in the [XNLI paper](https://arxiv.org/pdf/1809.05053.pdf).
Redistributing the datasets "XNLI 1.0.zip" and "XNLI-MT 1.0.zip" with attribution:
Alexis Conneau, Guillaume Lample, Ruty Rinott, Holger Schwenk, Ves Stoyanov. 2018. XNLI: Evaluating Cross-lingual Sentence Representations. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.
Original source: https://www.nyu.edu/projects/bowman/xnli/
The dataset is preprocessed to remove unused columns.

### The Stanford Question Answering Dataset (SQuAD)
>This dataset is provided under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
Redistributing the datasets "train-v1.1.json" and "dev-v1.1.json" with attribution:
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP).
Original source: https://github.com/rajpurkar/SQuAD-explorer


### The STSbenchmark dataset
>Redistributing the dataset "Stsbenchmark.tar.gz" with attribution:
Eneko Agirre, Daniel Cer, Mona Diab, Iñigo Lopez-Gazpio, Lucia
 Specia. Semeval-2017 Task 1: Semantic Textual Similarity
 Multilingual and Crosslingual Focused Evaluation. Proceedings of
 SemEval 2017.
 Orignal source:http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
 The dataset is preprocessed to remove unused columns.
>The scores are released under [Commons Attribution - Share Alike 4.0
International License](http://creativecommons.org/licenses/by-sa/4.0/)
> The text of each dataset has a license of its own, as follows:

>- MSR-Paraphrase, Microsoft Research Paraphrase Corpus. In order to use
  MSRpar, researchers need to agree with the license terms from
  Microsoft Research:
  http://research.microsoft.com/en-us/downloads/607d14d9-20cd-47e3-85bc-a2f65cd28042/

>- headlines: Mined from several news sources by European Media Monitor
  (Best et al. 2005). using the RSS feed. European Media Monitor (EMM)
  Real Time News Clusters are the top news stories for the last 4
  hours, updated every ten minutes. The article clustering is fully
  automatic. The selection and placement of stories are determined
  automatically by a computer program. This site is a joint project of
  DG-JRC and DG-COMM. The information on this site is subject to a
  disclaimer (see
  http://europa.eu/geninfo/legal_notices_en.htm). Please acknowledge
  EMM when (re)using this material.
  http://emm.newsbrief.eu/rss?type=rtn&language=en&duplicates=false

>- deft-news: A subset of news article data in the DEFT
  project.

>- MSR-Video, Microsoft Research Video Description Corpus.  In order to
  use MSRvideo, researchers need to agree with the license terms from
  Microsoft Research:
  http://research.microsoft.com/en-us/downloads/38cf15fd-b8df-477e-a4e4-a4680caa75af/

>- image: The Image Descriptions data set is a subset of
  the PASCAL VOC-2008 data set (Rashtchian et al., 2010) . PASCAL
  VOC-2008 data set consists of 1,000 images and has been used by a
  number of image description systems. The image captions of the data
  set are released under a CreativeCommons Attribution-ShareAlike
  license, the descriptions itself are free.

>- track5.en-en: This text is a subset of the Stanford Natural
  Language Inference (SNLI) corpus, by The Stanford NLP Group is
  licensed under a Creative Commons Attribution-ShareAlike 4.0
  International License. Based on a work at
  http://shannon.cs.illinois.edu/DenotationGraph/.
  https://creativecommons.org/licenses/by-sa/4.0/

>- answers-answers: user content from stack-exchange. Check the license
  below in ======ANSWERS-ANSWERS======

>- answers-forums: user content from stack-exchange. Check the license
  below in ======ANSWERS-FORUMS======



>======ANSWER-ANSWER======

>Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)
http://creativecommons.org/licenses/by-sa/3.0/

>Attribution Requirements:

>   "* Visually display or otherwise indicate the source of the content
      as coming from the Stack Exchange Network. This requirement is
      satisfied with a discreet text blurb, or some other unobtrusive but
      clear visual indication.

>    * Ensure that any Internet use of the content includes a hyperlink
      directly to the original question on the source site on the Network
      (e.g., http://stackoverflow.com/questions/12345)

>    * Visually display or otherwise clearly indicate the author names for
      every question and answer used

>    * Ensure that any Internet use of the content includes a hyperlink for
      each author name directly back to his or her user profile page on the
      source site on the Network (e.g.,
      http://stackoverflow.com/users/12345/username), directly to the Stack
      Exchange domain, in standard HTML (i.e. not through a Tinyurl or other
      such indirect hyperlink, form of obfuscation or redirection), without
      any “nofollow” command or any other such means of avoiding detection by
      search engines, and visible even with JavaScript disabled."

>    (https://archive.org/details/stackexchange)



>======ANSWERS-FORUMS======


>Stack Exchange Inc. generously made the data used to construct the STS 2015 answer-answer statement pairs available under a Creative Commons Attribution-ShareAlike (cc-by-sa) 3.0 license.

>The license is reproduced below from: https://archive.org/details/stackexchange

>The STS.input.answers-forums.txt file should be redistributed with this LICENSE text and the accompanying files in LICENSE.answers-forums.zip. The tsv files in the zip file contain the additional information that's needed to comply with the license.

>--

>All user content contributed to the Stack Exchange network is cc-by-sa 3.0 licensed, intended to be shared and remixed. We even provide all our data as a convenient data dump.

>http://creativecommons.org/licenses/by-sa/3.0/

>But our cc-by-sa 3.0 licensing, while intentionally permissive, does *require attribution*:

>"Attribution — You must attribute the work in the manner specified by the author or licensor (but not in any way that suggests that they endorse you or your use of the work)."

>Specifically the attribution requirements are as follows:

>  1. Visually display or otherwise indicate the source of the content as coming from the Stack Exchange Network. This requirement is satisfied with a discreet text blurb, or some other unobtrusive but clear visual indication.
>  2. Ensure that any Internet use of the content includes a hyperlink directly to the original question on the source site on the Network (e.g., http://stackoverflow.com/questions/12345)

>  3. Visually display or otherwise clearly indicate the author names for every question and answer so used.

>  4. Ensure that any Internet use of the content includes a hyperlink for each author name directly back to his or her user profile page on the source site on the Network (e.g., http://stackoverflow.com/users/12345/username), directly to the Stack Exchange domain, in standard HTML (i.e. not through a Tinyurl or other such indirect hyperlink, form of obfuscation or redirection), without any “nofollow” command or any other such means of avoiding detection by search engines, and visible even with JavaScript disabled.

>Our goal is to maintain the spirit of fair attribution. That means attribution to the website, and more importantly, to the individuals who so generously contributed their time to create that content in the first place!

>For more information, see the Stack Exchange Terms of Service: http://stackexchange.com/legal/terms-of-service
