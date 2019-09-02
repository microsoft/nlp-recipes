# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Download and tokenize data with MOSES tokenizer
#


""" adapted the original script to only download the STSBenchmark data"""
data_path=.
preprocess_exec=./tokenizer.sed

# Get MOSES
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
SCRIPTS=mosesdecoder/scripts
MTOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LOWER=$SCRIPTS/tokenizer/lowercase.perl

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

PTBTOKENIZER="sed -f tokenizer.sed"

mkdir $data_path

STSBenchmark='http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'


# STS 2012, 2013, 2014, 2015, 2016
declare -A STS_tasks
declare -A STS_paths
declare -A STS_subdirs

STS_tasks=(["STS12"]="MSRpar MSRvid SMTeuroparl surprise.OnWN surprise.SMTnews" ["STS13"]="FNWN headlines OnWN" ["STS14"]="deft-forum deft-news headlines OnWN images tweet-news" ["STS15"]="answers-forums answers-students belief headlines images" ["STS16"]="answer-answer headlines plagiarism postediting question-question")

STS_paths=(["STS12"]="http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip" ["STS13"]="http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip" ["STS14"]="http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip" ["STS15"]="http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip"
["STS16"]="http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip")

STS_subdirs=(["STS12"]="test-gold" ["STS13"]="test-gs" ["STS14"]="sts-en-test-gs-2014" ["STS15"]="test_evaluation_task2a" ["STS16"]="sts2016-english-with-gs-v1.0")




### STS datasets

# STS12, STS13, STS14, STS15, STS16
mkdir $data_path/STS

for task in "${!STS_tasks[@]}"; #"${!STS_tasks[@]}";
do
    fpath=${STS_paths[$task]}
    echo $fpath
    curl -Lo $data_path/STS/data_$task.zip $fpath
    unzip $data_path/STS/data_$task.zip -d $data_path/STS
    mv $data_path/STS/${STS_subdirs[$task]} $data_path/STS/$task-en-test
    rm $data_path/STS/data_$task.zip

    for sts_task in ${STS_tasks[$task]}
    do
        fname=STS.input.$sts_task.txt
        task_path=$data_path/STS/$task-en-test/

        if [ "$task" = "STS16" ] ; then
            echo 'Handling STS2016'
            mv $task_path/STS2016.input.$sts_task.txt $task_path/$fname
            mv $task_path/STS2016.gs.$sts_task.txt $task_path/STS.gs.$sts_task.txt
        fi



        cut -f1 $task_path/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp1
        cut -f2 $task_path/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp2
        paste $task_path/tmp1 $task_path/tmp2 > $task_path/$fname
        rm $task_path/tmp1 $task_path/tmp2
    done

done


# STSBenchmark (http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)

curl -Lo $data_path/Stsbenchmark.tar.gz $STSBenchmark
tar -zxvf $data_path/Stsbenchmark.tar.gz -C $data_path
rm $data_path/Stsbenchmark.tar.gz
mv $data_path/stsbenchmark $data_path/STS/STSBenchmark

for split in train dev test
do
    fname=sts-$split.csv
    fdir=$data_path/STS/STSBenchmark
    cut -f1,2,3,4,5 $fdir/$fname > $fdir/tmp1
    cut -f6 $fdir/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $fdir/tmp2
    cut -f7 $fdir/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $fdir/tmp3
    paste $fdir/tmp1 $fdir/tmp2 $fdir/tmp3 > $fdir/$fname
    rm $fdir/tmp1 $fdir/tmp2 $fdir/tmp3
done

