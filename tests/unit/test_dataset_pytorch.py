from utils_nlp.models.transformers.datasets import QADataset


def test_QADataset(qa_test_df):
    dataset = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
        answer_start_col=qa_test_df["answer_start_col"],
        answer_text_col=qa_test_df["answer_text_col"],
        qa_id_col=qa_test_df["qa_id_col"],
        is_impossible_col=qa_test_df["is_impossible_col"],
    )

    for i in range(2):
        assert dataset[i].doc_text == qa_test_df["test_df"][qa_test_df["doc_text_col"]][i]
        assert dataset[i].question_text == qa_test_df["test_df"][qa_test_df["question_text_col"]][i]
        assert dataset[i].answer_start == qa_test_df["test_df"][qa_test_df["answer_start_col"]][i]
        assert dataset[i].answer_text == qa_test_df["test_df"][qa_test_df["answer_text_col"]][i]
        assert dataset[i].qa_id == qa_test_df["test_df"][qa_test_df["qa_id_col"]][i]
        assert dataset[i].is_impossible == qa_test_df["test_df"][qa_test_df["is_impossible_col"]][i]

    dataset_default = QADataset(
        df=qa_test_df["test_df"],
        doc_text_col=qa_test_df["doc_text_col"],
        question_text_col=qa_test_df["question_text_col"],
    )

    for i in range(2):
        assert dataset_default[i].doc_text == qa_test_df["test_df"][qa_test_df["doc_text_col"]][i]
        assert (
            dataset_default[i].question_text
            == qa_test_df["test_df"][qa_test_df["question_text_col"]][i]
        )
        assert dataset_default[i].answer_start == -1
        assert dataset_default[i].answer_text == ""
        assert dataset_default[i].qa_id == i
        assert dataset_default[i].is_impossible == False
