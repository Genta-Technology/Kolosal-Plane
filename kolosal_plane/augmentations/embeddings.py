"""Dataset augmentation for Embedding QnA"""
from typing import List, Tuple

import polars as pl
from tqdm import tqdm
from distilabel.models.llms.base import AsyncLLM
from distilabel.steps.tasks import SelfInstruct


class EmbeddingAugmentation():
    documents: List[str]
    instruction: str
    lm: AsyncLLM
    question_per_document: int
    batch_size: int

    def __init__(self,
                 documents: List[str],
                 instruction: str,
                 lm: AsyncLLM,
                 question_per_document: int = 10,
                 batch_size: int = 10
                 ):
        self.documents = documents
        self.instruction = instruction
        self.lm = lm
        self.question_per_document = question_per_document
        self.batch_size = batch_size

    def augmentate(self) -> Tuple[pl.DataFrame, int, int]:
        """
        Augment the dataset by generating questions from documents using a language model.
        This method processes each document in the dataset and generates questions
        based on built instructions using a SelfInstruct generator and the provided
        language model. The questions are generated in batches for efficiency.
        Returns:
            pl.DataFrame: A DataFrame containing generated questions and their source documents,
                         with columns 'question' (text) and 'document' (text).
        Note:
            This method relies on several instance attributes:
            - self.lm: The language model used for generation
            - self.batch_size: Size of batches for processing
            - self.documents: Collection of documents to process
            - self.question_per_document: Number of questions to generate per document
            - self.build_instruction: Method to create instructions from documents
        """
        input_token_count = 0
        output_token_count = 0

        augmented_data = pl.DataFrame(
            schema={"question": pl.Utf8, "document": pl.Utf8})

        # Initiate the SelfInstruct instance
        generator = SelfInstruct(
            llm=self.lm,
            num_instructions=self.batch_size
        )

        # Load necessary resources for the generator
        generator.load()

        # Generate data for each document
        for document in tqdm(self.documents):
            # Build instruction for the document
            built_instruction = self.build_instruction(document)

            # Generate question per batch, for the document
            result = next(generator.process(
                [{"input": built_instruction}] * int(self.question_per_document / self.batch_size)))

            input_token_count += sum(res.get("distilabel_metadata", {}).get("statistics_self_instruct_0", {}).get("input_tokens", 0) for res in result)
            output_token_count += sum(res.get("distilabel_metadata", {}).get("statistics_self_instruct_0", {}).get("output_tokens", 0) for res in result)

            result = [instruction for res in result for instruction in res.get(
                "instructions", [])]

            # Append the generated questions to the augmented data
            new_rows = pl.DataFrame({
                "question": result,
                "document": [document] * len(result)
            })

            augmented_data = pl.concat(
                [augmented_data, new_rows], how="vertical")

        return augmented_data, input_token_count, output_token_count

    def build_instruction(self, document: str):
        # TODO: Implement the instruction builder based on prompt
        return f"{self.instruction} {document}"
