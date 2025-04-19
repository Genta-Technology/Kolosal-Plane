"""Dataset augmentation for Embedding QnA (Asynchronous version)"""
import asyncio
from typing import List, Tuple, Dict

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

    def augmentate(self) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Augment the dataset by generating questions from documents using a language model.
        This method processes each document in the dataset and generates questions
        based on built instructions using a SelfInstruct generator and the provided
        language model. The questions are generated in batches for efficiency.
        Returns:
            pl.DataFrame: A DataFrame containing generated questions and their source documents,
                         with columns 'question' (text) and 'document' (text).
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
        generator.load()

        # Generate data for each document
        for document in tqdm(self.documents, desc="Augmenting documents"):
            built_instruction = self.build_instruction(document)
            # Generate questions in batches for this document
            batch_count = int(self.question_per_document / self.batch_size)
            result = next(generator.process(
                [{"input": built_instruction}] * batch_count))
            input_token_count += sum(
                res.get("distilabel_metadata", {}).get("statistics_self_instruct_0", {}).get("input_tokens", 0)
                for res in result)
            output_token_count += sum(
                res.get("distilabel_metadata", {}).get("statistics_self_instruct_0", {}).get("output_tokens", 0)
                for res in result)

            questions = [instruction for res in result for instruction in res.get("instructions", [])]
            new_rows = pl.DataFrame({
                "question": questions,
                "document": [document] * len(questions)
            })
            augmented_data = pl.concat([augmented_data, new_rows], how="vertical")

        metadata = {"input_token_count": input_token_count,
                    "output_token_count": output_token_count}
        return augmented_data, metadata

    def build_instruction(self, document: str) -> str:
        # TODO: Customize the instruction builder as needed.
        return f"{self.instruction} {document}"


class AsyncEmbeddingAugmentation(EmbeddingAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prepare an empty DataFrame with the appropriate schema.
        self._augmented_data: pl.DataFrame = pl.DataFrame(
            schema={"question": pl.Utf8, "document": pl.Utf8})
        self._metadata: Dict[str, int] = {"input_token_count": 0, "output_token_count": 0}
        self._status: str = "Not started"
        self._task: asyncio.Task = None

    async def augmentate_async(self) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Asynchronously augment the dataset by generating questions for each document.
        Uses SelfInstruct to generate questions in batches for each document, updating
        token counts and appending the generated rows to the dataset.
        Returns:
            Tuple containing the augmented polars DataFrame and a metadata dictionary.
        """
        self._status = "Running"

        # Initialize the SelfInstruct generator and load its resources.
        generator = SelfInstruct(
            llm=self.lm,
            num_instructions=self.batch_size
        )
        # If load() is blocking, consider running it in a thread.
        await asyncio.to_thread(generator.load)

        for document in tqdm(self.documents, desc="Augmenting documents"):
            await asyncio.sleep(0)  # Yield control
            built_instruction = self.build_instruction(document)
            batch_count = int(self.question_per_document / self.batch_size)
            inputs = [{"input": built_instruction}] * batch_count

            # Wrap the blocking generator call in a thread to avoid blocking the event loop.
            result = await asyncio.to_thread(lambda: next(generator.process(inputs)))

            self._metadata["input_token_count"] += sum(
                res.get("distilabel_metadata", {}).get("statistics_self_instruct_0", {}).get("input_tokens", 0)
                for res in result)
            self._metadata["output_token_count"] += sum(
                res.get("distilabel_metadata", {}).get("statistics_self_instruct_0", {}).get("output_tokens", 0)
                for res in result)

            questions = [instruction for res in result for instruction in res.get("instructions", [])]
            new_rows = pl.DataFrame({
                "question": questions,
                "document": [document] * len(questions)
            })
            self._augmented_data = pl.concat([self._augmented_data, new_rows], how="vertical")

        self._status = "Finished"
        return self._augmented_data, self._metadata

    def start_augmentation(self) -> asyncio.Task:
        """
        Starts the augmentation process in the background.
        Returns:
            An asyncio.Task representing the augmentation process.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.augmentate_async())
        return self._task

    def get_result(self) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Returns the current augmented DataFrame and metadata.
        """
        return self._augmented_data, self._metadata

    def cancel_augmentation(self):
        """
        Cancels the running augmentation task if it is still in progress.
        """
        if self._task and not self._task.done():
            self._task.cancel()

    def get_status(self) -> Tuple[str, Dict[str, int]]:
        """
        Returns the current status of the augmentation process along with metadata.
        """
        return self._status, self._metadata
