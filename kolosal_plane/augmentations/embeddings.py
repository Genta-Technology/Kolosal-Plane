"""Dataset augmentation for Embedding QnA (Asynchronous version)"""
import asyncio
from typing import List, Tuple, Dict

import polars as pl
from tqdm import tqdm
from distilabel.models.llms.base import AsyncLLM
from distilabel.steps.tasks import SelfInstruct


class EmbeddingAugmentation():
    documents: List[str]
    instruction_positive: str
    instruction_negative: str
    lm: AsyncLLM
    question_per_document: int
    batch_size: int

    def __init__(self,
                 documents: List[str],
                 instruction_positive: str,
                 instruction_negative: str,
                 lm: AsyncLLM,
                 question_per_document: int = 10,
                 batch_size: int = 10
                 ):
        self.documents = documents
        self.instruction_positive = instruction_positive
        self.instruction_negative = instruction_negative
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
            schema={"question_positive": pl.Utf8, 
                    "question_negative": pl.Utf8,
                    "document": pl.Utf8})

        # Initiate the SelfInstruct instance
        generator = SelfInstruct(
            llm=self.lm,
            num_instructions=self.batch_size
        )
        generator.load()

        # Generate data for each document
        for document in tqdm(self.documents, desc="Augmenting documents"):
            batch_count = int(self.question_per_document / self.batch_size)

            # Generate related questions in batches for this document
            built_positive_instruction = self.build_positive_instruction(
                document)

            result = next(generator.process(
                [{"input": built_positive_instruction}] * batch_count))
            input_token_count += sum(
                res.get("distilabel_metadata", {}).get(
                    "statistics_self_instruct_0", {}).get("input_tokens", 0)
                for res in result)
            output_token_count += sum(
                res.get("distilabel_metadata", {}).get(
                    "statistics_self_instruct_0", {}).get("output_tokens", 0)
                for res in result)

            questions_positive = [
                instruction for res in result for instruction in res.get("instructions", [])]

            # Generate unrelated questions in batches for this document
            built_negative_instruction = self.build_negative_instruction(
                document)

            result = next(generator.process(
                [{"input": built_negative_instruction}] * batch_count))
            input_token_count += sum(
                res.get("distilabel_metadata", {}).get(
                    "statistics_self_instruct_0", {}).get("input_tokens", 0)
                for res in result)
            output_token_count += sum(
                res.get("distilabel_metadata", {}).get(
                    "statistics_self_instruct_0", {}).get("output_tokens", 0)
                for res in result)

            questions_negative = [
                instruction for res in result for instruction in res.get("instructions", [])]

            # Add the additional rows to the augmented data
            new_rows = pl.DataFrame({
                "question_positive": questions_positive,
                "question_negative": questions_negative,
                "document": [document] * len(questions_positive)
            })
            augmented_data = pl.concat(
                [augmented_data, new_rows], how="vertical")

        metadata = {"input_token_count": input_token_count,
                    "output_token_count": output_token_count}
        return augmented_data, metadata

    def build_positive_instruction(self, document: str) -> str:
        """
        Build a positive instruction by applying a template to a document.
        This method constructs a complete instruction for positive examples by combining
        the positive instruction template with the provided document.
        Parameters
        ----------
        document : str
            The document text to be incorporated into the instruction.
        Returns
        -------
        str
            The complete positive instruction with the document integrated into it.
        See Also
        --------
        build_instruction : The base method used for constructing instructions.
        """

        return self.build_instruction(self.instruction_positive, document)

    def build_negative_instruction(self, document: str) -> str:
        """
        Builds a negative instruction by applying the negative instruction template to the provided document.
        This method creates a formatted instruction using the negative template stored in `self.instruction_negative`
        and the given document content.
        Args:
            document (str): The document content to include in the negative instruction.
        Returns:
            str: The formatted negative instruction string.
        """

        return self.build_instruction(self.instruction_negative, document)

    def build_instruction(self, instruction: str, document: str) -> str:
        """
        Builds an instruction by combining an instruction prompt with a document content.
        This method concatenates the instruction and document strings with a space in between.
        Useful for formatting inputs to language models or other text processing systems.
        Args:
            instruction (str): The instruction or prompt to be included.
            document (str): The document or content text to be processed.
        Returns:
            str: A combined string containing the instruction followed by the document.
        Example:
            >>> build_instruction("Summarize the following:", "This is a long text...")
            "Summarize the following: This is a long text..."
        """

        # TODO: Customize the instruction builder as needed.
        return f"{instruction} {document}"


class AsyncEmbeddingAugmentation(EmbeddingAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prepare an empty DataFrame with the appropriate schema.
        self._augmented_data: pl.DataFrame = pl.DataFrame(
            schema={"question": pl.Utf8, "document": pl.Utf8})
        self._metadata: Dict[str, int] = {
            "input_token_count": 0, "output_token_count": 0}
        self._status: str = "Not started"
        self._task: asyncio.Task = None
        self._lock = asyncio.Lock()  # Add lock for thread safety

    async def _run_sync_function(self, func, *args, **kwargs):
        """Helper method to run synchronous functions in the running event loop"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def augmentate_async(self) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Asynchronously augment the dataset by generating questions for each document.
        Uses SelfInstruct to generate questions in batches for each document, updating
        token counts and appending the generated rows to the dataset.
        Returns:
            Tuple containing the augmented polars DataFrame and a metadata dictionary.
        """
        try:
            self._status = "Running"

            # Initialize the SelfInstruct generator and load its resources.
            generator = SelfInstruct(
                llm=self.lm,
                num_instructions=self.batch_size
            )
            # Run load() in a thread to avoid blocking
            await self._run_sync_function(generator.load)

            for document in tqdm(self.documents, desc="Augmenting documents"):
                await asyncio.sleep(0)  # Yield control
                batch_count = int(self.question_per_document / self.batch_size)

                # Generate related questions in batches for this document
                built_positive_instruction = self.build_positive_instruction(
                    document)

                inputs = [{"input": built_positive_instruction}] * batch_count

                # Run the generator in a thread to avoid event loop conflicts
                result = await self._run_sync_function(lambda: next(generator.process(inputs)))

                # Update metadata with token counts (thread-safe)
                async with self._lock:
                    self._metadata["input_token_count"] += sum(
                        res.get("distilabel_metadata", {}).get(
                            "statistics_self_instruct_0", {}).get("input_tokens", 0)
                        for res in result)
                    self._metadata["output_token_count"] += sum(
                        res.get("distilabel_metadata", {}).get(
                            "statistics_self_instruct_0", {}).get("output_tokens", 0)
                        for res in result)

                questions_positive = [instruction for res in result for instruction in res.get(
                    "instructions", [])]

                # Generate unrelated questions in batches for this document
                built_negative_instruction = self.build_negative_instruction(
                    document)

                inputs = [{"input": built_negative_instruction}] * batch_count

                # Run the generator in a thread to avoid event loop conflicts
                result = await self._run_sync_function(lambda: next(generator.process(inputs)))

                # Update metadata with token counts (thread-safe)
                async with self._lock:
                    self._metadata["input_token_count"] += sum(
                        res.get("distilabel_metadata", {}).get(
                            "statistics_self_instruct_0", {}).get("input_tokens", 0)
                        for res in result)
                    self._metadata["output_token_count"] += sum(
                        res.get("distilabel_metadata", {}).get(
                            "statistics_self_instruct_0", {}).get("output_tokens", 0)
                        for res in result)

                questions_negative = [instruction for res in result for instruction in res.get(
                    "instructions", [])]

                new_rows = pl.DataFrame({
                    "question_positive": questions_positive,
                    "question_negative": questions_negative,
                    "document": [document] * len(questions_positive)
                })

                # Thread-safe update of the augmented data
                async with self._lock:
                    self._augmented_data = pl.concat(
                        [self._augmented_data, new_rows], how="vertical")

            self._status = "Finished"
            return self._augmented_data, self._metadata

        except asyncio.CancelledError:
            self._status = "Cancelled"
            raise
        except Exception as e:
            self._status = f"Failed: {str(e)}"
            raise

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

    def is_running(self) -> bool:
        """Check if the augmentation task is currently running"""
        return self._task is not None and not self._task.done() and self._status == "Running"
