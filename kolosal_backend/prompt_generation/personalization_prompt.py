"""Default prompt for personalization dataset generation"""

NEXT_QUESTION_PROMPT = """Based on this chat_history between the user and AI, your task is to generate a followup question that a user might ask
{chat_history}

Previous AI Response: {response}

Your generated question might be related to the previous response or the chat history. You can also ask a question that is relevant to the context of the conversation."""
