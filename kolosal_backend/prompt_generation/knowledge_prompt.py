"""Default prompt for knowledge dataset generation"""

CONVERSATION_STARTER_PROMPT = """Conversation starter instruction: {instruction}
Relevant document: {document}
"""

CONVERSATION_SYSTEM_PROMPT = """Your main instruction is as follows: {instruction}
Relevant document: {document}"""

NEXT_QUESTION_SAME_TOPIC_PROMPT = """Based on this chat history between the user and AI, your task is to generate a followup question that a user might ask
{chat_history}

Previous AI Response: {response}

Your generated question might be related to the previous response or the chat history. You can also ask a question that is relevant to the context of the conversation.
For your information, the chat history is related or answered using this given information:
{document}
"""

NEXT_QUESTION_DIFFERENT_TOPIC_PROMPT = """Based on this chat history between the user and AI, your task is to generate a followup question that a user might ask
{chat_history}

Previous AI Response: {response}

Your generated question should be related to other topics or questions from this given document:
{document}
"""