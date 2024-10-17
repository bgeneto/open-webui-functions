"""
title: Chat Context Clipper (that works :-)
author: open-webui & bgeneto (several improvements)
author_url: https://github.com/bgeneto/open-webui-functions/blob/main/chat_context_clipper_function.py
funding_url: https://github.com/open-webui
version: 0.1.3
description: A filter that truncates chat history to retain the latest n-th user and assistant
             messages while always keeping the system prompt and also first message pair (if desired).
             It ensures that the first message (after the prompt if any) is a user message (Anthropic requirement).
             It also offers a user valve to set the number of messages to retain, which overrides the global setting.
"""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

DEBUG = False


def get_first_user_message(
    data: List[Dict[str, Union[str, dict]]]
) -> Optional[Dict[str, str]]:
    """
    Returns the first user message in the given data.

    Args:
        data (list): A list of dictionaries containing role and content.

    Returns:
        dict: The first user message.
    """
    for message in data:
        if message["role"] == "user":
            return message
    return None


def get_first_assistant_message(
    data: List[Dict[str, Union[str, dict]]]
) -> Optional[Dict[str, str]]:
    """
    Returns the first assistant message in the given data.

    Args:
        data (list): A list of dictionaries containing role and content.

    Returns:
        dict: The first assistant message.
    """
    for message in data:
        if message["role"] == "assistant":
            return message
    return None


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations"
        )
        n_last_messages: int = Field(
            default=4, description="Number of last messages to keep"
        )
        keep_first: bool = Field(
            default=True,
            description="Always Keep the first user message and assistant answer",
        )
        pass

    class UserValves(BaseModel):
        n_last_messages: int = Field(
            default=4,
            description="Number of last chat messages to keep in the assistant memory",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        pass

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        messages = body["messages"]

        # get n_last_messages from valves or user valves
        n_last_messages = int(
            self.user_valves.n_last_messages
            if self.user_valves.n_last_messages
            else self.valves.n_last_messages
        )

        # double the number of messages to keep to account for both user and assistant messages and...
        # ...also add one more pair of messages to account for keeping the first user and assistant message
        # this provide more meaningful context to some conversations
        n_last_messages = 2 * n_last_messages
        if self.valves.keep_first:
            n_last_messages = n_last_messages + 2

        # check if the number of messages is less than messages to keep (early exit)
        if len(messages) <= n_last_messages:
            return body

        if DEBUG:
            print("Original messages length:", len(messages))

        # Ensure we always keep the system prompt
        system_prompt = next(
            (message for message in messages if message.get("role") == "system"), None
        )

        # Always keep the first user message...
        first_user_message = get_first_user_message(messages)

        # ...along with its assistant response
        first_assistant_message = get_first_assistant_message(messages)

        recent_messages = [
            message for message in messages if message["role"] in ["user", "assistant"]
        ][-n_last_messages:]

        # Construct the new message list by appending the system prompt first (if any)
        new_messages = []
        if system_prompt:
            new_messages.append(system_prompt)

        # Check if we need to append the first couple of messages
        if self.valves.keep_first:
            if first_user_message:
                new_messages.append(first_user_message)
            if first_assistant_message:
                new_messages.append(first_assistant_message)

        # Ensure the sequence is system -> user -> assistant
        if (
            recent_messages
            and recent_messages[0]["role"] == "user"
            and len(recent_messages) > 1
        ):
            if recent_messages[1]["role"] == "user":
                recent_messages.pop(0)

        # remove/pop assistant message if it the first
        if (
            recent_messages
            and recent_messages[0]["role"] == "assistant"
            and len(recent_messages) > 1
        ):
            recent_messages.pop(0)
            if len(recent_messages) > 1 and recent_messages[0]["role"] == "assistant":
                recent_messages.pop(0)

        new_messages.extend(recent_messages)

        if DEBUG:
            print("Clipped messages length:", len(new_messages))

        # Update body messages
        body["messages"] = new_messages

        return body
