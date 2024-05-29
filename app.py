from __future__ import annotations

from typing import AsyncIterable
# from dotenv import load_dotenv
import os

from fastapi import FastAPI
import fastapi_poe as fp
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class LangchainOpenAIChatBot(fp.PoeBot):
    def __init__(self):
        super().__init__()
        self.chat_model = ChatOpenAI(
          temperature=0,
        )

    async def get_response(
        self, request: fp.QueryRequest
    ) -> AsyncIterable[fp.PartialResponse]:
        print("request.query=" + str(request.query))
        messages = []
        for message in request.query:
            if message.role == "bot":
                messages.append(AIMessage(content=message.content))
            elif message.role == "system":
                messages.append(SystemMessage(content=message.content))
            elif message.role == "user":
                messages.append(HumanMessage(content=message.content))

        response = self.chat_model.invoke(messages)
        if isinstance(response.content, str):
            yield fp.PartialResponse(text=response.content)
        else:
            yield fp.PartialResponse(text="There was an issue processing your query.")

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(allow_attachments=True, enable_image_comprehension=True)


# load_dotenv()
app = FastAPI()

fp.make_app(LangchainOpenAIChatBot(), access_key=os.environ['POE_ACCESS_KEY'], app=app)

if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
    print(ChatOpenAI(model="gpt-4o-2024-05-13",temperature=0).invoke('test').content)

