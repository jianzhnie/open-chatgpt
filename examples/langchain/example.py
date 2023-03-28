import os
from langchain.llms import OpenAI  # 导入 LLM wrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)

os.environ[
    "OPENAI_API_KEY"] = "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"

if __name__ == "__main__":

    llm = OpenAI(temperature=0.9)  # 大的 temperature 会让输出有更多的随机性
    text = "what is the results of 5+6?"
    print(llm(text))  # 返回 11
    text = "what is the results of 55+66?"
    print(llm(text))  # 返回 121
    text = "what is the results of 55555+66666?"
    print(llm(text))  # 返回 122221
    text = "what is the results of 512311+89749878?"
    print(llm(text))  # 返回 89,876,189，终于错了...

    text = "what word is similar to good?"
    print(llm(text))  # 返回 Excellent
    text = "what word is homophone of good?"
    print(llm(text))  # 返回 Goo

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    print(prompt.format(product="colorful socks")
          )  # 返回 What is a good name for a company that makes colorful socks?
    text = prompt.format(product="colorful socks")
    print(llm(text))  # 返回 Socktastic！
    text = prompt.format(product="chocolates")
    print(llm(text))  # 返回 ChocoDelightz！

    chain = LLMChain(llm=llm, prompt=prompt)
    chain.run("colorful socks")
    chat = ChatOpenAI(temperature=0)
    chat([
        HumanMessage(
            content=
            "Translate this sentence from English to French. I love programming."
        )
    ])

    messages = [
        SystemMessage(
            content=
            "You are a helpful assistant that translates English to French."),
        HumanMessage(
            content=
            "Translate this sentence from English to French. I love programming."
        )
    ]
    chat(messages)
