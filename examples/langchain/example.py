import os

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = ''

if __name__ == '__main__':

    llm = OpenAI(temperature=0.9)  # 大的 temperature 会让输出有更多的随机性
    text = 'what is the results of 5+6?'
    print(llm(text))  # 返回 11
    text = 'what is the results of 55+66?'
    print(llm(text))  # 返回 121
    text = 'what is the results of 55555+66666?'
    print(llm(text))  # 返回 122221
    text = 'what is the results of 512311+89749878?'
    print(llm(text))  # 返回 89,876,189，终于错了...

    text = 'what word is similar to good?'
    print(llm(text))  # 返回 Excellent
    text = 'what word is homophone of good?'
    print(llm(text))  # 返回 Goo

    prompt = PromptTemplate(
        input_variables=['product'],
        template='What is a good name for a company that makes {product}?',
    )
    print(prompt.format(product='colorful socks')
          )  # 返回 What is a good name for a company that makes colorful socks?
    text = prompt.format(product='colorful socks')
    print(llm(text))  # 返回 Socktastic！
    text = prompt.format(product='chocolates')
    print(llm(text))  # 返回 ChocoDelightz！

    llm = OpenAI(temperature=.7)
    template = """You are a teacher in physics for High School student. Given the text of question, \
        it is your job to write a answer that question with example.
    Question: {text}
    Answer:
    """
    prompt_template = PromptTemplate(input_variables=['text'],
                                     template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = answer_chain.run(
        'What is the formula for Gravitational Potential Energy (GPE)?')
    print(answer)

    # chatbot
    template = """You are a teacher in physics for High School student. Given the text of question, \
        it is your job to write a answer that question with example.
    {chat_history}
    Human: {question}
    AI:
    """
    prompt_template = PromptTemplate(
        input_variables=['chat_history', 'question'], template=template)
    memory = ConversationBufferMemory(memory_key='chat_history')

    llm_chain = LLMChain(
        llm=OpenAI(),
        prompt=prompt_template,
        verbose=True,
        memory=memory,
    )
    llm_chain.predict(
        question='What is the formula for Gravitational Potential Energy (GPE)?'
    )
    result = llm_chain.predict(question='What is Joules?')
    print(result)
