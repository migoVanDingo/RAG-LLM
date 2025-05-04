from langchain.prompts import PromptTemplate

def prompt_template(context: str, question: str):
    try:
        print("==== Generating Prompt ====")

        template = """
        <s>[INST]
        Given the following context, answer the question.
        {context}
        Question: {question}
        </s>
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        prompt_str = prompt.format(context=context, question=question)

        print(f"Prompt: {prompt_str}\n\n")
        
        return prompt
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    context = "This is a test context"
    question = "What is the capital of France?"
    prompt_template(context, question)
    