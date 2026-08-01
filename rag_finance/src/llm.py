import ollama


class LLM:

    def __init__(

        self,

        model="gemma3:4b",

        temperature=0,

    ):

        self.model = model

        self.temperature = temperature

    def generate(

        self,

        prompt,

    ):

        response = ollama.chat(

            model=self.model,

            messages=[

                {

                    "role": "user",

                    "content": prompt,

                }

            ],

            options={

                "temperature": self.temperature,

            },

        )

        return response["message"]["content"]