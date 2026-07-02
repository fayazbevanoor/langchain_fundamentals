from langchain_core.prompts import PromptTemplate

#template
template = PromptTemplate(
    template = """
Explain the Library "{library}" using the following specifications:
Explanation Level: {level}
Number of Lines: {size}
Instructions:
1. Tailor the explanation according to the selected level:
   - Beginner: Use simple language and basic examples.
   - Advanced: Include deeper concepts and practical usage.
   - Expert: Include technical details, best practices, and edge cases.
2. Focus on the selected library and provide relevant examples.
3. Keep the explanation within approximately {size} lines.
4. Use clear and concise language.
5. If the topic is not related to the selected library, explain it from a general programming perspective.
Ensure the response is accurate, easy to understand, and aligned with the selected level and length.
""",
input_variables=['library','level','size']
)

template.save('template.json')