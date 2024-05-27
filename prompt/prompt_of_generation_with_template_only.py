GENERATION_WITH_TEMPLATE_ONLY_SYSTEM = "You are a document writer"

GENERATION_WITH_TEMPLATE_ONLY_PROMPT = """Your task is to write the current section based on the provided section requirement.
<Section Requirement>
{section_requirement}
</Section Requirement>

<Knowledge>
{knowledge}
<Knowledge>

Note: The content in the knowledge section may be entirely useful or only partially useful. 
You need to think critically and extract the relevant information to write a complete section based on the section requirement."""